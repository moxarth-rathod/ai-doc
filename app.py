"""
AIDoc - Streamlit UI
A beautiful web interface for the AI-powered documentation generator.

Launch via:
    python main.py ui
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

import json
import difflib
import html as html_lib

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load .env before importing our modules
load_dotenv()

from config.settings import AppConfig, LLMConfig
from core.loader import RepoLoader
from core.indexer import CodebaseIndexer
from core.analyzer import HierarchicalAnalyzer, HierarchicalContext
from core.planner import ProjectPlanner, ProjectAnalysis
from core.writer import DocumentationWriter
from core.evaluator import DocumentationEvaluator, CoverageReport, QualityRating
from core.chat import ChatEngine
from core.db import DocVersionStore, derive_project_id

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AIDoc — AI Documentation Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Theme-aware custom properties ─────── */
    :root {
        --pd-accent: #667eea;
        --pd-accent2: #764ba2;
        --pd-muted: #6b7280;
        --pd-border: rgba(128, 128, 128, 0.25);
        --pd-card-bg: rgba(128, 128, 128, 0.08);
    }

    /* ── Header ────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--pd-accent) 0%, var(--pd-accent2) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        opacity: 0.7;
        font-size: 1.05rem;
    }

    /* ── Stat boxes ────────────────────────── */
    .stat-box {
        text-align: center;
        padding: 0.8rem;
        background: var(--pd-card-bg);
        border-radius: 10px;
        border: 1px solid var(--pd-border);
    }
    .stat-box .number {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--pd-accent);
    }
    .stat-box .label {
        font-size: 0.82rem;
        opacity: 0.65;
        margin-top: 0.1rem;
    }

    /* ── Section tags ──────────────────────── */
    .section-tag {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .tag-developer { background: rgba(29,78,216,0.15); color: #60a5fa; }
    .tag-end-user  { background: rgba(180,83,9,0.15);  color: #fbbf24; }
    .tag-both      { background: rgba(99,102,241,0.15); color: #a5b4fc; }

    /* ── Footer hide ───────────────────────── */
    footer { visibility: hidden; }

    /* ── Divider ───────────────────────────── */
    .subtle-divider {
        border: none;
        border-top: 1px solid var(--pd-border);
        margin: 1.5rem 0;
    }

    /* ── Score card ────────────────────────── */
    .score-card {
        text-align: center;
        padding: 1rem 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--pd-border);
        background: var(--pd-card-bg);
    }
    .score-card .score-value {
        font-size: 2rem;
        font-weight: 800;
    }
    .score-card .score-label {
        font-size: 0.8rem;
        opacity: 0.65;
        margin-top: 0.2rem;
    }
    .score-good { color: #34d399; }
    .score-ok   { color: #fbbf24; }
    .score-low  { color: #f87171; }

    /* ── Stop button (red styling via key) ── */
    [data-testid="stButton"] button[kind="secondary"] {
        border-color: #ef4444 !important;
        color: #ef4444 !important;
        font-weight: 600;
    }
    [data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: rgba(239, 68, 68, 0.15) !important;
        border-color: #dc2626 !important;
        color: #dc2626 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "results" not in st.session_state:
    st.session_state.results = {}
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "refinement_messages" not in st.session_state:
    st.session_state.refinement_messages = []
if "doc_writer" not in st.session_state:
    st.session_state.doc_writer = None
if "doc_approved" not in st.session_state:
    st.session_state.doc_approved = False
if "_cleanup_loader" not in st.session_state:
    st.session_state._cleanup_loader = None
if "_cleanup_path" not in st.session_state:
    st.session_state._cleanup_path = None
if "_pending_pipeline" not in st.session_state:
    st.session_state._pending_pipeline = None
if "doc_version" not in st.session_state:
    st.session_state.doc_version = "1.0.0"
if "project_id" not in st.session_state:
    st.session_state.project_id = ""

# ── MongoDB Version Store (singleton) ──
# Recreate if missing or stale (e.g. code updated but old object in session)
if "_version_store" not in st.session_state or not hasattr(st.session_state._version_store, "db_name"):
    _mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    _mongo_db = os.environ.get("MONGO_DB", "aidoc")
    _vs = DocVersionStore(_mongo_uri, db_name=_mongo_db)
    _vs.connect()  # graceful — returns False if unavailable
    st.session_state._version_store = _vs


# ──────────────────────────────────────────────
# Resource cleanup helpers
# ──────────────────────────────────────────────

def _cleanup_cloned_repo():
    """
    Clean up only the cloned repository on disk.
    Called when the user approves the documentation — the in-memory index,
    chat engine, and writer stay alive for continued use.
    """
    loader = st.session_state.get("_cleanup_loader")
    path = st.session_state.get("_cleanup_path")
    if loader and path:
        loader.cleanup(path)
    st.session_state._cleanup_loader = None
    st.session_state._cleanup_path = None


def _cleanup_all_resources():
    """
    Clean up everything: cloned repo, in-memory index, chat engine, writer.
    Called when starting a new generation (replaces previous session entirely).
    """
    _cleanup_cloned_repo()
    st.session_state.chat_engine = None
    st.session_state.doc_writer = None


def _save_doc_version(
    change_type: str,
    description: str,
    markdown: str | None = None,
    full_markdown: str | None = None,
) -> str | None:
    """
    Save the current documentation to MongoDB as a new version.
    Returns the new version string, or None if DB is unavailable.
    """
    vs: DocVersionStore = st.session_state._version_store
    if not vs._connected:
        return None

    project_id = st.session_state.project_id
    md = markdown or st.session_state.results.get("markdown", "")
    full_md = full_markdown or st.session_state.results.get("full_markdown", "")

    if change_type == "initial":
        # Check if project already exists → treat as regenerated (major bump)
        existing = vs.get_latest_version(project_id)
        if existing:
            change_type = "regenerated"
            new_version = vs.bump_version(existing["version"], change_type)
        else:
            new_version = "1.0.0"
    else:
        current = st.session_state.doc_version
        new_version = vs.bump_version(current, change_type)

    vs.save_version(
        project_id=project_id,
        version=new_version,
        markdown=md,
        full_markdown=full_md,
        description=description,
        change_type=change_type,
    )
    st.session_state.doc_version = new_version
    return new_version


def _render_diff_html(old_text: str, new_text: str) -> str:
    """
    Build an HTML diff view comparing *old_text* → *new_text*.
    Added lines are green, removed lines are red, unchanged lines are dimmed.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")

    html_parts: list[str] = [
        "<style>"
        ".diff-wrap { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.82rem; "
        "line-height: 1.55; overflow-x: auto; padding: 0.8rem; border-radius: 8px; "
        "border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.04); }"
        ".diff-add  { background: rgba(46,160,67,0.18); color: #3fb950; }"
        ".diff-del  { background: rgba(248,81,73,0.18); color: #f85149; text-decoration: line-through; }"
        ".diff-hdr  { color: #58a6ff; font-weight: 600; }"
        ".diff-ctx  { opacity: 0.5; }"
        "</style>"
        '<div class="diff-wrap"><pre>'
    ]

    has_changes = False
    for line in diff:
        escaped = html_lib.escape(line.rstrip("\n"))
        if line.startswith("@@"):
            html_parts.append(f'<span class="diff-hdr">{escaped}</span>\n')
            has_changes = True
        elif line.startswith("+") and not line.startswith("+++"):
            html_parts.append(f'<span class="diff-add">+ {escaped[1:]}</span>\n')
            has_changes = True
        elif line.startswith("-") and not line.startswith("---"):
            html_parts.append(f'<span class="diff-del">- {escaped[1:]}</span>\n')
            has_changes = True
        elif line.startswith("---") or line.startswith("+++"):
            continue  # skip file headers
        else:
            html_parts.append(f'<span class="diff-ctx">  {escaped[1:]}</span>\n')

    html_parts.append("</pre></div>")

    if not has_changes:
        return (
            '<div style="text-align:center; padding:2rem; opacity:0.6;">'
            "No differences — content is identical to the previous version."
            "</div>"
        )

    return "".join(html_parts)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    env_key = os.environ.get("CEREBRAS_API_KEY", "")
    api_key = st.text_input(
        "Cerebras API Key",
        value=env_key,
        type="password",
        help="Your Cerebras API key. Auto-loaded from .env if set.",
    )
    if env_key:
        st.caption("✅ API key loaded from `.env`")

    st.divider()

    model = st.selectbox(
        "LLM Model",
        options=["llama-3.3-70b", "llama-3.1-8b"],
        index=0,
        help="Choose the Cerebras model for analysis.",
    )

    st.divider()

    # ── MongoDB status & config ──
    _vs_sidebar: DocVersionStore = st.session_state._version_store
    if _vs_sidebar._connected:
        st.caption("🟢 MongoDB connected — version history enabled")
        st.caption(f"📂 Database: `{_vs_sidebar.db_name}`")
    else:
        st.caption("🔴 MongoDB offline — version history disabled")

    with st.expander("🗄️ MongoDB Settings", expanded=not _vs_sidebar._connected):
        new_uri = st.text_input(
            "Connection URI",
            value=_vs_sidebar._uri,
            key="mongo_uri_input",
            help="MongoDB connection string. Also configurable via `MONGO_URI` in `.env`.",
        )
        new_db = st.text_input(
            "Database Name",
            value=_vs_sidebar.db_name,
            key="mongo_db_input",
            help="MongoDB database to store versions. Also configurable via `MONGO_DB` in `.env`.",
        )
        if st.button("🔄 Save & Reconnect", key="mongo_reconnect", use_container_width=True):
            _vs_sidebar._uri = new_uri.strip() or "mongodb://localhost:27017"
            _vs_sidebar.db_name = new_db.strip() or "aidoc"
            _vs_sidebar._connected = False
            _vs_sidebar._client = None
            if _vs_sidebar.connect():
                st.success(
                    f"✅ Connected to **{_vs_sidebar.db_name}** "
                    f"at `{_vs_sidebar._uri}`"
                )
                st.rerun()
            else:
                err = _vs_sidebar._last_error or "Unknown error"
                st.error(f"❌ Connection failed:\n\n`{err}`")
                if "dnspython" in err.lower() or "srv" in err.lower():
                    st.info("💡 `mongodb+srv://` URIs require `dnspython`. Run:\n\n`pip install dnspython`")
                if "ssl" in err.lower() or "tls" in err.lower() or "certificate" in err.lower():
                    st.info("💡 SSL/TLS errors with Atlas? Run:\n\n`pip install certifi`\n\nThen click **Save & Reconnect** again.")

    st.divider()

    st.header("📖 About")
    st.write(
        "**AIDoc** uses AI to analyze your codebase and generate "
        "comprehensive, tailored documentation — covering only what "
        "actually matters for your specific project."
    )


# ──────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📄 AIDoc</h1>
    <p>AI-powered documentation generator — point it at any codebase and get instant, tailored docs.</p>
</div>
""", unsafe_allow_html=True)

_in_docker = os.environ.get("RUNNING_IN_DOCKER", "").lower() == "true"

col_input, col_btn = st.columns([4, 1])
with col_input:
    _placeholder = (
        "Enter path under /repos/ (e.g., /repos/myproject) or GitHub URL"
        if _in_docker
        else "Enter a local path (e.g., /Users/you/project) or GitHub URL"
    )
    repo_path = st.text_input(
        "Repository Path",
        placeholder=_placeholder,
        label_visibility="collapsed",
    )
with col_btn:
    if st.session_state.pipeline_running:
        stop_clicked = st.button(
            "🛑 Stop", type="secondary", use_container_width=True, key="stop_gen"
        )
        generate_clicked = False
    else:
        generate_clicked = st.button(
            "🚀 Generate", type="primary", use_container_width=True
        )

# ── Docker path hints ──
if _in_docker:
    st.caption(
        "🐳 **Docker mode** — Local projects are mounted at `/repos/`. "
        "Example: if `LOCAL_REPOS_PATH=~/Desktop`, use `/repos/myproject`. "
        "For private GitHub repos, set `GITHUB_TOKEN` in `.env`."
    )

# ── Handle interrupted pipeline (user clicked Stop while running) ──
if (
    st.session_state.pipeline_running
    and not st.session_state.pipeline_done
    and not st.session_state.get("_pending_pipeline")
):
    st.session_state.pipeline_running = False
    # Clean up cloned repo if pipeline was interrupted
    _cleanup_cloned_repo()
    st.warning("🛑 **Generation stopped.** The pipeline was interrupted.")

# Advanced options - target path for focused documentation
with st.expander("🔍 Advanced Options", expanded=False):
    target_path = st.text_input(
        "Target Path (optional)",
        placeholder="e.g., src/auth/ or core/payment.py — leave empty for full project",
        help="Narrow documentation to a specific folder or file within the repo.",
    )
    if target_path:
        st.caption(f"🎯 Will generate focused docs for: `{target_path}`")

    include_full_repo = False
    if target_path and target_path.strip():
        include_full_repo = st.checkbox(
            "🌐 Include full repo context (slower, richer cross-project references)",
            value=False,
            help=(
                "By default, only the targeted files/folders are indexed — "
                "much faster for large repos. Enable this to index the full "
                "repo for richer cross-project context in the docs."
            ),
        )

    st.divider()

    st.markdown("**📎 Reference Documentation (optional)**")
    reference_doc_file = st.file_uploader(
        "Upload an existing doc to match its format and structure",
        type=["md", "txt", "markdown"],
        help=(
            "Upload an existing documentation file (.md or .txt). "
            "AIDoc will analyze its structure, tone, and formatting, "
            "then generate new documentation that follows the same style. "
            "If no file is uploaded, the default format is used."
        ),
        label_visibility="collapsed",
    )
    reference_doc_content = None
    if reference_doc_file is not None:
        reference_doc_content = reference_doc_file.read().decode("utf-8", errors="replace")
        st.caption(f"📄 Loaded reference: **{reference_doc_file.name}** ({len(reference_doc_content):,} chars)")
        with st.popover("👁️ Preview reference doc"):
            st.markdown(reference_doc_content[:3000] + ("..." if len(reference_doc_content) > 3000 else ""))

st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers for evaluation display
# ──────────────────────────────────────────────

def _score_color_class(score: int) -> str:
    if score >= 70:
        return "score-good"
    elif score >= 50:
        return "score-ok"
    return "score-low"


def _score_emoji(score: int) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    return "🔴"


def _render_score_card(label: str, score: int):
    css_class = _score_color_class(score)
    st.markdown(f"""
    <div class="score-card">
        <div class="score-value {css_class}">{score}</div>
        <div class="score-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _display_evaluation(coverage: CoverageReport, rating: QualityRating):
    """Display the full evaluation section with charts."""
    st.markdown("## 📊 Documentation Evaluation")
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Overall score + dimension scores ───────
    st.markdown("### ⭐ Quality Scores")

    col_overall, col_sep, col_dims = st.columns([1, 0.2, 3])

    with col_overall:
        _render_score_card("Overall Score", rating.overall_score)
        st.markdown(f"*{rating.verdict}*")

    with col_dims:
        scores = {
            "📁 Coverage": rating.coverage_score,
            "✅ Correctness": rating.correctness_score,
            "📖 Clarity": rating.clarity_score,
            "🧩 Completeness": rating.completeness_score,
            "💡 Usefulness": rating.usefulness_score,
        }

        score_df = pd.DataFrame({
            "Dimension": list(scores.keys()),
            "Score": list(scores.values()),
        })
        st.bar_chart(score_df.set_index("Dimension"), height=280, color="#667eea")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Coverage statistics ────────────────────
    st.markdown("### 📁 Code Coverage in Documentation")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Files Covered", f"{coverage.covered_files}/{coverage.total_files}",
                   delta=f"{coverage.coverage_pct:.0f}%")
    with col_m2:
        st.metric("Covered", coverage.covered_files, delta=None)
    with col_m3:
        st.metric("Not Covered", coverage.uncovered_files, delta=None)

    # Coverage by category & directory
    col_cat, col_dir = st.columns(2)

    with col_cat:
        st.markdown("#### By Category")
        cat_data = []
        for cat, data in sorted(coverage.by_category.items()):
            cat_data.append({
                "Category": cat.title(),
                "Covered": data["covered"],
                "Uncovered": data["total"] - data["covered"],
            })
        cat_df = pd.DataFrame(cat_data)
        if not cat_df.empty:
            chart_df = cat_df[["Category", "Covered", "Uncovered"]].set_index("Category")
            st.bar_chart(chart_df, height=250, color=["#667eea", "#e5e7eb"])

    with col_dir:
        st.markdown("#### By Directory")
        dir_data = []
        for dir_name, data in sorted(coverage.by_directory.items()):
            dir_data.append({
                "Directory": f"{dir_name}/",
                "Covered": data["covered"],
                "Uncovered": data["total"] - data["covered"],
            })
        dir_df = pd.DataFrame(dir_data)
        if not dir_df.empty:
            chart_df = dir_df[["Directory", "Covered", "Uncovered"]].set_index("Directory")
            st.bar_chart(chart_df, height=250, color=["#10b981", "#e5e7eb"])

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Strengths & Gaps ───────────────────────
    col_s, col_g = st.columns(2)

    with col_s:
        st.markdown("### ✅ Strengths")
        for s in rating.strengths:
            st.markdown(f"- {s}")

    with col_g:
        st.markdown("### ⚠️ Areas for Improvement")
        for g in rating.gaps:
            st.markdown(f"- {g}")

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── File coverage details (expandable) ─────
    with st.expander("📂 File-by-File Coverage Details", expanded=False):
        detail_rows = []
        for f in coverage.file_details:
            detail_rows.append({
                "File": f.path,
                "Category": f.category.title(),
                "Covered": "✅" if f.covered else "❌",
            })
        detail_df = pd.DataFrame(detail_rows)
        if not detail_df.empty:
            detail_df["_sort"] = detail_df["Covered"].apply(lambda x: 0 if x == "❌" else 1)
            detail_df = detail_df.sort_values("_sort").drop(columns="_sort")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Plan display
# ──────────────────────────────────────────────

def _display_plan(analysis: ProjectAnalysis):
    """Display the documentation plan."""
    st.markdown("### 📋 Documentation Plan")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Project:** {analysis.project_name}")
        st.markdown(f"**Type:** {analysis.project_type}")
    with col_b:
        st.markdown(f"**Language:** {analysis.primary_language}")
        st.markdown(f"**Stack:** {', '.join(analysis.frameworks_and_tools)}")

    st.markdown(f"*{analysis.one_line_summary}*")

    st.markdown("**Planned Sections:**")
    for i, section in enumerate(analysis.documentation_sections, 1):
        if section.audience == "end-user":
            tag_class = "tag-end-user"
        elif section.audience == "developer":
            tag_class = "tag-developer"
        else:
            tag_class = "tag-both"

        st.markdown(
            f'{i}. **{section.title}** '
            f'<span class="section-tag {tag_class}">{section.audience}</span> '
            f'— {section.reason}',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Architecture tree helpers
# ──────────────────────────────────────────────

def _build_tree_json(files: list, project_name: str = "root") -> dict:
    """Convert flat file list into hierarchical tree structure for D3.js."""
    root = {"name": project_name, "children": [], "_type": "folder"}

    for f in files:
        parts = f["path"].split("/")
        current = root
        for i, part in enumerate(parts):
            is_file = (i == len(parts) - 1)
            existing = next(
                (c for c in current.get("children", []) if c["name"] == part),
                None,
            )
            if existing:
                current = existing
            else:
                node = {"name": part, "_type": "file" if is_file else "folder"}
                if not is_file:
                    node["children"] = []
                current.setdefault("children", []).append(node)
                current = node

    _sort_tree(root)
    return root


def _sort_tree(node: dict):
    """Recursively sort tree: folders first, then files, alphabetically."""
    children = node.get("children")
    if not children:
        return
    children.sort(key=lambda x: (0 if "children" in x else 1, x["name"].lower()))
    for child in children:
        _sort_tree(child)


_ARCH_TREE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: transparent; overflow: hidden;
           font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

    #tree-box { width: 100%; height: __HEIGHT__px; position: relative; }

    svg { cursor: grab; }
    svg:active { cursor: grabbing; }

    /* ── Controls ──────────────────────────── */
    .ctrls {
        position: absolute; top: 10px; right: 12px;
        display: flex; gap: 5px; z-index: 10;
    }
    .ctrls div {
        width: 32px; height: 32px;
        border: 1px solid rgba(102,126,234,0.35);
        background: rgba(102,126,234,0.12);
        color: #667eea; border-radius: 7px;
        cursor: pointer; font-size: 15px;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.15s;
        user-select: none;
    }
    .ctrls div:hover { background: rgba(102,126,234,0.28); }

    /* ── Legend ────────────────────────────── */
    .legend {
        position: absolute; bottom: 8px; left: 12px;
        display: flex; gap: 14px; font-size: 11px;
    }
    .legend .it { display: flex; align-items: center; gap: 5px; }
    .legend .dot { width: 9px; height: 9px; border-radius: 50%; }

    /* ── Tree nodes ───────────────────────── */
    .node { cursor: pointer; }
    .node circle { stroke-width: 2px; transition: all 0.15s; }
    .node:hover circle { stroke-width: 3px; filter: brightness(1.3); }
    .link { fill: none; stroke-width: 1.5px; }

    /* ── Theme-adaptive text ──────────────── */
    @media (prefers-color-scheme: dark) {
        .node text { fill: #c9cdd3; }
        .link     { stroke: #374151; }
        .legend .it span { color: #9ca3af; }
    }
    @media (prefers-color-scheme: light) {
        .node text { fill: #1f2937; }
        .link     { stroke: #e5e7eb; }
        .legend .it span { color: #6b7280; }
    }
    .node text {
        font-size: 12px;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
    }
</style>
</head>
<body>
<div id="tree-box">
    <div class="ctrls">
        <div onclick="zoomIn()"   title="Zoom In">+</div>
        <div onclick="zoomOut()"  title="Zoom Out">&minus;</div>
        <div onclick="resetView()" title="Reset View">&#8635;</div>
        <div onclick="expandAll()" title="Expand All">&#8862;</div>
        <div onclick="collapseAll()" title="Collapse All">&#8863;</div>
    </div>
    <div class="legend">
        <div class="it"><div class="dot" style="background:#667eea"></div><span>Folder</span></div>
        <div class="it"><div class="dot" style="background:#f59e0b"></div><span>Collapsed</span></div>
        <div class="it"><div class="dot" style="background:#10b981"></div><span>File</span></div>
    </div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function(){
    var data = __TREE_DATA__;
    var H    = __HEIGHT__;
    var C    = {open:"#667eea", shut:"#f59e0b", file:"#10b981"};
    var dur  = 350, uid = 0;

    var svg = d3.select("#tree-box").append("svg")
        .attr("width","100%").attr("height", H);
    var g = svg.append("g");

    var zm = d3.zoom().scaleExtent([0.15, 5])
        .on("zoom", function(e){ g.attr("transform", e.transform); });
    svg.call(zm);

    var layout = d3.tree().nodeSize([24, 220]);
    var root   = d3.hierarchy(data);
    root.x0 = 0; root.y0 = 0;

    /* helpers */
    function leaves(d){
        if(!d.children && !d._children) return 1;
        return (d.children||d._children||[]).reduce(function(s,c){return s+leaves(c);},0);
    }
    function collapseNode(d){
        if(d.children){d._children=d.children; d._children.forEach(collapseNode); d.children=null;}
    }
    function expandNode(d){
        if(d._children){d.children=d._children; d._children=null;}
        if(d.children) d.children.forEach(expandNode);
    }

    /* start collapsed to depth 1 */
    (root.children||[]).forEach(function(c){
        if(c.children) c.children.forEach(collapseNode);
    });
    update(root);
    svg.call(zm.transform, d3.zoomIdentity.translate(80, H/2));

    function update(src){
        layout(root);
        var nodes = root.descendants();
        var links = root.links();
        nodes.forEach(function(d){ d.y = d.depth * 200; });

        /* ── nodes ── */
        var node = g.selectAll("g.node").data(nodes, function(d){return d.id||(d.id=++uid);});

        var enter = node.enter().append("g").attr("class","node")
            .attr("transform","translate("+src.y0+","+src.x0+")")
            .on("click", function(ev,d){
                if(d.children){d._children=d.children; d.children=null;}
                else if(d._children){d.children=d._children; d._children=null;}
                update(d);
            });
        enter.append("circle").attr("r",1e-6);
        enter.append("text").attr("dy","0.35em");

        var merged = enter.merge(node);
        merged.transition().duration(dur)
            .attr("transform",function(d){return "translate("+d.y+","+d.x+")";});

        merged.select("circle")
            .attr("r", function(d){return (d.children||d._children)?6:4;})
            .style("fill", function(d){
                if(!d.children&&!d._children) return C.file;
                return d._children? C.shut : C.open;
            })
            .style("stroke", function(d){
                if(!d.children&&!d._children) return C.file;
                return d._children? C.shut : C.open;
            })
            .style("fill-opacity", function(d){return d._children?0.7:0.25;});

        merged.select("text")
            .attr("x", function(d){return (d.children||d._children)?-12:12;})
            .attr("text-anchor", function(d){return (d.children||d._children)?"end":"start";})
            .text(function(d){
                if(d.children||d._children){
                    return d.data.name + "/  (" + leaves(d) + ")";
                }
                return d.data.name;
            });

        var exit = node.exit().transition().duration(dur)
            .attr("transform","translate("+src.y+","+src.x+")").remove();
        exit.select("circle").attr("r",1e-6);
        exit.select("text").style("fill-opacity",0);

        /* ── links ── */
        var link = g.selectAll("path.link").data(links, function(d){return d.target.id;});

        var linkEnter = link.enter().insert("path","g").attr("class","link")
            .attr("d", diag({x:src.x0,y:src.y0},{x:src.x0,y:src.y0}));
        linkEnter.merge(link).transition().duration(dur)
            .attr("d", function(d){return diag(d.source, d.target);});
        link.exit().transition().duration(dur)
            .attr("d", diag({x:src.x,y:src.y},{x:src.x,y:src.y})).remove();

        nodes.forEach(function(d){d.x0=d.x; d.y0=d.y;});
    }

    function diag(s,t){
        return "M"+s.y+","+s.x
            +"C"+(s.y+t.y)/2+","+s.x
            +" "+(s.y+t.y)/2+","+t.x
            +" "+t.y+","+t.x;
    }

    /* controls */
    window.zoomIn     = function(){ svg.transition().duration(300).call(zm.scaleBy, 1.4); };
    window.zoomOut    = function(){ svg.transition().duration(300).call(zm.scaleBy, 0.7); };
    window.resetView  = function(){ svg.transition().duration(400).call(zm.transform, d3.zoomIdentity.translate(80,H/2)); };
    window.expandAll  = function(){ expandNode(root); update(root); };
    window.collapseAll= function(){ root.children.forEach(collapseNode); update(root); };
})();
</script>
</body>
</html>
"""


def _display_architecture(all_files: list, project_name: str = "Project"):
    """Render an interactive D3.js collapsible tree of the project architecture."""
    import streamlit.components.v1 as components

    st.markdown("## 🏗️ Project Architecture")
    st.caption(
        f"Interactive file tree — **{len(all_files)} files** analyzed. "
        "Click folders to expand/collapse. Scroll to zoom. Drag to pan."
    )

    tree_data = _build_tree_json(all_files, project_name)
    tree_json = json.dumps(tree_data)

    tree_height = min(max(500, len(all_files) * 8), 850)

    html = (
        _ARCH_TREE_HTML
        .replace("__TREE_DATA__", tree_json)
        .replace("__HEIGHT__", str(tree_height))
    )

    components.html(html, height=tree_height + 50, scrolling=False)


# ──────────────────────────────────────────────
# Refinement display
# ──────────────────────────────────────────────

def _display_refinement():
    """Render the refinement chat where users iteratively improve documentation."""
    st.markdown("## ✏️ Refine Documentation")
    st.caption(
        "Not happy with something? Tell the AI what to change — remove sections, "
        "add detail, fix inaccuracies. Each refinement builds on the previous one."
    )

    if st.session_state.doc_writer is None:
        st.info("💡 Generate documentation first to unlock refinement.")
        return

    # Refinement input — at the TOP so it doesn't get buried in history
    with st.form(key="refinement_form", clear_on_submit=True):
        feedback = st.text_input(
            "What would you like to change?",
            placeholder="e.g., 'Remove the auth section' or 'Add more detail about database migrations'",
            label_visibility="collapsed",
        )
        col_submit, col_clear, col_spacer = st.columns([1, 1, 4])
        with col_submit:
            submitted = st.form_submit_button("🔄 Refine", use_container_width=True)
        with col_clear:
            if st.session_state.refinement_messages:
                cleared = st.form_submit_button("🧹 Clear", use_container_width=True)
            else:
                cleared = False

    # Placeholder for the spinner — sits right below the Refine button
    spinner_placeholder = st.empty()

    if cleared:
        st.session_state.refinement_messages = []
        st.rerun()

    if submitted and feedback and feedback.strip():
        feedback = feedback.strip()
        # Show user feedback
        st.session_state.refinement_messages.append({
            "role": "user",
            "content": feedback,
            "timestamp": datetime.now(IST).strftime("%b %d, %Y  %I:%M %p IST"),
        })

        # Apply refinement — spinner appears right below the Refine button
        with spinner_placeholder, st.spinner(f"Applying changes for '{feedback}'..."):
            writer = st.session_state.doc_writer
            current_md = st.session_state.results["markdown"]
            updated_md = writer.refine(current_md, feedback)

        # Calculate what changed
        old_sections = writer._parse_sections(current_md)
        new_sections = writer._parse_sections(updated_md)
        old_titles = {s["title"] for s in old_sections if s["title"]}
        new_titles = {s["title"] for s in new_sections if s["title"]}

        changes = []
        removed = old_titles - new_titles
        added = new_titles - old_titles
        if removed:
            changes.append(f"🗑️ Removed: {', '.join(removed)}")
        if added:
            changes.append(f"➕ Added: {', '.join(added)}")
        modified = old_titles & new_titles
        for title in modified:
            old_c = next((s["content"] for s in old_sections if s["title"] == title), "")
            new_c = next((s["content"] for s in new_sections if s["title"] == title), "")
            if old_c != new_c:
                changes.append(f"✏️ Modified: {title}")

        if changes:
            summary = "**Changes applied:**\n" + "\n".join(f"- {c}" for c in changes)
        else:
            summary = "No structural changes detected — content may have been subtly refined."

        st.session_state.refinement_messages.append({"role": "assistant", "content": summary})

        # Update session state with refined docs
        st.session_state.results["markdown"] = updated_md

        # Rebuild full markdown (with coverage report if present)
        r = st.session_state.results
        coverage_md = ""
        if "coverage" in r and "rating" in r:
            from core.evaluator import DocumentationEvaluator
            # Keep the existing coverage report — it was for the original doc
            if "full_markdown" in r and "---\n\n## 📊" in r["full_markdown"]:
                # Extract existing coverage section
                split_idx = r["full_markdown"].rfind("\n\n---\n\n## 📊")
                if split_idx > 0:
                    coverage_md = r["full_markdown"][split_idx:]

        st.session_state.results["full_markdown"] = updated_md + coverage_md

        # Doc changed — require re-approval
        st.session_state.doc_approved = False

        # ── Save refined version to MongoDB ──
        vs: DocVersionStore = st.session_state._version_store
        if vs._connected:
            change_type = vs.compute_change_type(
                old_titles, new_titles, old_sections, new_sections,
            )
            desc_parts = []
            if added:
                desc_parts.append(f"Added: {', '.join(added)}")
            if removed:
                desc_parts.append(f"Removed: {', '.join(removed)}")
            mod_titles = [
                t for t in (old_titles & new_titles)
                if next((s["content"] for s in old_sections if s["title"] == t), "")
                != next((s["content"] for s in new_sections if s["title"] == t), "")
            ]
            if mod_titles:
                desc_parts.append(f"Modified: {', '.join(mod_titles)}")
            version_desc = "; ".join(desc_parts) if desc_parts else f"Refinement: {feedback[:80]}"

            new_ver = _save_doc_version(
                change_type=change_type,
                description=version_desc,
                markdown=updated_md,
                full_markdown=updated_md + coverage_md,
            )

        st.rerun()

    # Display refinement history — latest Q&A pair first, question above answer
    if st.session_state.refinement_messages:
        st.markdown("---")
        msgs = st.session_state.refinement_messages
        pairs = [msgs[i:i + 2] for i in range(0, len(msgs), 2)]
        for pair in reversed(pairs):
            for msg in pair:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user" and "timestamp" in msg:
                        st.caption(f"🕐 {msg['timestamp']}")
                    st.markdown(msg["content"])


# ──────────────────────────────────────────────
# Chat display
# ──────────────────────────────────────────────

def _display_chat():
    """Render the chat section with conversation history and input."""
    st.markdown("## 💬 Chat with Your Codebase")
    st.caption(
        "Ask anything about the repository — follow-up questions are understood in context. "
        "The AI remembers your entire conversation."
    )

    if st.session_state.chat_engine is None:
        st.info("💡 Generate documentation first to unlock the chat feature.")
        return

    # Chat input — at the TOP so it doesn't get buried in history
    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Ask about the codebase",
            placeholder="Ask about the codebase...",
            label_visibility="collapsed",
        )
        col_submit, col_clear, col_spacer = st.columns([1, 1, 4])
        with col_submit:
            chat_submitted = st.form_submit_button("💬 Ask", use_container_width=True)
        with col_clear:
            if st.session_state.chat_messages:
                chat_cleared = st.form_submit_button("🧹 Clear", use_container_width=True)
            else:
                chat_cleared = False

    if chat_cleared:
        st.session_state.chat_engine.reset()
        st.session_state.chat_messages = []
        st.rerun()

    if chat_submitted and prompt and prompt.strip():
        prompt = prompt.strip()
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now(IST).strftime("%b %d, %Y  %I:%M %p IST"),
        })

        with st.spinner("Thinking..."):
            answer = st.session_state.chat_engine.ask(prompt)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Display messages — latest Q&A pair first, question above answer
    if st.session_state.chat_messages:
        st.markdown("---")
        msgs = st.session_state.chat_messages
        pairs = [msgs[i:i + 2] for i in range(0, len(msgs), 2)]
        for pair in reversed(pairs):
            for msg in pair:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user" and "timestamp" in msg:
                        st.caption(f"🕐 {msg['timestamp']}")
                    st.markdown(msg["content"])


# ──────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────

def run_pipeline(
    repo_path: str,
    api_key: str,
    model: str,
    target_path: str = None,
    include_full_repo: bool = False,
    reference_doc: str = None,
):
    """Run the full documentation pipeline and store results in session state."""
    start_time = time.time()
    config = AppConfig(llm=LLMConfig(model=model, api_key=api_key))

    # Clean up any previous session's resources (re-generation scenario)
    _cleanup_all_resources()

    # Reset any previous state
    st.session_state.pipeline_done = False
    st.session_state.doc_approved = False
    st.session_state.chat_engine = None
    st.session_state.chat_messages = []
    st.session_state.doc_writer = None
    st.session_state.refinement_messages = []
    st.session_state._cleanup_loader = None
    st.session_state._cleanup_path = None

    # ── Phase 1: Load ─────────────────────────
    loader = RepoLoader(config.loader)
    local_path = None

    with st.status("📦 **Phase 1:** Loading Repository...", expanded=True) as phase1:
        try:
            local_path = loader.load(repo_path)
        except ValueError as e:
            st.error(f"❌ {e}")
            return
        except RuntimeError as e:
            st.error(f"❌ Clone failed: {e}")
            return

        all_files = loader.collect_files(local_path)
        file_tree = loader.get_file_tree(local_path)

        if not all_files:
            st.error("❌ No readable source files found in the repository.")
            loader.cleanup(local_path)
            return

        # Target scoping
        if target_path:
            target_files = loader.collect_files_for_target(local_path, target_path)
            if not target_files:
                st.error(f"❌ No files found matching target: `{target_path}`")
                loader.cleanup(local_path)
                return
            st.write(f"🎯 Target: **{target_path}** — {len(target_files)} files "
                     f"(from {len(all_files)} total)")
        else:
            target_files = all_files
            st.write(f"Found **{len(all_files)}** source files")

        phase1.update(label="📦 **Phase 1:** Repository Loaded ✅", state="complete")

    # Phases 2-5 — cloned repo is kept alive for refinement/chat until user approves
    try:
        # ── Phase 2: Index ─────────────────────────
        # When a target path is set, index only the target files by default.
        # The full repo is only indexed when explicitly requested or when
        # there is no target path.
        if target_path and not include_full_repo:
            files_to_index = target_files
            target_tree = loader.get_file_tree_for_target(local_path, target_path)
            index_label = f"Indexing **{len(files_to_index)}** target files"
        else:
            files_to_index = all_files
            target_tree = file_tree
            index_label = f"Indexing **{len(files_to_index)}** files (full repo)"

        with st.status(f"🧠 **Phase 2:** {index_label}...", expanded=True) as phase2:
            indexer = CodebaseIndexer(config)

            idx_progress = st.progress(0)
            idx_status = st.empty()

            def on_index_progress(current, total, path):
                idx_progress.progress(current / total)
                idx_status.write(f"Processing **{current}/{total}**: `{path}`")

            # First pass: index without hierarchical context (needed by analyzer)
            index = indexer.index_codebase(
                files_to_index, target_tree, on_progress=on_index_progress,
            )

            idx_progress.progress(1.0)
            idx_status.write(f"Indexed **{len(files_to_index)}** files into vector store")
            phase2.update(label="🧠 **Phase 2:** Codebase Indexed ✅", state="complete")

        # ── Phase 2.5: Hierarchical Codebase Analysis ──
        # Three-level analysis: File → Module → System
        # Produces heat zones, entry points, module summaries, and data flow.
        with st.status(
            f"🏗️ **Phase 2.5:** Hierarchical Analysis ({len(files_to_index)} files)...",
            expanded=True,
        ) as phase_hier:
            ha = HierarchicalAnalyzer(index, files_to_index, target_tree)

            hier_progress = st.progress(0)
            hier_status = st.empty()

            def on_hier_progress(step, total, description):
                hier_progress.progress(step / total)
                hier_status.write(f"Step **{step}/{total}**: {description}")

            hierarchical_ctx = ha.analyze(on_progress=on_hier_progress)

            hier_progress.progress(1.0)

            # Show summary
            if hierarchical_ctx.system_summary:
                s = hierarchical_ctx.system_summary
                if s.north_star:
                    st.write(f"🎯 **North Star:** {s.north_star}")
                if s.architecture_style:
                    st.write(f"🏛️ **Architecture:** {s.architecture_style}")
                if s.heat_zones:
                    hz_list = ", ".join(
                        f"{hz.name} ({hz.importance_score}/10)"
                        for hz in sorted(
                            s.heat_zones,
                            key=lambda h: h.importance_score,
                            reverse=True,
                        )[:5]
                    )
                    st.write(f"🔥 **Heat Zones:** {hz_list}")
                if s.entry_points:
                    st.write(f"⚡ **Entry Points:** {', '.join(s.entry_points[:5])}")

            core_count = sum(
                1 for f in hierarchical_ctx.file_metadata if f.importance == "core"
            )
            st.write(
                f"Analyzed **{len(hierarchical_ctx.file_metadata)}** files | "
                f"**{core_count}** core | "
                f"**{len(hierarchical_ctx.module_summaries)}** modules"
            )
            phase_hier.update(
                label="🏗️ **Phase 2.5:** Hierarchical Analysis Complete ✅",
                state="complete",
            )

        # ── Phase 2b: Analyze reference doc (if provided) ──
        reference_style = None
        if reference_doc:
            with st.status("📎 **Analyzing reference doc style...**", expanded=True) as phase_ref:
                from core.style_analyzer import DocStyleAnalyzer
                st.write("Extracting structure, tone, and formatting conventions...")
                reference_style = DocStyleAnalyzer.analyze(reference_doc)
                st.write(
                    f"Found **{len(reference_style.sections)}** sections | "
                    f"Tone: **{reference_style.tone}** | "
                    f"Depth: **{reference_style.section_depth}**"
                )
                phase_ref.update(label="📎 Reference Style Analyzed ✅", state="complete")

        # ── Phase 3: Analyze & Plan ──────────────────────
        with st.status(
            f"🔍 **Phase 3:** Planning documentation for {len(target_files)} files...",
            expanded=True,
        ) as phase3:
            planner = ProjectPlanner(
                index,
                file_paths=[f["path"] for f in target_files],
                reference_style=reference_style,
                hierarchical_context=hierarchical_ctx,
            )

            analysis_progress = st.progress(0)
            analysis_status = st.empty()

            def on_analysis_progress(step, total, description):
                analysis_progress.progress(step / total)
                analysis_status.write(f"Step **{step}/{total}**: {description}")

            analysis = planner.analyze(
                target_scope=target_path,
                on_progress=on_analysis_progress,
            )

            analysis_progress.progress(1.0)
            analysis_status.write(
                f"Detected **{analysis.project_type}** project: *{analysis.one_line_summary}*"
            )
            st.write(f"Planned **{len(analysis.documentation_sections)}** documentation sections")
            phase3.update(label="🔍 **Phase 3:** Analysis Complete ✅", state="complete")

        _display_plan(analysis)

        # ── Phase 4: Write ────────────────────────
        with st.status("✍️ **Phase 4:** Writing Documentation...", expanded=True) as phase4:
            writer = DocumentationWriter(
                index, analysis,
                source_files=target_files,
                reference_style=reference_style,
                hierarchical_context=hierarchical_ctx,
            )

            progress_bar = st.progress(0)
            section_status = st.empty()

            def on_section_start(current, total, title):
                section_status.write(f"Writing **{title}**... ({current}/{total})")
                progress_bar.progress(current / total)

            def on_section_done(current, total, title):
                progress_bar.progress(current / total)

            markdown = writer.generate(
                on_section_start=on_section_start,
                on_section_done=on_section_done,
            )

            progress_bar.progress(1.0)
            section_status.write("All sections written!")
            phase4.update(label="✍️ **Phase 4:** Documentation Generated ✅", state="complete")

        # ── Phase 5: Evaluate ─────────────────────
        with st.status("📊 **Phase 5:** Evaluating Documentation...", expanded=True) as phase5:
            evaluator = DocumentationEvaluator(index, analysis)

            st.write("Computing code coverage...")
            coverage = evaluator.compute_coverage(target_files, markdown)
            st.write(f"File coverage: **{coverage.covered_files}/{coverage.total_files}** "
                     f"({coverage.coverage_pct:.0f}%)")

            st.write("Rating documentation quality...")
            rating = evaluator.rate_quality(markdown, coverage)
            st.write(f"Overall quality score: **{rating.overall_score}/100**")

            phase5.update(label="📊 **Phase 5:** Evaluation Complete ✅", state="complete")

        # Append coverage report to markdown
        coverage_md = evaluator.build_coverage_markdown(coverage, rating)
        full_markdown = markdown + "\n\n---\n\n" + coverage_md

        # ── Save ──────────────────────────────────
        elapsed = time.time() - start_time

        os.makedirs(config.output_dir, exist_ok=True)
        final_path = os.path.join(config.output_dir, config.output_filename)
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        # Store results in session state for persistent display + chat
        project_id = derive_project_id(repo_path, target_path)
        st.session_state.project_id = project_id
        st.session_state.results = {
            "analysis": analysis,
            "markdown": markdown,
            "full_markdown": full_markdown,
            "coverage": coverage,
            "rating": rating,
            "files": target_files,
            "all_files": all_files,
            "elapsed": elapsed,
            "target_path": target_path,
            "hierarchical_context": hierarchical_ctx,
            "repo_path": repo_path,
        }
        st.session_state.pipeline_done = True
        st.session_state.pipeline_running = False
        st.session_state.chat_engine = ChatEngine(index, documentation_markdown=markdown)
        st.session_state.chat_messages = []
        st.session_state.doc_writer = writer
        st.session_state.refinement_messages = []

        # ── Register project & save initial version to MongoDB ──
        _vs_pipe: DocVersionStore = st.session_state._version_store
        if _vs_pipe._connected:
            _vs_pipe.register_project(
                project_id=project_id,
                display_name=analysis.project_name or project_id,
                repo_path=repo_path,
                target_path=target_path,
            )
        new_ver = _save_doc_version(
            change_type="initial",
            description=f"Initial documentation — {len(analysis.documentation_sections)} sections, "
                        f"quality {rating.overall_score}/100",
            markdown=markdown,
            full_markdown=full_markdown,
        )
        if new_ver:
            st.toast(f"📦 Saved as **v{new_ver}** in database")

        # Keep the loader and path alive for deferred cleanup (on approval)
        st.session_state._cleanup_loader = loader
        st.session_state._cleanup_path = local_path

    except Exception:
        st.session_state.pipeline_running = False
        # Only clean up the cloned repo if the pipeline failed
        if local_path:
            loader.cleanup(local_path)
        raise


# ──────────────────────────────────────────────
# Trigger (set pending pipeline on Generate click)
# ──────────────────────────────────────────────

if generate_clicked:
    if not repo_path.strip():
        st.warning("⚠️ Please enter a repository path or GitHub URL.")
    elif not api_key.strip():
        st.warning("⚠️ Please provide a Cerebras API key (in sidebar or `.env` file).")
    else:
        # Store pipeline args and rerun so the Stop button renders first
        st.session_state.pipeline_running = True
        st.session_state._pending_pipeline = {
            "repo_path": repo_path.strip(),
            "api_key": api_key.strip(),
            "model": model,
            "target_path": target_path.strip() if target_path and target_path.strip() else None,
            "include_full_repo": include_full_repo if target_path and target_path.strip() else False,
            "reference_doc": reference_doc_content,
        }
        st.rerun()


# ──────────────────────────────────────────────
# Top-level navigation
# ──────────────────────────────────────────────

tab_generate, tab_docs, tab_arch, tab_eval, tab_refine, tab_chat, tab_versions = st.tabs([
    "🚀 Generate",
    "📄 Documentation",
    "🏗️ Architecture",
    "📊 Evaluation",
    "🔧 Refine",
    "💬 Chat",
    "📚 Versions",
])

_PLACEHOLDER_STYLE = (
    'style="text-align: center; padding: 3rem 1rem; opacity: 0.6;"'
)


def _tab_placeholder(icon: str, title: str, body: str):
    """Render a friendly empty-state placeholder inside a tab."""
    st.markdown(
        f'<div {_PLACEHOLDER_STYLE}>'
        f'<p style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</p>'
        f'<p style="font-size: 1.1rem; font-weight: 500;">{title}</p>'
        f'<p style="font-size: 0.9rem;">{body}</p></div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Generate (pipeline progress & plan)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_generate:
    # Execute pending pipeline (second pass after rerun shows Stop button)
    if st.session_state.get("_pending_pipeline"):
        args = st.session_state._pending_pipeline
        st.session_state._pending_pipeline = None
        run_pipeline(**args)

    elif st.session_state.pipeline_done:
        r = st.session_state.results

        target_badge = f" | 🎯 `{r['target_path']}`" if r.get("target_path") else ""
        st.success(
            f"✅ Documentation generated in **{r['elapsed']:.1f}s** — "
            f"{len(r['analysis'].documentation_sections)} sections, "
            f"quality: {r['rating'].overall_score}/100{target_badge}"
        )

        # Stats row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="number">{len(r["files"])}</div>
                <div class="label">Files Analyzed</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="number">{len(r["analysis"].documentation_sections)}</div>
                <div class="label">Doc Sections</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-box">
                <div class="number">{r["coverage"].coverage_pct:.0f}%</div>
                <div class="label">File Coverage</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-box">
                <div class="number">{r["rating"].overall_score}</div>
                <div class="label">Quality Score</div>
            </div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="stat-box">
                <div class="number">{r["elapsed"]:.1f}s</div>
                <div class="label">Total Time</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        st.info(
            "📄 Switch to the **Documentation**, **Architecture**, **Evaluation**, or **Refine** tabs "
            "to explore, refine, and download your docs."
        )

    else:
        _tab_placeholder(
            "🔖",
            "Enter a repository path above and hit Generate",
            "Works with local directories and GitHub URLs.<br/>"
            "Use <strong>Advanced Options</strong> to target a specific folder or file.<br/>"
            "The AI will analyze the code and produce tailored documentation.",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Documentation (preview + raw + download + approve)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_docs:
    if st.session_state.pipeline_done:
        r = st.session_state.results

        # Version badge
        _vs_conn = st.session_state._version_store._connected
        _cur_ver = st.session_state.doc_version
        if _vs_conn:
            st.markdown(
                f"<div style='display:inline-block; background:rgba(102,126,234,0.15); "
                f"color:#667eea; padding:0.2rem 0.7rem; border-radius:8px; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>"
                f"📦 v{_cur_ver}</div>",
                unsafe_allow_html=True,
            )

        # Sub-tabs: rendered preview and raw markdown
        sub_preview, sub_raw = st.tabs(["📖 Preview", "📝 Raw Markdown"])

        with sub_preview:
            st.markdown(r["markdown"])

        with sub_raw:
            st.code(r["full_markdown"], language="markdown")

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️  Download DOCUMENTATION.md",
            data=r["full_markdown"],
            file_name="DOCUMENTATION.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if st.session_state.doc_approved:
            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
            st.success(
                "✅ **Documentation approved!** Cloned repository cleaned up. "
                "Refinement and chat remain available."
            )

        # Approve button — only shown before approval
        if not st.session_state.doc_approved:
            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

            st.markdown("### ✅ Finalize Documentation")
            st.caption(
                "Once you're satisfied, approve to clean up the cloned repository. "
                "Refinement and chat will remain available."
            )

            col_approve, col_spacer = st.columns([1, 3])
            with col_approve:
                if st.button(
                    "✅ Approve & Finish",
                    use_container_width=True,
                    type="primary",
                    key="approve_doc",
                ):
                    _cleanup_cloned_repo()
                    st.session_state.doc_approved = True
                    st.rerun()

    else:
        _tab_placeholder(
            "📄",
            "No documentation yet",
            "Head to the <strong>Generate</strong> tab and run the pipeline.<br/>"
            "Your documentation will appear here once it's ready.",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_arch:
    if st.session_state.pipeline_done:
        r = st.session_state.results
        hier_ctx: "HierarchicalContext | None" = r.get("hierarchical_context")

        # ── System Understanding ──
        if hier_ctx and hier_ctx.system_summary:
            s = hier_ctx.system_summary
            st.markdown("### 🎯 System Understanding")
            if s.north_star:
                st.info(f"**North Star:** {s.north_star}")
            col_a, col_b = st.columns(2)
            with col_a:
                if s.architecture_style:
                    st.markdown(f"**Architecture:** {s.architecture_style}")
                if s.entry_points:
                    st.markdown(
                        "**Entry Points:** "
                        + ", ".join(f"`{ep}`" for ep in s.entry_points)
                    )
            with col_b:
                if s.data_flow:
                    st.markdown(f"**Data Flow:** {s.data_flow}")

            # Heat Zones
            if s.heat_zones:
                st.markdown("### 🔥 Heat Zones")
                for hz in sorted(
                    s.heat_zones,
                    key=lambda h: h.importance_score,
                    reverse=True,
                ):
                    score_bar = "🟢" * min(hz.importance_score, 10)
                    st.markdown(
                        f"**{hz.name}** — `{hz.directory}` "
                        f"({hz.importance_score}/10 {score_bar})"
                    )
                    st.caption(hz.description)
                    if hz.key_files:
                        st.markdown(
                            "  " + ", ".join(f"`{f}`" for f in hz.key_files)
                        )

            # Developer's First Hour
            if s.developer_first_hour:
                st.markdown("### 🕐 Developer's First Hour")
                st.caption(
                    "The files a new developer should read first to understand the system:"
                )
                for i, fp in enumerate(s.developer_first_hour, 1):
                    st.markdown(f"{i}. `{fp}`")

            # Module Landscape
            if hier_ctx.module_summaries:
                st.markdown("### 📂 Module Landscape")
                for m in hier_ctx.module_summaries:
                    st.markdown(f"**`{m.directory}/`** — {m.purpose}")
                    if m.interactions:
                        st.caption(f"↔ {m.interactions}")

            st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # ── File Tree (existing display) ──
        _display_architecture(r["all_files"], r["analysis"].project_name)
    else:
        _tab_placeholder(
            "🏗️",
            "Architecture tree not available yet",
            "Generate documentation first — the project's file tree will appear here.",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_eval:
    if st.session_state.pipeline_done:
        r = st.session_state.results
        _display_evaluation(r["coverage"], r["rating"])
    else:
        _tab_placeholder(
            "📊",
            "Evaluation metrics not available yet",
            "Generate documentation first — coverage and quality scores will appear here.",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Refine Documentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_refine:
    if st.session_state.pipeline_done:
        _display_refinement()
    else:
        _tab_placeholder(
            "🔧",
            "Refinement not available yet",
            "Generate documentation first — then you can iteratively refine it here.",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — Chat with Codebase
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_chat:
    _display_chat()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 — Version History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_versions:
    vs: DocVersionStore = st.session_state._version_store

    if not vs._connected:
        st.warning(
            "⚠️ **MongoDB is not connected.** Version history requires a running MongoDB instance.\n\n"
            "Set `MONGO_URI` in your `.env` file (default: `mongodb://localhost:27017`).\n\n"
            "Documentation generation still works — versions just won't be saved."
        )
    else:
        st.markdown("## 📚 Version History")

        current_pid = st.session_state.project_id

        # ── Project picker ──
        projects = vs.list_projects()
        if projects:
            # Build display labels: "display_name (target)" or just "display_name"
            proj_options = []
            proj_id_map: dict[str, dict] = {}
            for p in projects:
                pid = p["project_id"]
                name = p.get("display_name", pid)
                target = p.get("target_path")
                label = f"{name}  →  {target}" if target else name
                label += f"  ({p['version_count']} versions)"
                proj_options.append(label)
                proj_id_map[label] = p

            # Default to current project
            default_idx = 0
            for idx, lbl in enumerate(proj_options):
                if proj_id_map[lbl]["project_id"] == current_pid:
                    default_idx = idx
                    break

            selected_label = st.selectbox(
                "Select Project",
                proj_options,
                index=default_idx,
                key="version_project_select",
            )
            selected_proj = proj_id_map[selected_label]
            selected_pid = selected_proj["project_id"]

            # Project metadata
            st.markdown(
                f"**Project:** `{selected_proj.get('display_name', selected_pid)}` · "
                f"**Repo:** `{selected_proj.get('repo_path', '—')}` · "
                + (f"**Target:** `{selected_proj['target_path']}`" if selected_proj.get("target_path") else "**Target:** full repo")
            )
            st.caption(
                f"Collection: `{selected_proj.get('collection_name', '—')}` in database `{vs.db_name}`"
            )

        elif current_pid:
            selected_pid = current_pid
        else:
            st.info("No documentation versions saved yet. Generate docs to get started!")
            selected_pid = None

        if selected_pid:
            versions = vs.get_all_versions(selected_pid)

            if not versions:
                st.info(f"No versions found for **{selected_pid}**.")
            else:
                # ── Current version badge ──
                latest = versions[0]
                st.markdown(
                    f"**Current version:** `v{latest['version']}` · "
                    f"**{len(versions)}** version(s) · "
                    f"Last updated: {latest['created_at'].strftime('%b %d, %Y %I:%M %p')} IST"
                )

                st.markdown("---")

                # ── Version list ──
                for i, ver in enumerate(versions):
                    v = ver["version"]
                    is_latest = (i == 0)

                    # Change type icon
                    ct_icons = {
                        "initial": "🆕",
                        "regenerated": "🔄",
                        "sections_added": "➕",
                        "sections_removed": "🗑️",
                        "content_modified": "✏️",
                    }
                    ct_icon = ct_icons.get(ver.get("change_type", ""), "📝")

                    col_info, col_actions = st.columns([4, 2])

                    with col_info:
                        label = f"**v{v}**" + (" &nbsp;`LATEST`" if is_latest else "")
                        st.markdown(
                            f"{ct_icon} {label} — "
                            f"<span style='opacity:0.6; font-size:0.85em;'>"
                            f"{ver['created_at'].strftime('%b %d, %Y %I:%M %p')} IST</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(ver.get("description", "No description"))

                    with col_actions:
                        btn_cols = st.columns(2)

                        # Preview button
                        with btn_cols[0]:
                            if st.button("👁️ View", key=f"view_{v}", use_container_width=True):
                                st.session_state[f"_preview_version_{selected_pid}"] = v

                        # Rollback button (not shown for latest)
                        with btn_cols[1]:
                            if not is_latest:
                                if st.button(
                                    "⏪ Restore",
                                    key=f"restore_{v}",
                                    use_container_width=True,
                                ):
                                    # Restore this version as the current doc
                                    st.session_state.results["markdown"] = ver["markdown"]
                                    st.session_state.results["full_markdown"] = ver["full_markdown"]
                                    st.session_state.doc_approved = False

                                    # Bump from the CURRENT latest (not the old version)
                                    # so the new version sorts above all existing ones
                                    new_ver = _save_doc_version(
                                        change_type="content_modified",
                                        description=f"Rolled back to v{v}",
                                        markdown=ver["markdown"],
                                        full_markdown=ver["full_markdown"],
                                    )
                                    st.toast(f"⏪ Restored v{v} content → saved as **v{new_ver}**")
                                    st.rerun()

                    # Expandable preview
                    preview_key = f"_preview_version_{selected_pid}"
                    if st.session_state.get(preview_key) == v:
                        with st.expander(f"📖 Preview — v{v}", expanded=True):
                            sub_prev, sub_changes, sub_raw = st.tabs(
                                ["📖 Preview", "🔄 Changes", "📝 Raw"]
                            )
                            with sub_prev:
                                st.markdown(ver["markdown"])
                            with sub_changes:
                                # Find the previous version to diff against
                                prev_ver = versions[i + 1] if i + 1 < len(versions) else None
                                if prev_ver is None:
                                    st.info(
                                        "🆕 This is the initial version — "
                                        "no previous version to compare against."
                                    )
                                else:
                                    st.caption(
                                        f"Comparing **v{prev_ver['version']}** → **v{v}**"
                                    )
                                    diff_html = _render_diff_html(
                                        prev_ver["markdown"], ver["markdown"]
                                    )
                                    st.markdown(diff_html, unsafe_allow_html=True)
                            with sub_raw:
                                st.code(ver["full_markdown"], language="markdown")
                            if st.button("Close preview", key=f"close_preview_{v}"):
                                del st.session_state[preview_key]
                                st.rerun()

                    if i < len(versions) - 1:
                        st.markdown(
                            '<hr style="margin: 0.5rem 0; border: none; '
                            'border-top: 1px dashed rgba(128,128,128,0.3);">',
                            unsafe_allow_html=True,
                        )

                # ── Danger zone ──
                st.markdown("---")
                st.markdown("### 🗑️ Danger Zone")
                st.caption(
                    f"Permanently delete **all {len(versions)} version(s)** of "
                    f"**{selected_pid}** and drop its collection from the database."
                )

                col_del, col_spacer = st.columns([1, 3])
                with col_del:
                    if st.button(
                        "🗑️ Delete Project & All Versions",
                        type="secondary",
                        use_container_width=True,
                        key="delete_all_versions",
                    ):
                        st.session_state._confirm_delete_project = selected_pid

                if st.session_state.get("_confirm_delete_project") == selected_pid:
                    st.warning(
                        f"⚠️ This will permanently delete **all {len(versions)} versions** "
                        f"of **{selected_pid}** and drop its collection. This cannot be undone."
                    )
                    col_yes, col_no, _ = st.columns([1, 1, 4])
                    with col_yes:
                        if st.button(
                            "Yes, delete",
                            type="primary",
                            key="confirm_delete_yes",
                            use_container_width=True,
                        ):
                            deleted = vs.delete_project(selected_pid)
                            st.session_state.pop("_confirm_delete_project", None)
                            st.toast(f"🗑️ Deleted {deleted} version(s) of **{selected_pid}**")
                            st.rerun()
                    with col_no:
                        if st.button(
                            "Cancel",
                            key="confirm_delete_no",
                            use_container_width=True,
                        ):
                            st.session_state.pop("_confirm_delete_project", None)
                            st.rerun()
