"""
Writer - Generates the final Markdown documentation based on the
planner's analysis. Each section is written independently using
RAG queries against the indexed codebase.

Enhanced with:
  - Direct File Injection: for each section, the actual source code of files
    mentioned in `focus_areas` is injected into the prompt so the LLM sees
    real code instead of relying solely on RAG retrieval.
  - Anti-Hallucination Rules: explicit prompt instructions that prevent the
    LLM from fabricating file paths, class names, or config values.
  - Post-Write Verification: lightweight check that flags file-path references
    in the generated text that do not correspond to real source files.

Also provides refinement: iteratively improving docs based on user feedback.
"""

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

from llama_index.core import Settings, VectorStoreIndex

from core.planner import ProjectAnalysis, DocumentationSection

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.style_analyzer import ReferenceStyle
    from core.analyzer import HierarchicalContext


# ──────────────────────────────────────────────
# Direct-context budget constants
# ──────────────────────────────────────────────
# These control how much raw source code is injected per section.
# The budget is PER SECTION, so even for a 1000-file repo the cost
# is bounded: each section injects only the 5-20 files it cares about.

_MAX_DIRECT_CONTEXT_CHARS = 40_000   # ~40 KB of source code per section
_MAX_LINES_PER_FILE = 300            # truncate very large files


# ──────────────────────────────────────────────
# File-path extraction pattern (used to pull paths from focus_areas)
# ──────────────────────────────────────────────

_FILE_PATH_RE = re.compile(
    r'(?:[\w./-]+/)?[\w.-]+\.(?:'
    r'py|js|ts|jsx|tsx|java|go|rs|rb|php|cs|cpp|c|h|hpp|swift|kt|kts|'
    r'scala|yaml|yml|toml|json|xml|html|css|scss|sql|sh|bash|md|txt|'
    r'cfg|ini|env|dockerfile|gradle|cmake|proto|graphql|vue|svelte'
    r')',
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
# Section writing prompts
# ──────────────────────────────────────────────

SECTION_PROMPT_TEMPLATE = """\
You are a senior technical writer creating the "{title}" section of comprehensive project \
documentation. You have full access to the indexed codebase via RAG.

**PROJECT CONTEXT:**
- Project: {project_name}
- Type: {project_type}
- Summary: {summary}
- Tech stack: {tech_stack}

**SECTION SCOPE:**
- Focus areas: {focus_areas}
- Target audience: {audience_description}
{file_list_block}
{direct_context_block}

**WRITING INSTRUCTIONS:**
1. DEEPLY ANALYZE the focus areas for this section. Don't write a shallow overview — \
provide thorough, detailed technical documentation that a developer or user can actually \
rely on.

2. For every concept you document, CITE the actual codebase:
   - Reference specific file paths: "In `path/to/file.py`..."
   - Name actual classes and functions: "The `ClassName.method()` handles..."
   - Show real configuration keys, environment variables, and constants
   - Describe actual parameters, return types, and behaviors from the code
   - If the code shows `class PaymentService` in `src/payments/service.py`, cite that \
     exactly — do not generalize or substitute.

3. EXPLAIN how things work, not just what they are:
   - Describe the data flow: what triggers what, where data enters, transforms, and exits
   - Explain architectural decisions: why the code is structured this way
   - Show component interactions: how this part connects to other parts of the system
   - Cover edge cases, error handling, and important configuration options

4. Each section is independent: use whatever examples are most relevant HERE, even if a \
different section uses different examples for the same concept. Precision > consistency.

5. Reference as many of the source files listed above as are relevant to THIS section. \
Mention file paths explicitly so readers know where to find things.

6. FORMATTING:
   - Use sub-headings (### level) to organize within this section
   - Use bullet points for lists of features, parameters, or options
   - Use code blocks (```) for file paths, class names, config keys, and short code snippets
   - Make it scannable — a reader should be able to skim and find what they need

7. If this section is for end-users, keep it action-oriented but still thorough.
   If for developers, include technical depth — explain implementation details, not just API surface.

8. Do NOT include the section title as a heading (## ...) — it will be added automatically.
9. Do NOT make up features that don't exist in the code. Only document what you can verify.
10. Aim for **300-800 words** — be comprehensive. If the topic is complex, go longer. \
Better to be thorough than to leave the reader guessing.

**ANTI-HALLUCINATION RULES — CRITICAL:**
- If GROUND TRUTH source code is provided above, use it as your PRIMARY source of facts. \
Every class name, function name, variable, environment variable, and config key you mention \
MUST appear in either the Ground Truth code or the RAG-retrieved context.
- NEVER invent file paths, class names, function names, environment variables, or \
configuration keys. If you are unsure about a specific detail, write \
"see `<relevant_file>` for details" instead of guessing.
- If you have NO evidence for a claim in the provided code, state that explicitly instead \
of fabricating content.
- Prefer being specific and correct over being comprehensive and wrong.

Write the content for this section now.
"""

AUDIENCE_MAP = {
    "developer": "software developers who will work on or contribute to this project",
    "end-user": "end-users who want to use this project/product",
    "both": "both developers and end-users who need to understand this project",
}


# ──────────────────────────────────────────────
# Refinement prompts
# ──────────────────────────────────────────────

REFINEMENT_PLAN_PROMPT = """\
You are a senior technical writer helping refine project documentation based on user feedback.

The documentation currently has these sections (with brief content summaries):
{section_list}

User feedback: "{feedback}"

Determine what changes to make. Think carefully about:
1. Does the user want to ADD entirely new content/sections?
2. Does the user want to MODIFY existing sections with more detail, examples, or corrections?
3. Does the user want to REMOVE sections?
4. Should modifications be DEEP (significant rewrite with new content from the codebase) \
or SHALLOW (small wording/structural tweaks)?

Return a JSON object (no markdown fences, no explanation):
{{
  "actions": [
    {{
      "type": "modify | remove | add",
      "section_title": "Exact Section Title to modify/remove, or new title for add",
      "instruction": "DETAILED description of what to change, what to add, what to focus on, \
and what codebase areas to query for supporting information"
    }}
  ]
}}

Rules:
- "modify": Rewrite the named section. The instruction MUST be detailed — specify what \
to add, elaborate, or change. Include which files, classes, or modules to investigate.
- "remove": Delete the named section entirely.
- "add": Create a new section. The instruction MUST describe scope, depth, and what \
codebase areas to analyze for content.
- You can list multiple actions if needed.
- For "modify", section_title must EXACTLY match one of the existing section titles.
- Make instructions specific and actionable — NOT vague like "improve this section". \
Instead say "Add detailed explanation of the authentication flow in auth/service.py, \
including how tokens are generated, validated, and refreshed. Include code references."
- Return ONLY the JSON object.
"""

SECTION_REWRITE_PROMPT = """\
You are a senior technical writer rewriting the "{title}" section of project documentation \
based on user feedback. You have full access to the indexed codebase via RAG.

**CURRENT CONTENT OF THIS SECTION:**
---
{current_content}
---

**WHAT THE USER WANTS:** {instruction}

**PROJECT CONTEXT:**
- Project: {project_name}
- Type: {project_type}
- Summary: {summary}
- Tech stack: {tech_stack}
{file_list_block}
{architecture_block}
{direct_context_block}

**REWRITING INSTRUCTIONS:**
1. DEEPLY ANALYZE the user's request. Don't just make surface-level edits — understand \
the intent and produce comprehensive, detailed content.
2. QUERY the codebase thoroughly. Pull in actual file paths, class names, function \
signatures, configuration keys, and data flows from the indexed code. Cite them precisely.
3. When adding or expanding content, write in-depth explanations with:
   - How the code actually works (cite specific files and functions)
   - Why it's designed that way (architectural decisions)
   - How different components interact (data flow, dependencies)
   - Concrete code references: "In `path/to/file.py`, the `ClassName.method()` handles..."
4. Preserve existing content that is correct and not affected by the user's request. \
Merge new content naturally with the existing material.
5. Use bullet points, sub-headings, and code blocks to make it scannable.
6. Aim for **300-800 words** — be thorough, not terse. If the topic warrants depth, \
go deeper. Better to be comprehensive than to leave the reader with questions.
7. Do NOT include the section title heading (## ...) — it will be added automatically.
8. Only cite code, paths, and features that ACTUALLY exist in the codebase.
9. Do NOT just repeat the user's question back as the answer — analyze, synthesize, and explain.

**ANTI-HALLUCINATION RULES — CRITICAL:**
- If GROUND TRUTH source code is provided above, treat it as your PRIMARY source of facts.
- NEVER invent file paths, class names, function names, or configuration keys.
- If you are unsure about a detail, write "see `<relevant_file>` for details" instead of guessing.

Write the improved section content now.
"""

NEW_SECTION_PROMPT = """\
You are a senior technical writer creating a NEW section titled "{title}" for project \
documentation. You have full access to the indexed codebase via RAG.

**USER INSTRUCTION:** {instruction}

**PROJECT CONTEXT:**
- Project: {project_name}
- Type: {project_type}
- Summary: {summary}
- Tech stack: {tech_stack}
{file_list_block}
{architecture_block}
{direct_context_block}

**WRITING INSTRUCTIONS:**
1. DEEPLY ANALYZE what the user wants this section to cover. Don't be shallow.
2. QUERY the codebase and pull in actual file paths, class names, function signatures, \
configuration keys, and real code references. Be precise and cite actual code.
3. Write a comprehensive, detailed section that:
   - Explains the topic thoroughly with references to actual source code
   - Shows how components interact (data flow, dependencies)
   - Uses concrete examples from the codebase: "In `path/to/file.py`, the `Function()` ..."
   - Covers edge cases, configuration options, and important details
4. Use bullet points, sub-headings, and code blocks to make it scannable.
5. Aim for **300-800 words** — thorough and detailed, not a brief summary.
6. Do NOT include the section title heading (## ...) — it will be added automatically.
7. Only cite code, paths, and features that ACTUALLY exist in the codebase.
8. Do NOT invent features. Only document what you can verify from the codebase.

**ANTI-HALLUCINATION RULES — CRITICAL:**
- If GROUND TRUTH source code is provided above, treat it as your PRIMARY source of facts.
- NEVER invent file paths, class names, function names, or configuration keys.
- If unsure, write "see `<relevant_file>` for details" instead of guessing.

Write the content for this section now.
"""


COVERAGE_GAP_PROMPT = """\
You are a senior technical writer covering files that were missed in the initial \
documentation. You have full access to the indexed codebase via RAG.

**PROJECT CONTEXT:**
- Project: {project_name}
- Type: {project_type}
- Summary: {summary}
- Tech stack: {tech_stack}

**UNCOVERED FILES** (NOT mentioned anywhere in existing documentation):
{uncovered_files}

**EXISTING SECTIONS** (already documented):
{existing_sections}
{direct_context_block}

**WRITING INSTRUCTIONS:**
Write a section titled "{section_title}" that thoroughly covers ALL of the uncovered files.

For each file or group of related files:
- State its path and explain its **purpose** in the system
- Describe **key classes, functions, or configuration** it provides — name them specifically
- Explain **how it relates to other parts of the project** (data flow, dependencies, imports)
- If it's a utility/helper, explain what other modules depend on it
- If it's a configuration file, explain what it configures and important default values

Requirements:
- You MUST mention EVERY file path from the uncovered list
- Use sub-headings (### level) to group related files
- Use bullet points for details within each file/group
- Aim for **200-600 words** — thorough coverage, not one-liners
- Do NOT include the section title heading (## ...)
- Only cite code that actually exists in the codebase

**ANTI-HALLUCINATION RULES — CRITICAL:**
- If GROUND TRUTH source code is provided above, treat it as your PRIMARY source of facts.
- NEVER invent class names, function names, or config keys — cite only what exists.
"""


class DocumentationWriter:
    """Generates final Markdown documentation from a project analysis."""

    def __init__(
        self,
        index: VectorStoreIndex,
        analysis: ProjectAnalysis,
        source_files: list[dict] = None,
        reference_style: "ReferenceStyle | None" = None,
        hierarchical_context: "HierarchicalContext | None" = None,
    ):
        self.index = index
        self.analysis = analysis
        self.source_files = source_files or []
        self.reference_style = reference_style
        self.hierarchical_context = hierarchical_context
        self.query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            similarity_top_k=20,
        )
        # Fast lookup dict: path → file info.  Built once, used per section.
        self._source_file_lookup: dict[str, dict] = {
            f["path"]: f for f in self.source_files
        }

    def generate(
        self,
        on_section_start: Optional[Callable[[int, int, str], None]] = None,
        on_section_done: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """
        Generate the complete documentation as a Markdown string.

        Args:
            on_section_start: Callback(current, total, title) called when starting a section.
            on_section_done: Callback(current, total, title) called when a section is done.
        """
        sections = []

        # Title and header
        header = self._build_header()
        sections.append(header)

        # Table of Contents
        toc = self._build_toc()
        sections.append(toc)

        # Generate each section
        total = len(self.analysis.documentation_sections)
        for i, section_plan in enumerate(self.analysis.documentation_sections, 1):
            if on_section_start:
                on_section_start(i, total, section_plan.title)
            else:
                print(f"  ✍️  [{i}/{total}] Writing: {section_plan.title}...")

            content = self._write_section(section_plan)
            sections.append(f"## {section_plan.title}\n\n{content}")

            if on_section_done:
                on_section_done(i, total, section_plan.title)

        # ── Coverage gap check ──
        # After writing all sections, verify every source file is referenced.
        # If any files are missing, generate a supplementary section to cover them.
        combined_so_far = "\n".join(sections)
        uncovered = self._find_uncovered_files(combined_so_far)
        if uncovered:
            gap_idx = total + 1
            if on_section_start:
                on_section_start(gap_idx, gap_idx, "Additional Files & Modules")
            else:
                print(
                    f"  ✍️  [{gap_idx}/{gap_idx}] "
                    f"Covering {len(uncovered)} missed file(s)..."
                )

            gap_content = self._write_coverage_gap_section(uncovered, sections)
            sections.append(
                f"## Additional Files & Modules\n\n{gap_content}"
            )

            if on_section_done:
                on_section_done(gap_idx, gap_idx, "Additional Files & Modules")

        # Footer
        sections.append(self._build_footer())

        # Rebuild TOC now that all sections (and their ### sub-headings) exist
        full_md = "\n\n---\n\n".join(sections)
        parsed = self._parse_sections(full_md)
        self._rebuild_toc(parsed)
        return self._reassemble_sections(parsed)

    def _build_header(self) -> str:
        """Build the document header."""
        lines = [
            f"# {self.analysis.project_name}",
            "",
            f"> {self.analysis.one_line_summary}",
            "",
            f"**Type:** {self.analysis.project_type}  ",
            f"**Language:** {self.analysis.primary_language}  ",
            f"**Stack:** {', '.join(self.analysis.frameworks_and_tools)}",
        ]
        return "\n".join(lines)

    def _build_toc(self) -> str:
        """Build a table of contents."""
        lines = ["## Table of Contents", ""]
        for i, section in enumerate(self.analysis.documentation_sections, 1):
            # Create anchor-friendly link
            anchor = section.title.lower().replace(" ", "-").replace("&", "").replace("/", "")
            anchor = anchor.replace("--", "-").strip("-")
            lines.append(f"{i}. [{section.title}](#{anchor})")
        return "\n".join(lines)

    def _build_style_block(self) -> str:
        """Build a style instruction block from the reference style, if available."""
        if not self.reference_style:
            return ""

        ref = self.reference_style
        return (
            f"\n--- STYLE MATCHING ---\n"
            f"You MUST match the style of the user's reference documentation:\n"
            f"- Tone: {ref.tone}\n"
            f"- Section depth: {ref.section_depth}\n"
            f"- Formatting: {ref.formatting_conventions}\n"
            f"- Style: {ref.style_notes}\n"
            f"Follow these conventions exactly. If the reference uses numbered steps "
            f"for procedures, you do too. If it uses tables, you do too. If it's brief "
            f"and punchy, be brief and punchy. Mirror the reference doc's personality.\n"
        )

    # ──────────────────────────────────────────
    # Direct file injection (anti-hallucination)
    # ──────────────────────────────────────────

    @staticmethod
    def _extract_focus_file_paths(text: str) -> list[str]:
        """
        Extract file paths mentioned in a string (typically focus_areas or instructions).

        Returns de-duplicated paths in the order they first appear.
        """
        matches = _FILE_PATH_RE.findall(text)
        return list(dict.fromkeys(matches))  # deduplicate, preserve order

    def _build_direct_context(self, reference_text: str) -> str:
        """
        Build a GROUND TRUTH block by injecting the actual source code of files
        mentioned in *reference_text* (e.g. a section's focus_areas).

        Strategy (scales to 500–1000+ file repos):
        - Only files referenced in the section's scope are injected — typically 5–20 files.
        - Each file is capped at _MAX_LINES_PER_FILE lines.
        - Total injected code is capped at _MAX_DIRECT_CONTEXT_CHARS characters.
        - The LLM prompt explicitly labels this block as "ground truth" so it takes
          priority over RAG-retrieved chunks.
        """
        if not self.source_files:
            return ""

        mentioned_paths = self._extract_focus_file_paths(reference_text)
        if not mentioned_paths:
            return ""

        lookup = self._source_file_lookup
        blocks: list[str] = []
        total_chars = 0

        for path_fragment in mentioned_paths:
            if total_chars >= _MAX_DIRECT_CONTEXT_CHARS:
                break

            # Try exact match first
            file_info = lookup.get(path_fragment)

            # Try suffix match (handles "auth/service.py" vs "src/auth/service.py")
            if not file_info:
                for full_path, info in lookup.items():
                    if full_path.endswith(path_fragment) or full_path.endswith("/" + path_fragment):
                        file_info = info
                        break

            if not file_info:
                continue

            content = file_info["content"]
            lines = content.split("\n")

            # Truncate very large files
            if len(lines) > _MAX_LINES_PER_FILE:
                content = "\n".join(lines[:_MAX_LINES_PER_FILE])
                content += (
                    f"\n\n... (truncated — {len(lines)} total lines, "
                    f"showing first {_MAX_LINES_PER_FILE})"
                )

            # Respect context budget
            if total_chars + len(content) > _MAX_DIRECT_CONTEXT_CHARS:
                remaining = _MAX_DIRECT_CONTEXT_CHARS - total_chars
                if remaining > 500:  # only include if meaningful amount
                    content = content[:remaining] + "\n... (truncated to fit context budget)"
                else:
                    continue

            blocks.append(f"--- FILE: {file_info['path']} ---\n{content}")
            total_chars += len(content)

        if not blocks:
            return ""

        return (
            "\n**GROUND TRUTH — ACTUAL SOURCE CODE:**\n"
            "The following is the real source code from files relevant to this section. "
            "Treat this as your PRIMARY source of truth. Base your documentation on what "
            "you see here. Do NOT contradict or fabricate details beyond what this code "
            "and the RAG-retrieved context show.\n\n"
            + "\n\n".join(blocks)
            + "\n"
        )

    def _verify_section(self, content: str) -> str:
        """
        Lightweight post-write verification that flags file-path references in the
        generated text when they don't correspond to any real source file.

        This catches the most common hallucination pattern: the LLM inventing
        plausible-sounding file paths that don't actually exist.

        For performance, this is pure regex + set-lookup — zero LLM calls.
        """
        if not self.source_files:
            return content

        # Build sets for fast membership checks
        known_paths_lower: set[str] = set()
        known_names_lower: set[str] = set()
        known_stems_lower: set[str] = set()

        for f in self.source_files:
            p = f["path"]
            known_paths_lower.add(p.lower())
            known_names_lower.add(Path(p).name.lower())
            stem = Path(p).stem.lower()
            if len(stem) > 3:
                known_stems_lower.add(stem)

        # Find backtick-quoted file path references
        ref_pattern = re.compile(
            r'`((?:[\w./-]+/)?[\w.-]+\.(?:'
            r'py|js|ts|jsx|tsx|java|go|rs|rb|php|cs|cpp|c|h|hpp|swift|'
            r'kt|scala|yaml|yml|toml|json|xml|html|css|sql|sh|bash|md|'
            r'txt|cfg|ini|env|dockerfile|gradle|cmake'
            r'))`',
            re.IGNORECASE,
        )

        hallucinated: list[str] = []
        for match in ref_pattern.finditer(content):
            ref = match.group(1)
            ref_lower = ref.lower()

            # Exact path match
            if ref_lower in known_paths_lower:
                continue
            # Filename match
            name_lower = Path(ref).name.lower()
            if name_lower in known_names_lower:
                continue
            # Stem match (e.g., "planner" matches "planner.py")
            stem_lower = Path(ref).stem.lower()
            if len(stem_lower) > 3 and stem_lower in known_stems_lower:
                continue
            # Substring match (covers partial paths)
            if any(ref_lower in p for p in known_paths_lower):
                continue

            hallucinated.append(ref)

        # Tag hallucinated references so they stand out in review
        for h_path in hallucinated:
            content = content.replace(
                f"`{h_path}`",
                f"`{h_path}` *(unverified)*",
            )

        return content

    # ──────────────────────────────────────────
    # Section writing
    # ──────────────────────────────────────────

    def _write_section(self, section: DocumentationSection) -> str:
        """Write a single documentation section using RAG + direct file injection."""
        audience_desc = AUDIENCE_MAP.get(section.audience, AUDIENCE_MAP["both"])

        # Build file list block so the LLM knows ALL files in scope
        file_list_block = ""
        if self.source_files:
            file_paths = [f["path"] for f in self.source_files]
            cap = 100
            file_list_block = (
                "\nSource files in scope (reference relevant ones in this section):\n"
                + "\n".join(f"  - {p}" for p in file_paths[:cap])
            )
            if len(file_paths) > cap:
                file_list_block += f"\n  ... and {len(file_paths) - cap} more files"
            file_list_block += "\n"

        # Build direct source code injection from focus_areas file paths
        direct_context_block = self._build_direct_context(section.focus_areas)

        prompt = SECTION_PROMPT_TEMPLATE.format(
            title=section.title,
            project_name=self.analysis.project_name,
            project_type=self.analysis.project_type,
            summary=self.analysis.one_line_summary,
            tech_stack=", ".join(self.analysis.frameworks_and_tools),
            focus_areas=section.focus_areas,
            audience_description=audience_desc,
            file_list_block=file_list_block,
            direct_context_block=direct_context_block,
        )

        # Inject hierarchical context (module summaries, heat zones, data flow)
        prompt += self._build_hierarchical_block(section.title)

        # Append style matching instructions if reference doc was provided
        prompt += self._build_style_block()

        response = self.query_engine.query(prompt)
        content = str(response).strip()

        # Post-write verification: flag hallucinated file paths
        content = self._verify_section(content)

        return content

    def _build_hierarchical_block(self, section_title: str) -> str:
        """Build a hierarchical context block relevant to the current section."""
        if not self.hierarchical_context:
            return ""

        ctx = self.hierarchical_context
        lines: list[str] = ["\n--- ARCHITECTURE CONTEXT ---"]

        # Include system summary essentials
        if ctx.system_summary:
            s = ctx.system_summary
            if s.north_star:
                lines.append(f"Project mission: {s.north_star}")
            if s.architecture_style:
                lines.append(f"Architecture: {s.architecture_style}")
            if s.data_flow:
                lines.append(f"Data flow: {s.data_flow}")

            if s.heat_zones:
                lines.append(
                    "Heat zones: "
                    + "; ".join(
                        f"{hz.name} ({hz.directory})"
                        for hz in sorted(
                            s.heat_zones,
                            key=lambda h: h.importance_score,
                            reverse=True,
                        )[:5]
                    )
                )

        # Include module summaries (compact form)
        if ctx.module_summaries:
            lines.append("Module landscape:")
            for m in ctx.module_summaries:
                lines.append(f"  {m.directory}: {m.purpose}")

        lines.append(
            "Use this architecture context to write more accurate, "
            "architecture-aware content for this section."
        )

        return "\n".join(lines) + "\n"

    def _find_uncovered_files(self, markdown: str) -> list[str]:
        """Return source file paths that are NOT referenced anywhere in the markdown."""
        if not self.source_files:
            return []

        markdown_lower = markdown.lower()
        uncovered = []

        for f in self.source_files:
            path = f["path"]
            path_lower = path.lower()

            # Check direct path reference
            if path_lower in markdown_lower:
                continue

            # Check filename only
            filename = Path(path).name.lower()
            if filename in markdown_lower:
                continue

            # Check stem (skip very short names to avoid false positives)
            stem = Path(path).stem.lower()
            if len(stem) > 3:
                pattern = r"\b" + re.escape(stem) + r"\b"
                if re.search(pattern, markdown_lower):
                    continue

            # Check module-style path (e.g., core.planner)
            module_path = path_lower.replace("/", ".").replace("\\", ".")
            for ext in (".py", ".js", ".ts", ".java", ".go", ".rs"):
                module_path = module_path.replace(ext, "")
            if module_path in markdown_lower:
                continue

            uncovered.append(path)

        return uncovered

    def _write_coverage_gap_section(
        self, uncovered_files: list[str], existing_sections: list[str]
    ) -> str:
        """Generate documentation for uncovered files."""
        # Build a summary of existing section titles
        section_titles = []
        for s in existing_sections:
            match = re.match(r"^## (.+?)(?:\n|$)", s)
            if match:
                section_titles.append(match.group(1).strip())

        # Build direct context from the actual uncovered files
        uncovered_text = " ".join(uncovered_files)
        direct_context_block = self._build_direct_context(uncovered_text)

        prompt = COVERAGE_GAP_PROMPT.format(
            project_name=self.analysis.project_name,
            project_type=self.analysis.project_type,
            summary=self.analysis.one_line_summary,
            tech_stack=", ".join(self.analysis.frameworks_and_tools),
            uncovered_files="\n".join(f"  - {p}" for p in uncovered_files),
            existing_sections="\n".join(f"  - {t}" for t in section_titles),
            section_title="Additional Files & Modules",
            direct_context_block=direct_context_block,
        )
        prompt += self._build_style_block()

        response = self.query_engine.query(prompt)
        content = str(response).strip()
        content = self._verify_section(content)
        return content

    def _build_footer(self) -> str:
        """Build the document footer."""
        lines = [
            "",
            "*This documentation was auto-generated by analyzing the project codebase. "
            "While it aims to be accurate, always refer to the source code for the "
            "most up-to-date information.*",
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Refinement — iterative doc improvement
    # ──────────────────────────────────────────

    def refine(self, markdown: str, feedback: str) -> str:
        """
        Apply user feedback to refine the existing documentation.

        Parses the markdown into sections, determines which sections to
        modify/remove/add, applies changes via RAG, and reassembles.

        Args:
            markdown: The current full documentation markdown.
            feedback: User's refinement instruction.

        Returns:
            Updated markdown string.
        """
        # Parse current markdown into sections
        sections = self._parse_sections(markdown)
        section_titles = [s["title"] for s in sections if s["title"]]

        # Build section list with content previews (first 150 chars) for better planning
        section_summaries = []
        for s in sections:
            if s["title"]:
                preview = s["content"][:150].replace("\n", " ").strip()
                if len(s["content"]) > 150:
                    preview += "..."
                section_summaries.append(f"  - **{s['title']}**: {preview}")

        # Step 1: Ask LLM what changes to make (with content awareness)
        plan_prompt = REFINEMENT_PLAN_PROMPT.format(
            section_list="\n".join(section_summaries),
            feedback=feedback,
        )
        plan_response = Settings.llm.complete(plan_prompt)
        plan_text = str(plan_response)

        try:
            json_str = self._extract_json(plan_text)
            plan = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat the entire doc as needing modification
            plan = {"actions": [{
                "type": "modify",
                "section_title": section_titles[0] if section_titles else "",
                "instruction": feedback,
            }]}

        actions = plan.get("actions", [])
        if not actions:
            return markdown  # No changes needed

        # Step 2: Apply each action
        for action in actions:
            action_type = action.get("type", "modify")
            title = action.get("section_title", "")
            instruction = action.get("instruction", feedback)

            if action_type == "remove":
                sections = [s for s in sections if s["title"] != title]

            elif action_type == "modify":
                for s in sections:
                    if s["title"] == title:
                        new_content = self._rewrite_section(
                            title, s["content"], instruction
                        )
                        s["content"] = new_content
                        break

            elif action_type == "add":
                new_content = self._write_new_section(title, instruction)
                # Insert before the footer
                insert_idx = len(sections) - 1  # before last element (footer)
                sections.insert(insert_idx, {
                    "title": title,
                    "content": new_content,
                    "raw": f"## {title}\n\n{new_content}",
                })

        # Step 3: Rebuild the Table of Contents to reflect added/removed sections
        self._rebuild_toc(sections)

        # Step 4: Reassemble markdown
        return self._reassemble_sections(sections)

    def _build_file_list_block(self) -> str:
        """Build a file list block so the LLM knows all source files in scope."""
        if not self.source_files:
            return ""
        file_paths = [f["path"] for f in self.source_files]
        cap = 100
        block = (
            "\n**Source files in the codebase** (reference relevant ones):\n"
            + "\n".join(f"  - {p}" for p in file_paths[:cap])
        )
        if len(file_paths) > cap:
            block += f"\n  ... and {len(file_paths) - cap} more files"
        return block + "\n"

    def _build_architecture_block(self) -> str:
        """Build a compact architecture context block for refinement prompts."""
        return self._build_hierarchical_block("refinement")

    def _rewrite_section(self, title: str, current_content: str, instruction: str) -> str:
        """Rewrite a single section using RAG + direct file injection + user instruction."""
        # Build direct context from both the instruction and existing content
        combined_ref = instruction + "\n" + current_content
        direct_context_block = self._build_direct_context(combined_ref)

        prompt = SECTION_REWRITE_PROMPT.format(
            title=title,
            current_content=current_content,
            instruction=instruction,
            project_name=self.analysis.project_name,
            project_type=self.analysis.project_type,
            summary=self.analysis.one_line_summary,
            tech_stack=", ".join(self.analysis.frameworks_and_tools),
            file_list_block=self._build_file_list_block(),
            architecture_block=self._build_architecture_block(),
            direct_context_block=direct_context_block,
        )
        prompt += self._build_style_block()
        response = self.query_engine.query(prompt)
        content = str(response).strip()
        content = self._verify_section(content)
        return content

    def _write_new_section(self, title: str, instruction: str) -> str:
        """Write a brand new section using RAG + direct file injection."""
        direct_context_block = self._build_direct_context(instruction)

        prompt = NEW_SECTION_PROMPT.format(
            title=title,
            instruction=instruction,
            project_name=self.analysis.project_name,
            project_type=self.analysis.project_type,
            summary=self.analysis.one_line_summary,
            tech_stack=", ".join(self.analysis.frameworks_and_tools),
            file_list_block=self._build_file_list_block(),
            architecture_block=self._build_architecture_block(),
            direct_context_block=direct_context_block,
        )
        prompt += self._build_style_block()
        response = self.query_engine.query(prompt)
        content = str(response).strip()
        content = self._verify_section(content)
        return content

    @staticmethod
    def _parse_sections(markdown: str) -> list[dict]:
        """
        Parse a markdown document into a list of section dicts.
        Each dict has: title (str or None for preamble), content (str), raw (str).
        """
        sections = []
        # Split on section dividers (--- between sections)
        parts = re.split(r"\n---\n", markdown)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this part starts with a ## heading
            heading_match = re.match(r"^## (.+?)(?:\n|$)", part)
            if heading_match:
                title = heading_match.group(1).strip()
                content = part[heading_match.end():].strip()
            else:
                title = None  # preamble (header, TOC) or footer
                content = part

            sections.append({
                "title": title,
                "content": content,
                "raw": part,
            })

        return sections

    @staticmethod
    def _make_anchor(text: str) -> str:
        """Convert a heading text to a GitHub-style markdown anchor."""
        anchor = text.lower().strip()
        anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)   # remove special chars
        anchor = re.sub(r"\s+", "-", anchor)             # spaces → hyphens
        anchor = re.sub(r"-+", "-", anchor).strip("-")   # collapse multiple hyphens
        return anchor

    @staticmethod
    def _rebuild_toc(sections: list[dict]) -> None:
        """
        Find the Table of Contents section and rebuild it from current sections.
        Includes ## section titles and ### sub-headings found in section content.
        Modifies the sections list in-place.
        """
        toc_idx = None
        for i, s in enumerate(sections):
            if s.get("title") == "Table of Contents":
                toc_idx = i
                break

        if toc_idx is None:
            return  # No TOC section found — nothing to rebuild

        # Collect all titled sections (skip TOC itself)
        toc_lines = [""]
        num = 1
        for s in sections:
            title = s.get("title")
            if not title or title == "Table of Contents":
                continue

            anchor = DocumentationWriter._make_anchor(title)
            toc_lines.append(f"{num}. [{title}](#{anchor})")

            # Scan content for ### sub-headings
            # Use 4-space indent + standard "1." numbering for valid nested Markdown lists
            sub_num = 1
            for match in re.finditer(r"^###\s+(.+)", s.get("content", ""), re.MULTILINE):
                sub_title = match.group(1).strip()
                sub_anchor = DocumentationWriter._make_anchor(sub_title)
                toc_lines.append(f"    {sub_num}. [{sub_title}](#{sub_anchor})")
                sub_num += 1

            num += 1

        # Update the TOC section in-place
        new_toc_content = "\n".join(toc_lines)
        sections[toc_idx]["content"] = new_toc_content
        sections[toc_idx]["raw"] = f"## Table of Contents\n{new_toc_content}"

    @staticmethod
    def _reassemble_sections(sections: list[dict]) -> str:
        """Reassemble parsed sections back into a markdown string."""
        parts = []
        for s in sections:
            if s["title"]:
                parts.append(f"## {s['title']}\n\n{s['content']}")
            else:
                parts.append(s["raw"])
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract a JSON object from text."""
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        brace_start = text.find("{")
        if brace_start == -1:
            raise ValueError("No JSON object found.")

        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start:i + 1]

        return text[brace_start:]
