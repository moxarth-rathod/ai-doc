"""
AIDoc - AI-Powered Project Documentation Generator

Entry point for CLI generation, Streamlit UI, and interactive chat.

Usage:
    python main.py generate <repo_path>                        # generate docs (CLI)
    python main.py generate <repo_path> -o docs/OUT.md         # custom output
    python main.py generate <github_url>                       # from GitHub
    python main.py generate <repo_path> --target src/auth/     # targeted docs
    python main.py generate <repo_path> --model ...            # specify model
    python main.py ui                                          # launch Streamlit UI
    python main.py ui --port 8502                              # custom port
    python main.py chat <repo_path>                            # interactive Q&A
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prepare_doc",
        description="📄 AIDoc — AI-powered project documentation generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py generate ./my-project
  python main.py generate https://github.com/user/repo
  python main.py generate ./my-project -o MY_DOCS.md
  python main.py ui
  python main.py ui --port 8502
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── generate ───────────────────────────────
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate documentation for a project (CLI mode).",
    )
    gen_parser.add_argument(
        "repo_path",
        help="Path to the project directory or a GitHub URL to clone.",
    )
    gen_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path for the generated documentation. "
             "Defaults to 'output/DOCUMENTATION.md'.",
    )
    gen_parser.add_argument(
        "--model",
        default=None,
        help="Cerebras LLM model to use (default from .env or llama-3.3-70b).",
    )
    gen_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    gen_parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the documentation evaluation phase.",
    )
    gen_parser.add_argument(
        "--target",
        default=None,
        help="Target a specific folder or file for focused documentation "
             "(e.g., 'src/auth/' or 'core/payment.py').",
    )
    gen_parser.add_argument(
        "--full-repo",
        action="store_true",
        help="When using --target, index the full repo instead of only the "
             "targeted files. Slower but provides richer cross-project context.",
    )
    gen_parser.add_argument(
        "--reference-doc",
        default=None,
        help="Path to a reference documentation file (.md or .txt). "
             "The generated docs will match its structure, tone, and formatting.",
    )

    # ── ui ─────────────────────────────────────
    ui_parser = subparsers.add_parser(
        "ui",
        help="Launch the Streamlit web interface.",
    )
    ui_parser.add_argument(
        "--port",
        default="8501",
        help="Port to run Streamlit on (default: 8501).",
    )
    ui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open the browser automatically.",
    )

    # ── chat ──────────────────────────────────
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive Q&A chat about a project's codebase.",
    )
    chat_parser.add_argument(
        "repo_path",
        help="Path to the project directory or a GitHub URL to clone.",
    )
    chat_parser.add_argument(
        "--model",
        default=None,
        help="Cerebras LLM model to use (default from .env or llama-3.3-70b).",
    )

    return parser.parse_args()


def run_generate(
    repo_path: str,
    output_path: str | None = None,
    model: str | None = None,
    quiet: bool = False,
    skip_eval: bool = False,
    target: str | None = None,
    full_repo: bool = False,
    reference_doc_path: str | None = None,
) -> str:
    """
    Main pipeline: Load → Index → Discover → Write → Evaluate → Save.
    Returns the path to the generated documentation file.
    """
    # Lazy imports so `python main.py ui` stays fast
    from config.settings import AppConfig, LLMConfig
    from core.loader import RepoLoader
    from core.indexer import CodebaseIndexer
    from core.analyzer import HierarchicalAnalyzer
    from core.planner import ProjectPlanner
    from core.writer import DocumentationWriter
    from core.evaluator import DocumentationEvaluator

    start_time = time.time()

    # ── Configuration ──────────────────────────
    llm_config = LLMConfig(model=model) if model else LLMConfig()
    config = AppConfig(llm=llm_config, verbose=not quiet)

    if output_path:
        config.output_filename = os.path.basename(output_path)
        config.output_dir = os.path.dirname(output_path) or "."

    # ── Phase 1: Load the repository ──────────
    if not quiet:
        print("\n" + "=" * 60)
        print("📦 PHASE 1: Loading Repository")
        print("=" * 60)

    loader = RepoLoader(config.loader)
    local_path = loader.load(repo_path)

    try:
        files = loader.collect_files(local_path)
        file_tree = loader.get_file_tree(local_path)

        if not files:
            print("❌ No readable source files found in the repository.")
            sys.exit(1)

        if not quiet:
            print(f"📂 Found {len(files)} source files to analyze.")

        # Target-scoped files (for focused documentation)
        if target:
            target_files = loader.collect_files_for_target(local_path, target)
            if not target_files:
                print(f"❌ No files found for target: {target}")
                sys.exit(1)
            if not quiet:
                print(f"🎯 Target scope: {target} ({len(target_files)} files)")
        else:
            target_files = files

        # ── Phase 2: Index the codebase ───────────
        # When a target is specified, default to indexing only those files
        # unless --full-repo is explicitly requested.
        if target and not full_repo:
            files_to_index = target_files
            index_tree = loader.get_file_tree_for_target(local_path, target)
        else:
            files_to_index = files
            index_tree = file_tree

        if not quiet:
            print("\n" + "=" * 60)
            mode = "full repo" if (not target or full_repo) else "target files only"
            print(f"🧠 PHASE 2: Indexing Codebase ({len(files_to_index)} files — {mode})")
            print("=" * 60)

        indexer = CodebaseIndexer(config)
        index = indexer.index_codebase(files_to_index, index_tree)

        # ── Phase 2.5: Hierarchical Codebase Analysis ──
        if not quiet:
            print("\n" + "=" * 60)
            print("🏗️  PHASE 2.5: Hierarchical Codebase Analysis")
            print("=" * 60)

        ha = HierarchicalAnalyzer(index, files_to_index, index_tree)
        hierarchical_ctx = ha.analyze()

        if not quiet and hierarchical_ctx.system_summary:
            s = hierarchical_ctx.system_summary
            print(f"   🎯 North Star:    {s.north_star}")
            print(f"   🏛️  Architecture:  {s.architecture_style}")
            if s.heat_zones:
                hz_str = ", ".join(
                    f"{hz.name} ({hz.importance_score}/10)"
                    for hz in sorted(s.heat_zones, key=lambda h: h.importance_score, reverse=True)[:5]
                )
                print(f"   🔥 Heat Zones:    {hz_str}")
            if s.entry_points:
                print(f"   ⚡ Entry Points:  {', '.join(s.entry_points[:5])}")
            core_count = sum(1 for f in hierarchical_ctx.file_metadata if f.importance == "core")
            print(f"   📊 Files: {len(hierarchical_ctx.file_metadata)} total | {core_count} core | {len(hierarchical_ctx.module_summaries)} modules")

        # ── Phase 2b: Analyze reference doc (if provided) ──
        reference_style = None
        if reference_doc_path:
            from core.style_analyzer import DocStyleAnalyzer
            if not quiet:
                print("\n📎 Analyzing reference documentation style...")
            ref_content = Path(reference_doc_path).read_text(encoding="utf-8")
            reference_style = DocStyleAnalyzer.analyze(ref_content)
            if not quiet:
                print(
                    f"   Found {len(reference_style.sections)} sections | "
                    f"Tone: {reference_style.tone} | "
                    f"Depth: {reference_style.section_depth}"
                )

        # ── Phase 3: Discover & Plan ──────────────
        if not quiet:
            print("\n" + "=" * 60)
            print("🔍 PHASE 3: Analyzing Project & Planning Documentation")
            print("=" * 60)

        planner = ProjectPlanner(
            index,
            file_paths=[f["path"] for f in target_files],
            reference_style=reference_style,
            hierarchical_context=hierarchical_ctx,
        )
        analysis = planner.analyze(target_scope=target)

        if not quiet:
            planner.display_plan(analysis)

        # ── Phase 4: Write Documentation ──────────
        if not quiet:
            print("=" * 60)
            print("✍️  PHASE 4: Generating Documentation")
            print("=" * 60)

        writer = DocumentationWriter(
            index, analysis,
            source_files=target_files,
            reference_style=reference_style,
            hierarchical_context=hierarchical_ctx,
        )
        markdown = writer.generate()

        # ── Phase 5: Evaluate ─────────────────────
        if not skip_eval:
            if not quiet:
                print("\n" + "=" * 60)
                print("📊 PHASE 5: Evaluating Documentation")
                print("=" * 60)

            evaluator = DocumentationEvaluator(index, analysis)
            coverage = evaluator.compute_coverage(target_files, markdown)
            rating = evaluator.rate_quality(markdown, coverage)

            if not quiet:
                evaluator.display_cli_report(coverage, rating)

            # Append coverage report to the markdown
            coverage_section = evaluator.build_coverage_markdown(coverage, rating)
            markdown += "\n\n---\n\n" + coverage_section

        # ── Save output ───────────────────────────
        os.makedirs(config.output_dir, exist_ok=True)
        final_path = os.path.join(config.output_dir, config.output_filename)

        with open(final_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        elapsed = time.time() - start_time

        if not quiet:
            print(f"\n{'=' * 60}")
            print(f"✅ Documentation generated successfully!")
            print(f"   📄 Output: {os.path.abspath(final_path)}")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            print(f"   📑 Sections: {len(analysis.documentation_sections)}")
            print(f"{'=' * 60}\n")

        return os.path.abspath(final_path)

    finally:
        loader.cleanup(local_path)


def run_chat(repo_path: str, model: str | None = None):
    """Interactive Q&A session about a codebase."""
    # Lazy imports
    from config.settings import AppConfig, LLMConfig
    from core.loader import RepoLoader
    from core.indexer import CodebaseIndexer
    from core.chat import ChatEngine

    llm_config = LLMConfig(model=model) if model else LLMConfig()
    config = AppConfig(llm=llm_config)

    print("📦 Loading repository...")
    loader = RepoLoader(config.loader)
    local_path = loader.load(repo_path)

    try:
        files = loader.collect_files(local_path)
        file_tree = loader.get_file_tree(local_path)

        if not files:
            print("❌ No readable source files found.")
            sys.exit(1)

        print(f"📂 Found {len(files)} source files.")
        print("🧠 Indexing codebase...")
        indexer = CodebaseIndexer(config)
        index = indexer.index_codebase(files, file_tree)

        chat = ChatEngine(index)

        print(f"\n{'='*60}")
        print("💬 AIDoc Chat — Ask questions about the codebase")
        print("   Type 'quit' or 'exit' to end the session.")
        print("   Type 'clear' to reset conversation history.")
        print(f"{'='*60}\n")

        try:
            while True:
                try:
                    question = input("You: ").strip()
                except EOFError:
                    break

                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    break
                if question.lower() == "clear":
                    chat.reset()
                    print("🧹 Conversation history cleared.\n")
                    continue

                print("🤔 Thinking...")
                answer = chat.ask(question)
                print(f"\nAssistant: {answer}\n")
        except KeyboardInterrupt:
            pass

        print("\n👋 Goodbye!")

    finally:
        loader.cleanup(local_path)


def run_ui(port: str = "8501", no_browser: bool = False):
    """Launch the Streamlit web UI."""
    app_path = Path(__file__).resolve().parent / "app.py"

    if not app_path.exists():
        print(f"❌ Streamlit app not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", port,
    ]
    if no_browser:
        cmd.extend(["--server.headless", "true"])

    print(f"🚀 Launching AIDoc UI on port {port}...")
    print(f"   Open http://localhost:{port} in your browser.\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down AIDoc UI.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Install it with: pip install streamlit")
        sys.exit(1)


def main():
    args = parse_args()

    if args.command == "generate":
        run_generate(
            repo_path=args.repo_path,
            output_path=args.output,
            model=args.model,
            quiet=args.quiet,
            skip_eval=args.skip_eval,
            target=args.target,
            full_repo=args.full_repo,
            reference_doc_path=args.reference_doc,
        )
    elif args.command == "ui":
        run_ui(port=args.port, no_browser=args.no_browser)
    elif args.command == "chat":
        run_chat(repo_path=args.repo_path, model=args.model)
    else:
        # No command given — show help
        print("📄 AIDoc — AI-powered project documentation generator\n")
        print("Commands:")
        print("  python main.py generate <repo_path>   Generate documentation (CLI)")
        print("  python main.py ui                     Launch Streamlit web UI")
        print("  python main.py chat <repo_path>       Interactive Q&A about codebase")
        print("\nRun 'python main.py --help' for full usage details.")


if __name__ == "__main__":
    main()
