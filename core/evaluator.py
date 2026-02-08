"""
Evaluator - Analyzes the generated documentation for coverage and quality.
Computes which files are referenced, rates the documentation, and produces
visual-ready metrics.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Callable

from pydantic import BaseModel, Field
from llama_index.core import Settings, VectorStoreIndex

from core.planner import ProjectAnalysis


# ──────────────────────────────────────────────
# Coverage data structures
# ──────────────────────────────────────────────

@dataclass
class FileCoverage:
    """Coverage status of a single file."""
    path: str
    category: str  # source, config, test, documentation, dependencies
    covered: bool


@dataclass
class CoverageReport:
    """Complete coverage analysis of the documentation."""
    total_files: int
    covered_files: int
    uncovered_files: int
    coverage_pct: float
    by_category: dict  # category -> {"total": int, "covered": int, "pct": float}
    by_directory: dict  # dir -> {"total": int, "covered": int, "pct": float}
    file_details: list  # list of FileCoverage
    covered_list: list  # paths of covered files
    uncovered_list: list  # paths of uncovered files


# ──────────────────────────────────────────────
# Quality rating models (LLM structured output)
# ──────────────────────────────────────────────

class QualityRating(BaseModel):
    """AI-assessed quality rating of the documentation."""
    coverage_score: int = Field(
        description="0-100: How much of the project's key functionality and files are documented"
    )
    correctness_score: int = Field(
        description="0-100: How accurate the documentation claims are based on actual code"
    )
    clarity_score: int = Field(
        description="0-100: How readable, well-organized, and easy to follow the documentation is"
    )
    completeness_score: int = Field(
        description="0-100: Whether all important aspects (setup, usage, config, architecture) are covered"
    )
    usefulness_score: int = Field(
        description="0-100: How useful this documentation would be for a new developer or end-user"
    )
    overall_score: int = Field(
        description="0-100: Overall quality score considering all factors"
    )
    strengths: List[str] = Field(
        description="3-5 specific things the documentation does well"
    )
    gaps: List[str] = Field(
        description="2-4 specific areas that are missing or could be improved"
    )
    verdict: str = Field(
        description="One sentence overall verdict on the documentation quality"
    )


# ──────────────────────────────────────────────
# Rating prompt
# ──────────────────────────────────────────────

RATING_PROMPT = """\
You are a Senior Technical Writer reviewing auto-generated documentation.

**Project context:**
- Name: {project_name}
- Type: {project_type}
- Language: {primary_language}
- Stack: {tech_stack}
- Total source files: {total_files}
- Files referenced in docs: {covered_files}

**Coverage stats:**
- Overall coverage: {coverage_pct:.0f}%
- Categories: {category_summary}

**The generated documentation:**
{documentation}

Rate this documentation honestly across these dimensions (0-100 each):
1. Coverage, 2. Correctness, 3. Clarity, 4. Completeness, 5. Usefulness

Also identify 3-5 strengths, 2-4 gaps, and give a one-sentence verdict.

Typical good docs score 65-85. Deduct for missing files, vague statements, or gaps.
"""

RATING_STRUCTURING_PROMPT = """\
Convert the following documentation review into a JSON object. \
Return ONLY valid JSON, no markdown, no explanation, no code fences.

The JSON must match this exact schema:
{{
  "coverage_score": integer_0_to_100,
  "correctness_score": integer_0_to_100,
  "clarity_score": integer_0_to_100,
  "completeness_score": integer_0_to_100,
  "usefulness_score": integer_0_to_100,
  "overall_score": integer_0_to_100,
  "strengths": ["string", "string", "string"],
  "gaps": ["string", "string"],
  "verdict": "one sentence string"
}}

Here is the review to convert:

{raw_review}
"""


# ──────────────────────────────────────────────
# JSON extraction helper
# ──────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain markdown fences or extra text."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]

    return text[brace_start:]


# ──────────────────────────────────────────────
# Evaluator class
# ──────────────────────────────────────────────

class DocumentationEvaluator:
    """Evaluates generated documentation for coverage and quality."""

    def __init__(self, index: VectorStoreIndex, analysis: ProjectAnalysis):
        self.index = index
        self.analysis = analysis

    # ── Coverage Analysis ──────────────────────

    def compute_coverage(
        self,
        all_files: list[dict],
        markdown: str,
    ) -> CoverageReport:
        """
        Compute how many source files are referenced in the documentation.

        Args:
            all_files: List of {"path": ..., "content": ...} from loader.
            markdown: The generated documentation markdown string.
        """
        markdown_lower = markdown.lower()
        file_details = []

        for f in all_files:
            path = f["path"]
            category = self._classify_file(path)

            covered = self._is_file_covered(path, markdown_lower)
            file_details.append(FileCoverage(
                path=path,
                category=category,
                covered=covered,
            ))

        total = len(file_details)
        covered_count = sum(1 for f in file_details if f.covered)
        uncovered_count = total - covered_count
        coverage_pct = (covered_count / total * 100) if total > 0 else 0

        # Group by category
        by_category = defaultdict(lambda: {"total": 0, "covered": 0})
        for f in file_details:
            by_category[f.category]["total"] += 1
            if f.covered:
                by_category[f.category]["covered"] += 1

        for cat, data in by_category.items():
            data["pct"] = (data["covered"] / data["total"] * 100) if data["total"] > 0 else 0

        # Group by top-level directory
        by_directory = defaultdict(lambda: {"total": 0, "covered": 0})
        for f in file_details:
            parts = Path(f.path).parts
            dir_name = parts[0] if len(parts) > 1 else "root"
            by_directory[dir_name]["total"] += 1
            if f.covered:
                by_directory[dir_name]["covered"] += 1

        for dir_name, data in by_directory.items():
            data["pct"] = (data["covered"] / data["total"] * 100) if data["total"] > 0 else 0

        return CoverageReport(
            total_files=total,
            covered_files=covered_count,
            uncovered_files=uncovered_count,
            coverage_pct=coverage_pct,
            by_category=dict(by_category),
            by_directory=dict(by_directory),
            file_details=file_details,
            covered_list=[f.path for f in file_details if f.covered],
            uncovered_list=[f.path for f in file_details if not f.covered],
        )

    def _is_file_covered(self, path: str, markdown_lower: str) -> bool:
        """Check if a file is referenced in the documentation."""
        path_lower = path.lower()

        # Direct path reference
        if path_lower in markdown_lower:
            return True

        # Filename only (without directory)
        filename = Path(path).name.lower()
        if filename in markdown_lower:
            return True

        # Filename without extension for common references
        stem = Path(path).stem.lower()
        if len(stem) > 3:  # skip short names to avoid false positives
            pattern = r'\b' + re.escape(stem) + r'\b'
            if re.search(pattern, markdown_lower):
                return True

        # Module path (e.g., "core.planner" for "core/planner.py")
        module_path = path_lower.replace("/", ".").replace("\\", ".")
        for ext in (".py", ".js", ".ts", ".java", ".go", ".rs"):
            module_path = module_path.replace(ext, "")
        if module_path in markdown_lower:
            return True

        return False

    # ── Quality Rating ─────────────────────────

    def rate_quality(
        self,
        markdown: str,
        coverage: CoverageReport,
    ) -> QualityRating:
        """
        Use the LLM to rate the documentation quality.
        Two-step approach: RAG query for raw review → LLM to structure as JSON.
        """
        print("⭐ Rating documentation quality...")

        category_summary = ", ".join(
            f"{cat}: {data['covered']}/{data['total']} ({data['pct']:.0f}%)"
            for cat, data in coverage.by_category.items()
        )

        # Truncate markdown if very long to stay within context limits
        doc_text = markdown[:8000] if len(markdown) > 8000 else markdown

        prompt = RATING_PROMPT.format(
            project_name=self.analysis.project_name,
            project_type=self.analysis.project_type,
            primary_language=self.analysis.primary_language,
            tech_stack=", ".join(self.analysis.frameworks_and_tools),
            total_files=coverage.total_files,
            covered_files=coverage.covered_files,
            coverage_pct=coverage.coverage_pct,
            category_summary=category_summary,
            documentation=doc_text,
        )

        # Step 1: Get raw review via RAG (no output_cls)
        query_engine = self.index.as_query_engine(
            response_mode="tree_summarize",
            similarity_top_k=10,
        )
        raw_result = query_engine.query(prompt)
        raw_text = str(raw_result)

        # Step 2: Convert raw review into structured JSON via direct LLM call
        structuring_prompt = RATING_STRUCTURING_PROMPT.format(raw_review=raw_text)
        response = Settings.llm.complete(structuring_prompt)
        response_text = str(response)

        try:
            json_str = _extract_json(response_text)
            data = json.loads(json_str)
            rating = QualityRating.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Retry with stricter prompt
            print("⚠️  Retrying structured parsing for rating...")
            retry_prompt = (
                "Return ONLY a raw JSON object (no markdown, no ```). "
                f"Schema: {QualityRating.model_json_schema()}\n\n"
                f"Review:\n{raw_text}"
            )
            retry_response = Settings.llm.complete(retry_prompt)
            json_str = _extract_json(str(retry_response))
            data = json.loads(json_str)
            rating = QualityRating.model_validate(data)

        return rating

    # ── Coverage Report for Markdown ───────────

    def build_coverage_markdown(
        self,
        coverage: CoverageReport,
        rating: QualityRating,
    ) -> str:
        """
        Build a Markdown section summarizing coverage and quality.
        This gets appended to the generated documentation.
        """
        lines = [
            "## 📊 Documentation Coverage & Quality Report",
            "",
            "### Quality Scores",
            "",
            "| Dimension | Score |",
            "|-----------|-------|",
            f"| 📁 Coverage | **{rating.coverage_score}/100** |",
            f"| ✅ Correctness | **{rating.correctness_score}/100** |",
            f"| 📖 Clarity | **{rating.clarity_score}/100** |",
            f"| 🧩 Completeness | **{rating.completeness_score}/100** |",
            f"| 💡 Usefulness | **{rating.usefulness_score}/100** |",
            f"| ⭐ **Overall** | **{rating.overall_score}/100** |",
            "",
            f"> *{rating.verdict}*",
            "",
            "### Coverage Statistics",
            "",
            f"- **{coverage.covered_files}/{coverage.total_files}** files referenced "
            f"({coverage.coverage_pct:.0f}% coverage)",
            "",
            "#### By Category",
            "",
            "| Category | Covered | Total | Coverage |",
            "|----------|---------|-------|----------|",
        ]

        for cat, data in sorted(coverage.by_category.items()):
            bar = self._text_bar(data["pct"])
            lines.append(
                f"| {cat.title()} | {data['covered']} | {data['total']} | {bar} {data['pct']:.0f}% |"
            )

        lines.extend([
            "",
            "#### By Directory",
            "",
            "| Directory | Covered | Total | Coverage |",
            "|-----------|---------|-------|----------|",
        ])

        for dir_name, data in sorted(coverage.by_directory.items()):
            bar = self._text_bar(data["pct"])
            lines.append(
                f"| `{dir_name}/` | {data['covered']} | {data['total']} | {bar} {data['pct']:.0f}% |"
            )

        # Strengths & Gaps
        lines.extend([
            "",
            "### ✅ Strengths",
            "",
        ])
        for s in rating.strengths:
            lines.append(f"- {s}")

        lines.extend([
            "",
            "### ⚠️ Areas for Improvement",
            "",
        ])
        for g in rating.gaps:
            lines.append(f"- {g}")

        return "\n".join(lines)

    def _text_bar(self, pct: float, width: int = 10) -> str:
        """Create a text-based progress bar."""
        filled = int(pct / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    # ── CLI display ────────────────────────────

    def display_cli_report(
        self,
        coverage: CoverageReport,
        rating: QualityRating,
    ):
        """Print coverage and quality report to terminal."""
        print(f"\n{'='*60}")
        print("📊 DOCUMENTATION EVALUATION")
        print(f"{'='*60}")

        # Quality scores
        print("\n  ⭐ Quality Scores:")
        scores = [
            ("Coverage", rating.coverage_score),
            ("Correctness", rating.correctness_score),
            ("Clarity", rating.clarity_score),
            ("Completeness", rating.completeness_score),
            ("Usefulness", rating.usefulness_score),
        ]
        for name, score in scores:
            bar = self._text_bar(score, 20)
            print(f"    {name:15s} {bar} {score}/100")
        print(f"    {'─'*45}")
        bar = self._text_bar(rating.overall_score, 20)
        print(f"    {'OVERALL':15s} {bar} {rating.overall_score}/100")

        # Verdict
        print(f"\n  💬 {rating.verdict}")

        # Coverage stats
        print(f"\n  📁 File Coverage: {coverage.covered_files}/{coverage.total_files} "
              f"({coverage.coverage_pct:.0f}%)")

        print("\n  By Category:")
        for cat, data in sorted(coverage.by_category.items()):
            bar = self._text_bar(data["pct"], 15)
            print(f"    {cat.title():15s} {bar} {data['covered']}/{data['total']} ({data['pct']:.0f}%)")

        # Strengths
        print("\n  ✅ Strengths:")
        for s in rating.strengths:
            print(f"    • {s}")

        # Gaps
        print("\n  ⚠️  Areas for Improvement:")
        for g in rating.gaps:
            print(f"    • {g}")

        print(f"\n{'='*60}")

    # ── Helpers ─────────────────────────────────

    def _classify_file(self, path: str) -> str:
        """Classify a file into a category."""
        lower = path.lower()
        if any(name in lower for name in ("readme", "changelog", "contributing", "license")):
            return "documentation"
        elif any(name in lower for name in (
            "requirements", "package.json", "cargo.toml", "go.mod",
            "gemfile", "pom.xml", "build.gradle", "setup.py", "pyproject.toml",
        )):
            return "dependencies"
        elif any(name in lower for name in (
            "dockerfile", "docker-compose", "makefile", "procfile",
        )):
            return "configuration"
        elif any(name in lower for name in (".yml", ".yaml", ".toml", ".json", ".env", ".cfg")):
            if any(name in lower for name in ("config", "settings", ".env")):
                return "configuration"
        if any(name in lower for name in ("test", "spec", "__test__")):
            return "test"
        return "source"
