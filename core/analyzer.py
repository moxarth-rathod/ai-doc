"""
Hierarchical Codebase Analyzer — moves from flat retrieval to layered understanding.

Processes the codebase in three levels:
  Level 1 (Files):   Extract metadata — classes, functions, imports, entry-point flags.
  Level 2 (Modules): Summarize what each directory/package contributes.
  Level 3 (System):  Synthesize the project's "North Star", heat zones, and data flow.

The resulting HierarchicalContext is fed into the Planner and Writer for
architecture-aware documentation that scales to hundreds of files.

References:
  - "Documentation Pyramid": Project > Folder > File
  - "Ghost Index": search file names and function signatures, not raw code text
  - "Agentic Drill-Down": multi-step analysis loop
  - "Entry-Point Bias": prioritize main/routes/controllers/models
  - "Importance Filter": focus on top 25% unique business-logic files
"""

import json
import os
import re
from collections import defaultdict
from typing import Callable, List, Optional

from pydantic import BaseModel, Field
from llama_index.core import Settings, VectorStoreIndex


# ──────────────────────────────────────────────
# Data models (the Documentation Pyramid)
# ──────────────────────────────────────────────

class FileMetadata(BaseModel):
    """Level-1: metadata extracted from a single source file (zero LLM calls)."""
    path: str
    classes: List[str] = Field(default_factory=list)
    functions: List[str] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    is_entry_point: bool = False
    importance: str = "supporting"  # core | supporting | config | test | boilerplate
    line_count: int = 0


class ModuleSummary(BaseModel):
    """Level-2: summary of what a directory/package contributes."""
    directory: str
    purpose: str
    key_files: List[str] = Field(default_factory=list)
    interactions: str = ""  # How this module connects to others


class HeatZone(BaseModel):
    """A 'hot' area of the codebase where the most important logic lives."""
    name: str
    directory: str
    description: str
    importance_score: int = Field(default=5, ge=1, le=10)
    key_files: List[str] = Field(default_factory=list)


class SystemSummary(BaseModel):
    """Level-3: global understanding of the project."""
    north_star: str = ""  # The ONE thing this repo does
    architecture_style: str = ""  # MVC, microservices, monolith, etc.
    entry_points: List[str] = Field(default_factory=list)
    heat_zones: List[HeatZone] = Field(default_factory=list)
    data_flow: str = ""  # How data flows from entry to storage
    developer_first_hour: List[str] = Field(default_factory=list)  # Files to open first


class HierarchicalContext(BaseModel):
    """Complete hierarchical analysis passed to Planner & Writer."""
    file_metadata: List[FileMetadata] = Field(default_factory=list)
    module_summaries: List[ModuleSummary] = Field(default_factory=list)
    system_summary: Optional[SystemSummary] = None

    # ── Prompt helpers ────────────────────────

    def build_prompt_block(self) -> str:
        """Build a compact text block for injection into LLM prompts."""
        lines: list[str] = []
        if self.system_summary:
            s = self.system_summary
            lines.append("=== HIERARCHICAL CODEBASE ANALYSIS ===")
            if s.north_star:
                lines.append(f"NORTH STAR: {s.north_star}")
            if s.architecture_style:
                lines.append(f"ARCHITECTURE: {s.architecture_style}")
            if s.entry_points:
                lines.append(f"ENTRY POINTS: {', '.join(s.entry_points)}")
            if s.data_flow:
                lines.append(f"DATA FLOW: {s.data_flow}")

            if s.heat_zones:
                lines.append("\nHEAT ZONES (most critical areas):")
                for hz in sorted(
                    s.heat_zones,
                    key=lambda h: h.importance_score,
                    reverse=True,
                ):
                    lines.append(
                        f"  [{hz.importance_score}/10] {hz.name} "
                        f"({hz.directory}): {hz.description}"
                    )

            if s.developer_first_hour:
                lines.append(
                    f"\nDEVELOPER'S FIRST HOUR: "
                    f"{', '.join(s.developer_first_hour)}"
                )

        if self.module_summaries:
            lines.append("\nMODULE LANDSCAPE:")
            for m in self.module_summaries:
                lines.append(f"  📂 {m.directory}: {m.purpose}")
                if m.interactions:
                    lines.append(f"     ↔ {m.interactions}")

        return "\n".join(lines)

    def get_importance_sorted_files(self) -> list[str]:
        """Return file paths sorted by importance (core first)."""
        order = {"core": 0, "supporting": 1, "config": 2, "test": 3, "boilerplate": 4}
        ranked = sorted(
            self.file_metadata,
            key=lambda f: order.get(f.importance, 5),
        )
        return [f.path for f in ranked]

    def get_heat_zone_files(self) -> list[str]:
        """Return file paths that fall inside heat zones."""
        if not self.system_summary:
            return []
        files: list[str] = []
        for hz in self.system_summary.heat_zones:
            files.extend(hz.key_files)
        return list(dict.fromkeys(files))  # deduplicate, preserve order


# ──────────────────────────────────────────────
# Constants for metadata extraction
# ──────────────────────────────────────────────

ENTRY_POINT_PATTERNS = [
    r"(?:^|/)main\.(py|go|rs|java|kt|scala)$",
    r"(?:^|/)app\.(py|js|ts|jsx|tsx)$",
    r"(?:^|/)index\.(js|ts|jsx|tsx)$",
    r"(?:^|/)server\.(py|js|ts)$",
    r"(?:^|/)manage\.py$",
    r"(?:^|/)wsgi\.py$",
    r"(?:^|/)asgi\.py$",
    r"(?:^|/)cli\.(py|js|ts)$",
    r"(?:^|/)__main__\.py$",
]

ENTRY_POINT_DIRS = frozenset({
    "routes", "controllers", "api", "handlers", "endpoints", "views",
})

CORE_DIRS = frozenset({
    "core", "src", "lib", "app", "services", "models", "domain",
    "engine", "modules", "components",
})

CONFIG_PATTERNS = frozenset({
    "settings", "config", ".env", "constants", "defaults",
})

TEST_PATTERNS = frozenset({
    "test", "tests", "spec", "specs", "__tests__", "__test__",
})

BOILERPLATE_FILES = frozenset({
    "__init__.py", "setup.py", "setup.cfg", "conftest.py",
})


# ──────────────────────────────────────────────
# Signature extractors (zero LLM calls)
# ──────────────────────────────────────────────

_CLASS_RE = {
    "python":     re.compile(r"^class\s+(\w+)", re.MULTILINE),
    "javascript": re.compile(r"(?:class|interface)\s+(\w+)", re.MULTILINE),
    "typescript": re.compile(r"(?:class|interface|type|enum)\s+(\w+)", re.MULTILINE),
    "java":       re.compile(r"(?:class|interface|enum)\s+(\w+)", re.MULTILINE),
    "go":         re.compile(r"type\s+(\w+)\s+struct", re.MULTILINE),
    "rust":       re.compile(r"(?:struct|enum|trait)\s+(\w+)", re.MULTILINE),
    "ruby":       re.compile(r"class\s+(\w+)", re.MULTILINE),
    "kotlin":     re.compile(r"(?:class|interface|object|enum)\s+(\w+)", re.MULTILINE),
    "scala":      re.compile(r"(?:class|trait|object|case class)\s+(\w+)", re.MULTILINE),
    "c":          re.compile(r"typedef\s+struct\s+(\w+)", re.MULTILINE),
    "cpp":        re.compile(r"(?:class|struct)\s+(\w+)", re.MULTILINE),
    "php":        re.compile(r"class\s+(\w+)", re.MULTILINE),
    "swift":      re.compile(r"(?:class|struct|protocol|enum)\s+(\w+)", re.MULTILINE),
}

_FUNC_RE = {
    "python":     re.compile(r"^(?:def|async def)\s+(\w+)\s*\(", re.MULTILINE),
    "javascript": re.compile(
        r"(?:function\s+(\w+)"
        r"|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"(?:function\s+(\w+)"
        r"|(?:const|let|var)\s+(\w+)\s*(?::\s*[\w<>\[\]|]+)?\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
    "java":       re.compile(
        r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+(\w+)\s*\(",
        re.MULTILINE,
    ),
    "go":         re.compile(r"func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(", re.MULTILINE),
    "rust":       re.compile(r"(?:pub\s+)?fn\s+(\w+)", re.MULTILINE),
    "ruby":       re.compile(r"def\s+(\w+)", re.MULTILINE),
    "kotlin":     re.compile(r"fun\s+(\w+)", re.MULTILINE),
    "scala":      re.compile(r"def\s+(\w+)", re.MULTILINE),
    "c":          re.compile(r"[\w*]+\s+(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
    "cpp":        re.compile(r"[\w*:&<>]+\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?\{", re.MULTILINE),
    "php":        re.compile(r"function\s+(\w+)\s*\(", re.MULTILINE),
    "swift":      re.compile(r"func\s+(\w+)", re.MULTILINE),
}

_IMPORT_RE = {
    "python":     re.compile(r"^(?:from\s+(\S+)\s+)?import\s+(.+)", re.MULTILINE),
    "javascript": re.compile(
        r"(?:import\s+.+\s+from\s+['\"](.+?)['\"]"
        r"|require\(['\"](.+?)['\"]\))",
        re.MULTILINE,
    ),
    "typescript": re.compile(r"import\s+.+\s+from\s+['\"](.+?)['\"]", re.MULTILINE),
    "java":       re.compile(r"import\s+([\w.]+);", re.MULTILINE),
    "go":         re.compile(r'"([\w./]+)"', re.MULTILINE),
    "rust":       re.compile(r"use\s+([\w:]+)", re.MULTILINE),
}

_EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".php": "php", ".swift": "swift",
}


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

MODULE_ANALYSIS_PROMPT = """\
You are a Senior Architect analyzing a codebase. Below is a structured summary of \
ALL directories and their files with extracted metadata (classes, functions, imports, \
importance scores).

DIRECTORY STRUCTURE:
{directory_structure}

FILE TREE:
{file_tree}

Analyze this codebase and return a JSON object (no markdown fences, no explanation):
{{
  "modules": [
    {{
      "directory": "path/to/dir",
      "purpose": "One-sentence summary of what this module/directory contributes",
      "key_files": ["file1.py", "file2.py"],
      "interactions": "How this module connects to other parts of the system"
    }}
  ],
  "system": {{
    "north_star": "The ONE thing this repository does — its core mission",
    "architecture_style": "e.g., MVC, microservices, monolith, pipeline, CLI tool, etc.",
    "entry_points": ["path/to/main.py", "path/to/app.py"],
    "heat_zones": [
      {{
        "name": "Short label",
        "directory": "path/to/dir",
        "description": "Why this is a critical area",
        "importance_score": 8,
        "key_files": ["path/to/file1.py"]
      }}
    ],
    "data_flow": "Describe how data flows from the entry point through the system to storage/output",
    "developer_first_hour": ["path/to/file1.py", "path/to/file2.py"]
  }}
}}

RULES:
1. Identify the TOP 5 "Heat Zones" — the areas where the most important business logic lives.
2. For heat_zones, importance_score ranges from 1 (minor) to 10 (critical core logic).
3. developer_first_hour: List the 5-10 files a new developer should read FIRST.
4. Ignore boilerplate (__init__.py, setup.py, etc.) in heat zones unless they contain real logic.
5. Every directory that contains source files MUST appear in the modules list.
6. Return ONLY the JSON object — no text before or after it.
"""


# ──────────────────────────────────────────────
# Hierarchical Analyzer
# ──────────────────────────────────────────────

class HierarchicalAnalyzer:
    """
    Analyzes a codebase in layers (File → Module → System) to produce
    a rich context that guides the documentation planner and writer.

    This replaces the flat "throw everything at the AI" approach with
    a structured understanding that scales to hundreds of files.

    Cost: 1 LLM-via-RAG call (+ 1 retry on parse failure).
    Speed: ~5-15 s depending on repo size.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        files: list[dict],
        file_tree: str,
    ):
        self.index = index
        self.files = files
        self.file_tree = file_tree

    def analyze(
        self,
        on_progress: Optional[Callable] = None,
    ) -> HierarchicalContext:
        """
        Run the full hierarchical analysis.

        Returns a HierarchicalContext with:
        - file_metadata: per-file classes, functions, imports, importance
        - module_summaries: per-directory purpose and interactions
        - system_summary: north star, heat zones, data flow, entry points
        """
        total_steps = 3

        # ── Level 1: File metadata (zero LLM calls — instant) ──
        if on_progress:
            on_progress(1, total_steps, "Extracting file signatures and metadata…")
        print("📐 Level 1: Extracting file signatures (zero LLM calls)…")
        file_metadata = self._extract_all_metadata()

        core_count = sum(1 for f in file_metadata if f.importance == "core")
        entry_count = sum(1 for f in file_metadata if f.is_entry_point)
        print(
            f"   → {len(file_metadata)} files | "
            f"{core_count} core | "
            f"{entry_count} entry points"
        )

        # ── Level 2 + 3: Module & System synthesis (1–2 LLM calls) ──
        if on_progress:
            on_progress(2, total_steps, "Synthesizing module structure and heat zones…")
        print("🧩 Level 2+3: Synthesizing modules and system understanding…")
        dir_structure = self._build_directory_structure(file_metadata)
        modules, system = self._synthesize_modules_and_system(
            dir_structure, file_metadata
        )

        if system:
            hz_names = [hz.name for hz in system.heat_zones[:5]]
            print(f"   → North Star: {system.north_star[:80]}")
            print(f"   → Architecture: {system.architecture_style}")
            print(f"   → Heat Zones: {', '.join(hz_names)}")
            print(f"   → Modules analyzed: {len(modules)}")

        if on_progress:
            on_progress(3, total_steps, "Hierarchical analysis complete")

        return HierarchicalContext(
            file_metadata=file_metadata,
            module_summaries=modules,
            system_summary=system,
        )

    # ──────────────────────────────────────────
    # Level 1: Zero-LLM metadata extraction
    # ──────────────────────────────────────────

    def _extract_all_metadata(self) -> list[FileMetadata]:
        """Extract classes, functions, imports from every file using regex."""
        results: list[FileMetadata] = []
        for f in self.files:
            path = f["path"]
            content = f["content"]
            ext = os.path.splitext(path)[1].lower()
            lang = _EXT_TO_LANG.get(ext)

            classes = self._extract_names(_CLASS_RE.get(lang), content) if lang else []
            functions = self._extract_names(_FUNC_RE.get(lang), content) if lang else []
            imports = self._extract_names(_IMPORT_RE.get(lang), content) if lang else []

            is_entry = self._is_entry_point(path)
            importance = self._classify_importance(path, content, is_entry)

            results.append(FileMetadata(
                path=path,
                classes=classes[:20],
                functions=functions[:30],
                imports=imports[:20],
                is_entry_point=is_entry,
                importance=importance,
                line_count=content.count("\n") + 1,
            ))

        return results

    @staticmethod
    def _extract_names(pattern: "re.Pattern | None", content: str) -> list[str]:
        """Extract names from regex matches, handling groups that may be None."""
        if not pattern:
            return []
        names: list[str] = []
        for m in pattern.finditer(content):
            for g in m.groups():
                if g:
                    names.append(g)
                    break
        return names

    @staticmethod
    def _is_entry_point(path: str) -> bool:
        """Detect if a file is an entry point based on name and directory patterns."""
        path_lower = path.lower().replace("\\", "/")
        for pattern in ENTRY_POINT_PATTERNS:
            if re.search(pattern, path_lower):
                return True
        parts = set(path_lower.split("/"))
        return bool(parts & ENTRY_POINT_DIRS)

    @staticmethod
    def _classify_importance(path: str, content: str, is_entry: bool) -> str:
        """Classify a file's importance level without any LLM calls."""
        path_lower = path.lower().replace("\\", "/")
        filename = os.path.basename(path_lower)

        if is_entry:
            return "core"

        parts = set(path_lower.split("/"))

        if parts & TEST_PATTERNS:
            return "test"
        if parts & CORE_DIRS:
            return "core"

        # Config patterns
        if any(p in path_lower for p in CONFIG_PATTERNS):
            return "config"

        # Boilerplate
        if filename in BOILERPLATE_FILES:
            if len(content.strip()) < 50:
                return "boilerplate"

        return "supporting"

    # ──────────────────────────────────────────
    # Level 2 + 3: LLM-powered synthesis
    # ──────────────────────────────────────────

    def _build_directory_structure(
        self, file_metadata: list[FileMetadata]
    ) -> str:
        """Build a structured summary of files grouped by directory for the LLM.

        This is the "Ghost Index" — the LLM sees file names, classes, and function
        signatures instead of raw code, making it dramatically cheaper and more
        accurate for architectural understanding.
        """
        dirs: dict[str, list[FileMetadata]] = defaultdict(list)
        for fm in file_metadata:
            directory = os.path.dirname(fm.path) or "root"
            dirs[directory].append(fm)

        lines: list[str] = []
        for directory in sorted(dirs.keys()):
            files_in_dir = dirs[directory]
            lines.append(f"\n📂 {directory}/  ({len(files_in_dir)} files)")
            for fm in files_in_dir:
                tag = f"[{fm.importance.upper()}]"
                entry_flag = " ⚡ENTRY" if fm.is_entry_point else ""
                lines.append(
                    f"  {tag}{entry_flag} {os.path.basename(fm.path)} "
                    f"({fm.line_count} lines)"
                )
                if fm.classes:
                    lines.append(f"    Classes: {', '.join(fm.classes[:10])}")
                if fm.functions:
                    lines.append(f"    Functions: {', '.join(fm.functions[:15])}")
                if fm.imports:
                    lines.append(f"    Imports: {', '.join(fm.imports[:10])}")

        return "\n".join(lines)

    def _synthesize_modules_and_system(
        self,
        dir_structure: str,
        file_metadata: list[FileMetadata],
    ) -> tuple[list[ModuleSummary], "SystemSummary | None"]:
        """
        Use a direct LLM call to synthesize module summaries and system understanding.

        This is the Agentic Drill-Down: we already have the structural metadata
        (Level 1 — the "Ghost Index"), so we do NOT need RAG retrieval here.
        A direct LLM.complete() call avoids the context-window overflow that
        happens when LlamaIndex stuffs both the large prompt AND retrieved
        code chunks into the context.
        """
        # Truncate inputs to stay within context limits (~6k chars each).
        # The Ghost Index (dir_structure) is more valuable than the raw tree.
        max_dir = 6000
        max_tree = 2000
        truncated_dir = dir_structure[:max_dir]
        if len(dir_structure) > max_dir:
            truncated_dir += f"\n  … and more (truncated from {len(dir_structure)} chars)"
        truncated_tree = self.file_tree[:max_tree]
        if len(self.file_tree) > max_tree:
            truncated_tree += f"\n  … and more (truncated from {len(self.file_tree)} chars)"

        prompt = MODULE_ANALYSIS_PROMPT.format(
            directory_structure=truncated_dir,
            file_tree=truncated_tree,
        )

        # Direct LLM call — no RAG needed; the Ghost Index IS the context
        print("   Using direct LLM call (Ghost Index approach)…")
        response = Settings.llm.complete(prompt)
        response_text = str(response)

        try:
            json_str = _extract_json(response_text)
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Retry with a stricter instruction
            print("⚠️  Retrying hierarchical analysis with stricter prompt…")
            retry_response = Settings.llm.complete(
                "Return ONLY a raw JSON object (no markdown, no ```).\n\n" + prompt
            )
            try:
                json_str = _extract_json(str(retry_response))
                data = json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                print("⚠️  Hierarchical JSON parsing failed — using fallback.")
                return self._build_fallback(file_metadata)

        return self._parse_synthesis_response(data, file_metadata)

    def _parse_synthesis_response(
        self, data: dict, file_metadata: list[FileMetadata]
    ) -> tuple[list[ModuleSummary], SystemSummary]:
        """Parse the JSON response into ModuleSummary and SystemSummary objects."""
        # Parse modules
        modules: list[ModuleSummary] = []
        for m in data.get("modules", []):
            modules.append(ModuleSummary(
                directory=m.get("directory", ""),
                purpose=m.get("purpose", ""),
                key_files=m.get("key_files", []),
                interactions=m.get("interactions", ""),
            ))

        # Parse system
        sys_data = data.get("system", {})
        heat_zones: list[HeatZone] = []
        for hz in sys_data.get("heat_zones", []):
            heat_zones.append(HeatZone(
                name=hz.get("name", ""),
                directory=hz.get("directory", ""),
                description=hz.get("description", ""),
                importance_score=min(10, max(1, int(hz.get("importance_score", 5)))),
                key_files=hz.get("key_files", []),
            ))

        system = SystemSummary(
            north_star=sys_data.get("north_star", ""),
            architecture_style=sys_data.get("architecture_style", ""),
            entry_points=sys_data.get("entry_points", []),
            heat_zones=heat_zones,
            data_flow=sys_data.get("data_flow", ""),
            developer_first_hour=sys_data.get("developer_first_hour", []),
        )

        return modules, system

    def _build_fallback(
        self, file_metadata: list[FileMetadata]
    ) -> tuple[list[ModuleSummary], SystemSummary]:
        """Build minimal summaries when LLM parsing fails entirely."""
        dirs: dict[str, list[FileMetadata]] = defaultdict(list)
        for fm in file_metadata:
            directory = os.path.dirname(fm.path) or "root"
            dirs[directory].append(fm)

        modules = [
            ModuleSummary(
                directory=d,
                purpose=f"Contains {len(fms)} file(s)",
                key_files=[f.path for f in fms if f.importance == "core"][:5],
            )
            for d, fms in dirs.items()
        ]

        entry_points = [f.path for f in file_metadata if f.is_entry_point]

        system = SystemSummary(
            north_star="(Could not determine — see source code for details)",
            architecture_style="Unknown",
            entry_points=entry_points,
            heat_zones=[],
            data_flow="",
            developer_first_hour=entry_points[:5],
        )

        return modules, system


# ──────────────────────────────────────────────
# JSON helper (shared pattern across modules)
# ──────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain markdown fences."""
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
                return text[brace_start : i + 1]

    return text[brace_start:]

