"""
Planner - Discovery phase that analyzes the project and creates
a custom documentation plan based on what actually matters.
"""

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field
from llama_index.core import Settings, VectorStoreIndex

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.style_analyzer import ReferenceStyle
    from core.analyzer import HierarchicalContext


# ──────────────────────────────────────────────
# Structured output models
# ──────────────────────────────────────────────

class DocumentationSection(BaseModel):
    """A single section that should appear in the generated documentation."""
    title: str = Field(description="Section heading for the documentation")
    reason: str = Field(description="Why this section is important for this project")
    focus_areas: str = Field(
        description="Specific things to cover in this section based on the actual code"
    )
    audience: str = Field(
        description="Who this section is for: 'developer', 'end-user', or 'both'"
    )


class ProjectAnalysis(BaseModel):
    """Complete analysis of a project that drives documentation generation."""
    project_name: str = Field(description="Human-readable project name")
    project_type: str = Field(
        description="Type of project: e.g., 'Web Application', 'CLI Tool', 'Library', 'API Service', 'Data Pipeline', etc."
    )
    one_line_summary: str = Field(
        description="One sentence that captures what this project does"
    )
    primary_language: str = Field(description="The main programming language used")
    frameworks_and_tools: List[str] = Field(
        description="Key frameworks, libraries, and tools the project depends on"
    )
    documentation_sections: List[DocumentationSection] = Field(
        description="The custom table of contents - 5 to 8 sections that this specific project needs documented"
    )


# ──────────────────────────────────────────────
# Discovery prompts
# ──────────────────────────────────────────────

DISCOVERY_PROMPT = """\
You are a Lead Engineer who just inherited this project. Analyze the codebase \
thoroughly and produce a structured analysis.

Your task:
1. **Identify the project**: What does it do? What type of software is it?
2. **Detect the tech stack**: Languages, frameworks, databases, APIs, external services.
3. **Find integrations**: Third-party services, payment gateways, auth providers, \
   message queues, cloud services, etc.
4. **Assess configuration**: How is the project configured? Env vars, config files, etc.
5. **Understand the architecture**: Entry points, key modules, data flow.
6. **Discover what makes this project unique**: What are its core concepts, key \
   workflows, and distinctive features that a reader MUST understand?

Based on your findings, design a CUSTOM documentation structure (5-8 sections) that \
is **unique to THIS project**. The sections should read like a hand-crafted README \
written by the original author — not a generic template.

CRITICAL — Dynamic Section Design:
- DO NOT fall back on generic template names like "Project Overview", "Getting Started", \
  "Configuration", "Architecture", "Utilities and Helpers". These are boring and tell \
  the reader nothing about what makes this project special.
- Instead, INVENT section titles that capture this project's actual concepts, workflows, \
  and components. The section titles alone should tell someone what this project is about.
- Think about what a developer or user would ACTUALLY search for when using this project.

Examples of GOOD project-specific sections (for different projects):
  • A payment gateway: "Payment Flow & Lifecycle", "Webhook Event Handling", \
    "Fraud Detection Pipeline", "Merchant Onboarding"
  • A web scraper: "Crawling Strategy & Rate Limiting", "Data Extraction Pipelines", \
    "Storage Backends", "Proxy Rotation & Anti-Detection"
  • A documentation generator: "Generation Pipeline", "Codebase Indexing & RAG", \
    "AI-Driven Section Planning", "Quality Evaluation & Coverage"
  • A chat application: "Real-Time Messaging Architecture", "User Presence & Typing \
    Indicators", "Message Persistence & Search", "Push Notifications"
  • A CLI tool: "Command Reference & Usage", "Plugin System", "Output Formatting"

Examples of BAD generic sections (avoid these):
  ✗ "Project Overview" → Instead: "What Is <ProjectName> & Why It Exists"
  ✗ "Getting Started" → Instead: "Quick Start & Installation" or roll into Overview
  ✗ "Configuration" → Instead: "Environment & Runtime Settings" or merge into relevant sections
  ✗ "Utilities and Helpers" → Instead: name the actual utilities, or merge into relevant sections
  ✗ "Architecture" → Instead: "System Design & Data Flow" or "<Specific>Pipeline Architecture"

Rules:
- The FIRST section should always orient the reader: what is this project, why does it \
  exist, and how to get it running. Give it a project-specific title.
- Every other section must be justified by actual code you found — not by a template.
- Section titles should be descriptive and specific to the domain of this project.
- If a concept is small (e.g. a few env vars), fold it into a related section rather \
  than giving it its own generic section.
- Audience should be practical: a new developer joining the team or an end-user trying to use it.

IMPORTANT — "Senior Auditor" Rule:
You may be looking at a repo with hundreds of files, but you only have a limited \
attention budget. Identify the top 25% of files that contain the **Unique Business \
Logic** and build the documentation primarily from those. Libraries, boilerplate, and \
standard framework code should be mentioned only when necessary for understanding.

IMPORTANT: Be specific in focus_areas. Don't say "explain how auth works" - say \
"JWT tokens via middleware in auth/jwt.py, OAuth2 with Google in auth/oauth.py, \
role-based access in decorators/permissions.py". Each section's focus_areas should list \
the actual file paths, classes, and functions that belong in that section.
"""

STRUCTURING_PROMPT = """\
Convert the following project analysis into a JSON object. Return ONLY valid JSON, \
no markdown, no explanation, no code fences.

The JSON must match this exact schema:
{{
  "project_name": "string",
  "project_type": "string (e.g. Web Application, CLI Tool, Library, API Service)",
  "one_line_summary": "string (one sentence)",
  "primary_language": "string",
  "frameworks_and_tools": ["string"],
  "documentation_sections": [
    {{
      "title": "string — must be project-specific, NOT generic template names",
      "reason": "string",
      "focus_areas": "string (be very specific: list actual file paths, classes, and functions relevant to this section)",
      "audience": "developer | end-user | both"
    }}
  ]
}}

Include 5-8 documentation_sections. Each section's focus_areas must reference specific \
files and modules from the project — never use generic placeholders.

SECTION TITLE RULES:
- Section titles must be specific to THIS project's domain and concepts.
- AVOID generic names: "Project Overview", "Getting Started", "Configuration", \
  "Architecture", "Utilities and Helpers". These are template filler.
- GOOD titles capture the project's actual concepts: e.g. "Generation Pipeline", \
  "AI-Driven Section Planning", "Codebase Indexing & RAG", "Quality Evaluation".
- The first section should orient the reader (what this is + how to run it) but \
  still have a project-specific title, not just "Project Overview".
- Small topics (e.g. a few config vars) should be folded into related sections, \
  not given their own generic "Configuration" section.

CRITICAL: Every file in the project must appear in at least one section's focus_areas. \
If any file is missing, add it to the most relevant section. 100% file coverage is required.

Here is the analysis to convert:

{raw_analysis}
"""


# ──────────────────────────────────────────────
# JSON extraction helper
# ──────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain markdown fences or extra text."""
    # Try to find JSON in code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a JSON object directly
    brace_start = text.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found in LLM response.")

    # Find the matching closing brace
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start:i + 1]

    # Fallback: take everything from the first brace
    return text[brace_start:]


class ProjectPlanner:
    """Analyzes a codebase and produces a documentation plan."""

    def __init__(
        self,
        index: VectorStoreIndex,
        file_paths: list[str] = None,
        reference_style: "ReferenceStyle | None" = None,
        hierarchical_context: "HierarchicalContext | None" = None,
    ):
        self.index = index
        self.file_paths = file_paths or []
        self.reference_style = reference_style
        self.hierarchical_context = hierarchical_context

    def analyze(
        self,
        target_scope: str = None,
        on_progress: "Callable[[int, int, str], None] | None" = None,
    ) -> ProjectAnalysis:
        """
        Run the discovery phase using a two-step approach:
        1. RAG query to get raw analysis text from the codebase
        2. Separate LLM call to structure the response into JSON

        Args:
            target_scope: Optional path to focus analysis on a specific folder/file.
            on_progress: Optional callback(step, total_steps, description) for progress.
        """
        total_steps = 3  # discovery, structuring, parsing

        if target_scope:
            print(f"🔍 Analyzing target scope: {target_scope}")
        else:
            print("🔍 Analyzing project DNA...")

        # Build discovery prompt (optionally scoped to a target)
        prompt = DISCOVERY_PROMPT

        # ── Inject hierarchical context (if available) ──
        # This gives the LLM a pre-analysed understanding of the architecture:
        # entry points, heat zones, module purposes, data flow.
        if self.hierarchical_context:
            ctx_block = self.hierarchical_context.build_prompt_block()
            if ctx_block.strip():
                prompt += (
                    f"\n\n{ctx_block}\n\n"
                    "The above hierarchical analysis has already been performed. "
                    "USE it to guide your documentation plan — prioritise the "
                    "Heat Zones and Entry Points. The module landscape tells you "
                    "what each directory does; use that knowledge to create precise, "
                    "architecture-aware sections rather than generic ones."
                )

        # Provide the complete file list so the LLM knows ALL files in scope
        if self.file_paths:
            cap = 200
            paths_str = "\n".join(f"  - {p}" for p in self.file_paths[:cap])
            prompt += (
                f"\n\n--- COMPLETE FILE LIST ({len(self.file_paths)} files in scope) ---\n"
                f"{paths_str}"
            )
            if len(self.file_paths) > cap:
                prompt += f"\n  ... and {len(self.file_paths) - cap} more files"
            prompt += (
                "\n\nCRITICAL REQUIREMENT — 100% FILE COVERAGE:\n"
                "Every single file listed above MUST appear in at least one section's "
                "focus_areas. No file may be left out. After creating your section plan, "
                "mentally check each file against your sections — if any file is not "
                "covered, add it to the most relevant section's focus_areas. "
                "Config files, utility files, helper modules, and test files all count. "
                "If a file doesn't fit neatly into any section, create an appropriate "
                "section for it (e.g., 'Utilities & Helpers' or 'Testing')."
            )

        if target_scope:
            prompt += (
                f"\n\nIMPORTANT: Focus your analysis specifically on the code within "
                f"'{target_scope}'. This is a targeted documentation request — document "
                f"only the modules, features, and architecture within that scope. "
                f"The project name should reflect this focus "
                f"(e.g., 'ProjectName — {target_scope} module')."
            )

        # Inject reference documentation structure if provided
        if self.reference_style:
            ref = self.reference_style
            # Build the section structure from the reference
            ref_sections = [
                s for s in ref.sections if s.depth <= 2  # Only top-level sections
            ]
            if ref_sections:
                section_list = "\n".join(
                    f"  {i}. {s.title} — {s.summary[:80]}"
                    for i, s in enumerate(ref_sections, 1)
                )
                prompt += (
                    f"\n\n--- REFERENCE DOCUMENTATION FORMAT ---\n"
                    f"The user has provided a reference documentation whose structure "
                    f"and style should be FOLLOWED. Your documentation plan MUST mirror "
                    f"this structure as closely as possible, adapting section titles and "
                    f"content to THIS project's actual code.\n\n"
                    f"Reference sections:\n{section_list}\n\n"
                    f"Style: {ref.tone}\n"
                    f"Section depth: {ref.section_depth}\n"
                    f"Conventions: {ref.formatting_conventions}\n\n"
                    f"IMPORTANT:\n"
                    f"- Map each reference section to what makes sense for THIS project.\n"
                    f"- If a reference section doesn't apply (e.g., 'Database' but this "
                    f"project has no DB), skip it.\n"
                    f"- If THIS project has features not covered by the reference "
                    f"structure, add sections for them.\n"
                    f"- Keep the same NUMBER of sections (roughly) as the reference.\n"
                    f"- Use similar section TITLES where applicable.\n"
                )

        # Step 1: Get raw analysis via RAG (no output_cls — let the LLM write freely)
        if on_progress:
            on_progress(1, total_steps, "Discovering project DNA via RAG...")
        query_engine = self.index.as_query_engine(
            response_mode="compact",
            similarity_top_k=20,
        )
        raw_result = query_engine.query(prompt)
        raw_text = str(raw_result)

        print("🧬 Structuring analysis into documentation plan...")

        # Step 2: Convert the raw text into structured JSON via a direct LLM call
        if on_progress:
            on_progress(2, total_steps, "Structuring analysis into documentation plan...")
        structuring_prompt = STRUCTURING_PROMPT.format(raw_analysis=raw_text)
        response = Settings.llm.complete(structuring_prompt)
        response_text = str(response)

        # Step 3: Parse JSON from the response
        if on_progress:
            on_progress(3, total_steps, "Parsing documentation plan...")
        try:
            json_str = _extract_json(response_text)
            data = json.loads(json_str)
            analysis = ProjectAnalysis.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # One retry with a stricter prompt
            print("⚠️  Retrying structured parsing...")
            retry_prompt = (
                "The previous response was not valid JSON. "
                "Return ONLY a raw JSON object (no markdown, no ```). "
                f"Schema: {ProjectAnalysis.model_json_schema()}\n\n"
                f"Analysis:\n{raw_text}"
            )
            retry_response = Settings.llm.complete(retry_prompt)
            json_str = _extract_json(str(retry_response))
            data = json.loads(json_str)
            analysis = ProjectAnalysis.model_validate(data)

        return analysis

    def display_plan(self, analysis: ProjectAnalysis):
        """Pretty-print the documentation plan."""
        print(f"\n{'='*60}")
        print(f"📋 DOCUMENTATION PLAN")
        print(f"{'='*60}")
        print(f"  Project    : {analysis.project_name}")
        print(f"  Type       : {analysis.project_type}")
        print(f"  Summary    : {analysis.one_line_summary}")
        print(f"  Language   : {analysis.primary_language}")
        print(f"  Stack      : {', '.join(analysis.frameworks_and_tools)}")
        print(f"\n  📑 Sections ({len(analysis.documentation_sections)}):")
        for i, section in enumerate(analysis.documentation_sections, 1):
            print(f"    {i}. {section.title} [{section.audience}]")
            print(f"       → {section.reason}")

        print(f"{'='*60}\n")
