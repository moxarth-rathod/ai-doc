"""
Style Analyzer - Extracts structure, tone, and formatting conventions from
a reference documentation file so the generator can mimic its style.
"""

import re
from typing import List, Optional

from pydantic import BaseModel, Field
from llama_index.core import Settings


# ──────────────────────────────────────────────
# Structured output
# ──────────────────────────────────────────────

class ReferenceSection(BaseModel):
    """A section extracted from the reference documentation."""
    title: str = Field(description="Section heading")
    depth: int = Field(description="Heading level (1=H1, 2=H2, etc.)")
    summary: str = Field(description="Brief summary of what this section covers")


class ReferenceStyle(BaseModel):
    """Complete style profile extracted from a reference documentation."""
    sections: List[ReferenceSection] = Field(
        description="Ordered list of sections found in the reference doc"
    )
    style_notes: str = Field(
        description="Observations about the writing style, tone, and formatting"
    )
    formatting_conventions: str = Field(
        description="Specific formatting patterns: bullet vs numbered lists, "
        "code block usage, table usage, heading style, etc."
    )
    tone: str = Field(
        description="The overall tone: e.g., formal-technical, casual-friendly, "
        "tutorial-style, reference-style, etc."
    )
    section_depth: str = Field(
        description="Typical depth of sections: brief/concise, moderate, or deep/detailed"
    )


# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────

STYLE_ANALYSIS_PROMPT = """\
You are a documentation expert analyzing a reference documentation file. Your goal is to \
extract the writing style, tone, and formatting conventions so that a new documentation can \
be generated in the same style.

Here is the reference documentation (may be truncated):

---
{doc_content}
---

Analyze this document and return a JSON object (no markdown fences, no explanation) with:
{{
  "style_notes": "Detailed observations about the writing style — e.g., uses concise \
sentences, avoids jargon, includes examples after every concept, uses analogies, etc.",
  "formatting_conventions": "Specific patterns — e.g., uses bullet points for lists, \
numbered steps for procedures, code blocks with language tags, tables for comparisons, \
callout boxes/blockquotes for warnings, etc.",
  "tone": "The overall tone: formal-technical, casual-friendly, tutorial-style, \
reference-style, conversational, academic, etc.",
  "section_depth": "How deep each section goes: brief (50-100 words), moderate (100-300 \
words), or detailed (300-600+ words)"
}}

Return ONLY the JSON object.
"""


# ──────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────

class DocStyleAnalyzer:
    """Extracts style and structure from a reference documentation file."""

    @staticmethod
    def analyze(doc_content: str) -> ReferenceStyle:
        """
        Analyze a reference documentation file and extract its style profile.

        Args:
            doc_content: The full text content of the reference documentation.

        Returns:
            A ReferenceStyle object with structure and style information.
        """
        # Step 1: Extract sections deterministically from markdown headings
        sections = DocStyleAnalyzer._extract_sections(doc_content)

        # Step 2: Use the LLM to analyze writing style and conventions
        # Truncate to avoid token limits while preserving enough for analysis
        truncated = doc_content[:8000]
        if len(doc_content) > 8000:
            # Also include a chunk from the middle for variety
            mid = len(doc_content) // 2
            truncated += "\n\n[... truncated ...]\n\n" + doc_content[mid:mid + 2000]

        prompt = STYLE_ANALYSIS_PROMPT.format(doc_content=truncated)
        response = Settings.llm.complete(prompt)
        response_text = str(response)

        # Parse the JSON response
        try:
            import json
            json_str = DocStyleAnalyzer._extract_json(response_text)
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback defaults if parsing fails
            data = {
                "style_notes": "Standard technical documentation style.",
                "formatting_conventions": "Uses markdown headings, bullet points, and code blocks.",
                "tone": "formal-technical",
                "section_depth": "moderate",
            }

        return ReferenceStyle(
            sections=sections,
            style_notes=data.get("style_notes", ""),
            formatting_conventions=data.get("formatting_conventions", ""),
            tone=data.get("tone", "formal-technical"),
            section_depth=data.get("section_depth", "moderate"),
        )

    @staticmethod
    def _extract_sections(doc_content: str) -> List[ReferenceSection]:
        """Extract section headings and their hierarchy from markdown content."""
        sections = []
        for match in re.finditer(r"^(#{1,6})\s+(.+?)$", doc_content, re.MULTILINE):
            depth = len(match.group(1))
            title = match.group(2).strip()
            # Skip very short or utility headings
            if len(title) < 2:
                continue

            # Generate a brief summary based on the content following this heading
            start = match.end()
            # Find the next heading or end of content
            next_heading = re.search(r"^#{1,6}\s+", doc_content[start:], re.MULTILINE)
            end = start + next_heading.start() if next_heading else len(doc_content)
            section_text = doc_content[start:end].strip()

            # Create a brief summary (first 100 chars of content)
            summary = section_text[:150].replace("\n", " ").strip()
            if len(section_text) > 150:
                summary += "..."

            sections.append(ReferenceSection(
                title=title,
                depth=depth,
                summary=summary,
            ))

        return sections

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

    @staticmethod
    def build_structure_summary(style: ReferenceStyle) -> str:
        """Build a human-readable summary of the reference doc structure."""
        lines = ["Reference documentation structure:"]
        for s in style.sections:
            indent = "  " * (s.depth - 1)
            lines.append(f"{indent}- {s.title}")
        lines.append(f"\nStyle: {style.tone}")
        lines.append(f"Depth: {style.section_depth}")
        lines.append(f"Conventions: {style.formatting_conventions}")
        return "\n".join(lines)

