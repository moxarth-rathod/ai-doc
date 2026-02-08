"""
Code indexer - indexes the codebase using LlamaIndex for RAG-based analysis.
Creates a vector store index from source files for querying.

Enhanced with metadata enrichment: each document carries importance,
category, and entry-point flags so the retriever can prioritise the
most architecturally significant code.
"""

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.cerebras import Cerebras

from config.settings import AppConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.analyzer import HierarchicalContext


# Map file extensions to languages for CodeSplitter
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".c": "c",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".html": "html",
}


class CodebaseIndexer:
    """Indexes a codebase into a queryable vector store."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._setup_llm()

    def _setup_llm(self):
        """Configure the LLM and embedding model globally."""
        Settings.llm = Cerebras(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding.model_name,
        )

    def index_codebase(
        self,
        files: list[dict],
        file_tree: str,
        on_progress: "Callable[[int, int, str], None] | None" = None,
        hierarchical_context: "HierarchicalContext | None" = None,
    ) -> VectorStoreIndex:
        """
        Create a VectorStoreIndex from collected files.

        If a *hierarchical_context* is provided (from the HierarchicalAnalyzer),
        each document is enriched with importance, entry-point, and category
        metadata so that downstream queries naturally favour the most
        architecturally significant code.

        Args:
            files: List of dicts with 'path' and 'content' keys.
            file_tree: String representation of the file tree.
            on_progress: Optional callback(current, total, file_path) for progress.
            hierarchical_context: Optional pre-analysed context from HierarchicalAnalyzer.

        Returns:
            A VectorStoreIndex ready for querying.
        """
        documents = []
        total = len(files)

        # Build a quick lookup from hierarchical analysis (if available)
        meta_lookup: dict[str, dict] = {}
        if hierarchical_context:
            for fm in hierarchical_context.file_metadata:
                meta_lookup[fm.path] = {
                    "importance": fm.importance,
                    "is_entry_point": fm.is_entry_point,
                    "classes": ", ".join(fm.classes[:10]) if fm.classes else "",
                    "functions": ", ".join(fm.functions[:10]) if fm.functions else "",
                }

        # Add the file tree as a special document for structural context
        tree_doc = Document(
            text=f"PROJECT FILE STRUCTURE:\n\n{file_tree}",
            metadata={"source": "file_tree", "type": "structure"},
        )
        documents.append(tree_doc)

        # If hierarchical context exists, add the system summary as a doc
        if hierarchical_context:
            ctx_block = hierarchical_context.build_prompt_block()
            if ctx_block.strip():
                ctx_doc = Document(
                    text=f"HIERARCHICAL CODEBASE ANALYSIS:\n\n{ctx_block}",
                    metadata={"source": "hierarchical_analysis", "type": "structure"},
                )
                documents.append(ctx_doc)

        # Process each source file
        for i, file_info in enumerate(files, 1):
            path = file_info["path"]
            content = file_info["content"]

            if on_progress:
                on_progress(i, total, path)

            if not content.strip():
                continue

            # Base metadata
            doc_meta: dict = {
                "source": path,
                "type": self._classify_file(path),
            }

            # Merge hierarchical metadata when available
            if path in meta_lookup:
                doc_meta.update(meta_lookup[path])

            doc = Document(
                text=f"FILE: {path}\n\n{content}",
                metadata=doc_meta,
            )
            documents.append(doc)

        if not documents:
            raise ValueError("No indexable documents found in the repository.")

        print(f"📊 Indexed {len(documents)} documents ({len(files)} source files + project structure)")

        # Build the vector index
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=self.config.verbose,
        )

        return index

    def _classify_file(self, path: str) -> str:
        """Classify a file into a category for metadata."""
        lower = path.lower()

        if any(name in lower for name in (
            "readme", "changelog", "contributing", "license", "authors",
        )):
            return "documentation"
        elif any(name in lower for name in (
            "requirements", "package.json", "cargo.toml", "go.mod",
            "gemfile", "pom.xml", "build.gradle", "setup.py",
            "pyproject.toml",
        )):
            return "dependencies"
        elif any(name in lower for name in (
            "dockerfile", "docker-compose", ".yml", ".yaml",
            "makefile", "procfile", ".env",
        )):
            return "configuration"
        elif any(name in lower for name in ("test", "spec", "__test__")):
            return "test"
        else:
            return "source"

