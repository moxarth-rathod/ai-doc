"""
Configuration settings for the documentation generator.
Loads values from .env file automatically.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "cerebras"
    model: str = os.environ.get("CEREBRAS_MODEL", "llama-3.3-70b")
    api_key: Optional[str] = None
    temperature: float = 0.3

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("CEREBRAS_API_KEY", "")


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    provider: str = "huggingface"
    model_name: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


@dataclass
class LoaderConfig:
    """Repository loading configuration."""
    # File extensions to include when scanning the project
    include_extensions: tuple = (
        # Source code
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
        ".rb", ".php", ".cs", ".cpp", ".c", ".h", ".hpp", ".swift",
        ".kt", ".kts", ".scala", ".groovy",
        # Shell & scripts
        ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
        # Config & data
        ".yaml", ".yml", ".toml", ".json", ".xml", ".html", ".css",
        ".scss", ".sass", ".less",
        ".properties", ".conf", ".cfg", ".ini",
        # Database & API
        ".sql", ".graphql", ".proto",
        # Infrastructure & CI
        ".tf", ".tfvars", ".dockerfile", ".hcl",
        # Templates
        ".jsp", ".jspx", ".ftl", ".vm", ".mustache", ".hbs",
        ".ejs", ".pug", ".jade", ".erb", ".twig",
        # Frontend frameworks
        ".vue", ".svelte",
        # Build files
        ".gradle", ".sbt", ".cmake",
        # Documentation
        ".md", ".txt", ".rst",
        # Environment
        ".env.example", ".env.sample", ".gitignore", ".editorconfig",
        # Other languages
        ".r", ".jl", ".lua", ".pl", ".pm", ".ex", ".exs",
        ".clj", ".cljs", ".edn", ".hs", ".elm", ".dart",
    )
    # Directories to always skip
    exclude_dirs: tuple = (
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "env", ".env", "dist", ".next", ".nuxt",
        "bin", "obj", ".idea", ".vscode",
        "vendor", "packages", ".tox", "eggs", "*.egg-info",
        ".mypy_cache", ".pytest_cache", "htmlcov", "coverage",
    )
    # Max file size to read (in bytes) - skip very large files
    max_file_size: int = 500_000  # 500KB
    # Temp directory for cloned repos
    clone_dir: str = "/tmp/prepare_doc_repos"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    output_dir: str = "output"
    output_filename: str = "DOCUMENTATION.md"
    verbose: bool = True
    mongo_uri: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.environ.get("MONGO_DB", "aidoc")
