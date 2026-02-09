# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-02-09

### Changed
- **Direct File Injection** — Each documentation section now injects the actual source code of files referenced in its `focus_areas` directly into the LLM prompt as "Ground Truth". This dramatically reduces hallucination, especially for large codebases (500–1000+ files), because the LLM sees real code instead of relying solely on RAG-retrieved chunks. The context budget is bounded per section (~40 KB), so performance scales regardless of repo size.
- **Anti-Hallucination Prompt Rules** — All writing prompts (section, rewrite, new section, coverage gap) now include explicit anti-hallucination instructions that forbid the LLM from fabricating file paths, class names, function names, environment variables, or config keys.
- **Post-Write Verification** — A lightweight regex-based verification pass runs after each section is generated; any file-path reference that does not correspond to a real source file is tagged `*(unverified)*` so reviewers can catch remaining inaccuracies.
- **Increased RAG Retrieval Depth** — `similarity_top_k` raised from 8 → 20 in the Planner and from 10 → 20 in the Writer, giving the LLM substantially more context per query.
- **Larger Chunk Size** — Indexer chunk size increased from default ~1024 → 2048 tokens with 256-token overlap, preserving more complete functions and class definitions per retrieval node.

## [1.0.0] - 2026-02-08

### Added
- AI-powered documentation generation from local repos and GitHub URLs
- Streamlit web UI for interactive documentation generation
- CLI interface with `generate`, `ui`, and `chat` commands
- Targeted documentation for specific directories or files
- Interactive Q&A chat over indexed codebases
- MongoDB integration for storing and retrieving documentation
- Docker and Docker Compose support for containerized deployment
- Multiple LLM model support via Cerebras
- Style-aware documentation with customizable tone and format
- Documentation evaluation and quality scoring

