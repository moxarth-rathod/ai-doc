# 📄 AIDoc

**AI-powered project documentation generator** that reads your codebase and produces comprehensive, context-aware documentation — automatically.

AIDoc uses **Cerebras LLM** (via LlamaIndex) to scan a repository, understand its architecture, and generate tailored Markdown documentation. It doesn't rely on rigid templates — instead, it discovers what matters in *your* project and writes about that.

---

## ✨ Features

- **Smart Documentation Generation** — Analyzes your codebase and produces a structured Markdown document with sections that actually matter to your project.
- **Three Interfaces** — CLI, Streamlit Web UI, and an interactive Chat mode.
- **Git URL Support** — Point it at a GitHub, GitLab, or Bitbucket repo URL and it will clone and document it. Private repos are supported via `GITHUB_TOKEN`.
- **Targeted Documentation** — Generate focused docs for a specific directory or file (e.g., `src/auth/` or `core/payment.py`).
- **Reference Style Matching** — Provide an existing doc as a reference and AIDoc will match its tone, structure, and formatting.
- **Quality Evaluation** — Built-in evaluator scores the generated documentation for coverage, correctness, clarity, completeness, and usefulness.
- **Hierarchical Analysis** — Three-level codebase analysis (File → Module → System) that identifies the project's "north star", architecture style, heat zones, entry points, and data flow before writing.
- **Direct File Injection** — Each section's prompt receives the actual source code of relevant files as ground truth, dramatically reducing hallucination.
- **Anti-Hallucination Safeguards** — Explicit prompt rules forbid the LLM from fabricating file paths, class names, or config values. A post-write verification pass flags unverified references.
- **Iterative Refinement** — Refine generated documentation with natural-language feedback; the AI plans modifications, rewrites sections, and adds new ones on request.
- **Version History** — MongoDB integration stores every generated doc with semantic versioning, diffs, rollback, and per-project version browsing.
- **Docker Ready** — Full Docker Compose setup with MongoDB included.

---

## 🏗️ Architecture

```
ai-doc/
├── main.py              # CLI entry point (generate / ui / chat)
├── app.py               # Streamlit web UI
├── core/
│   ├── loader.py        # Repository loading (local + GitHub/GitLab/Bitbucket clone)
│   ├── indexer.py        # Codebase indexing with LlamaIndex vector store
│   ├── analyzer.py       # Hierarchical codebase analysis (File → Module → System)
│   ├── planner.py        # AI-driven documentation planning
│   ├── writer.py         # Documentation generation & refinement
│   ├── evaluator.py      # Coverage & quality evaluation
│   ├── style_analyzer.py # Reference doc style analysis
│   ├── chat.py           # Interactive Q&A engine with conversation memory
│   └── db.py             # MongoDB version store (semantic versioning + diffs)
├── config/
│   └── settings.py       # App configuration & env loading
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── CHANGELOG.md
```

### Generation Pipeline

```
Load Repo → Index Codebase → Hierarchical Analysis → (Style Analysis) → Plan Sections → Write Docs → Evaluate Quality → Save
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.13+**
- A **Cerebras API key** (get one at [cerebras.ai](https://cerebras.ai))
- **MongoDB** (optional — required only for version history in the Web UI)
- **Git** (required if documenting remote repositories)

---

### Option 1: Run Locally

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-doc.git
cd ai-doc
```

#### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure environment variables

Create a `.env` file in the project root:

```env
# Required
CEREBRAS_API_KEY=your-cerebras-api-key

# Optional (defaults shown)
CEREBRAS_MODEL=llama-3.3-70b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MONGO_URI=mongodb://localhost:27017
MONGO_DB=aidoc

# For private GitHub repos (optional)
GITHUB_TOKEN=your-github-token
```

#### 5. Run

**Generate documentation (CLI):**

```bash
# Document a local project
python main.py generate /path/to/your/project

# Document a GitHub repository
python main.py generate https://github.com/user/repo

# Custom output path
python main.py generate /path/to/project -o docs/MY_DOCS.md

# Specify a different LLM model
python main.py generate /path/to/project --model llama-3.1-8b

# Target a specific folder
python main.py generate /path/to/project --target src/auth/

# Use full repo context with a target
python main.py generate /path/to/project --target src/auth/ --full-repo

# Match a reference doc's style
python main.py generate /path/to/project --reference-doc EXAMPLE.md

# Skip the evaluation phase
python main.py generate /path/to/project --skip-eval

# Quiet mode (suppress progress output)
python main.py generate /path/to/project -q
```

**Launch the Web UI:**

```bash
python main.py ui                # opens at http://localhost:8501
python main.py ui --port 8502    # custom port
python main.py ui --no-browser   # headless mode
```

**Interactive Chat:**

```bash
python main.py chat /path/to/your/project
python main.py chat https://github.com/user/repo
python main.py chat /path/to/project --model llama-3.1-8b
```

---

### Option 2: Run with Docker

Docker Compose spins up both the Streamlit app and a MongoDB instance.

#### 1. Configure environment

Create a `.env` file in the project root (same variables as above). At minimum:

```env
CEREBRAS_API_KEY=your-cerebras-api-key
```

#### 2. Build and start

```bash
docker compose up --build
```

The Web UI will be available at **http://localhost:8501**.

#### 3. Stop

```bash
docker compose down
```

#### Docker environment variables

These can be set in `.env` or passed directly:

| Variable | Default | Description |
|---|---|---|
| `CEREBRAS_API_KEY` | — | **(Required)** Cerebras API key |
| `CEREBRAS_MODEL` | `llama-3.3-70b` | LLM model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `MONGO_USER` | `admin` | MongoDB root username |
| `MONGO_PASS` | `admin` | MongoDB root password |
| `MONGO_PORT` | `27017` | MongoDB exposed port |
| `MONGO_DB` | `aidoc` | MongoDB database name |
| `APP_PORT` | `8501` | Streamlit app exposed port |
| `LOCAL_REPOS_PATH` | `~` | Host directory mounted for local repo access |
| `GITHUB_TOKEN` | — | GitHub token for cloning private repos |

---

## 📖 Usage Examples

### CLI — Quick one-shot docs

```bash
python main.py generate ./my-project
```

Output is saved to `output/DOCUMENTATION.md` by default.

### Web UI — Full-featured interface

```bash
python main.py ui
```

The Streamlit dashboard provides seven tabs:

| Tab | Description |
|---|---|
| **🚀 Generate** | Enter a local path or GitHub URL, configure advanced options (target scope, full-repo context, reference doc upload), and run the pipeline with real-time progress |
| **📄 Documentation** | Preview rendered Markdown, view raw source, download the file, and approve/finalize the result |
| **🏗️ Architecture** | Explore the system's north star, architecture style, heat zones, module landscape, and developer's-first-hour reading list |
| **📊 Evaluation** | Review quality scores (coverage, correctness, clarity, completeness, usefulness) and per-directory/per-category coverage breakdowns |
| **🔧 Refine** | Iteratively improve the docs with natural-language feedback — the AI will modify, add, or remove sections as requested |
| **💬 Chat** | Ask questions about the indexed codebase and get AI-powered answers with conversation memory |
| **📚 Versions** | Browse, preview, diff, and restore past documentation versions across all your projects (requires MongoDB) |

### Chat — Ask questions about code

```bash
python main.py chat ./my-project
```

Type questions about the codebase and get AI-powered answers. Use `clear` to reset history and `quit` to exit.

---

## 🛠️ Configuration

All configuration is managed through environment variables (loaded from `.env`) and the `config/settings.py` file.

| Setting | Env Variable | Default |
|---|---|---|
| LLM Provider | — | `cerebras` |
| LLM Model | `CEREBRAS_MODEL` | `llama-3.3-70b` |
| API Key | `CEREBRAS_API_KEY` | — |
| Embedding Model | `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |
| MongoDB URI | `MONGO_URI` | `mongodb://localhost:27017` |
| MongoDB Database | `MONGO_DB` | `aidoc` |
| Output Directory | — | `output/` |
| Output Filename | — | `DOCUMENTATION.md` |

---

## 📝 License

This project is provided as-is for personal and educational use.
