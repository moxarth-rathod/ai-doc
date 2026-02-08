"""
Repository loader - handles both local directories and GitHub URLs.
Clones remote repos and filters relevant source files.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from config.settings import LoaderConfig


class RepoLoader:
    """Loads and prepares a repository for analysis."""

    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()

    def load(self, repo_path: str) -> str:
        """
        Load a repository from a local path or GitHub URL.
        Returns the local directory path to the repo.
        """
        if self._is_github_url(repo_path):
            print(f"🌐 Detected GitHub URL: {repo_path}")
            return self._clone_repo(repo_path)
        elif os.path.isdir(repo_path):
            print(f"📂 Using local directory: {repo_path}")
            return os.path.abspath(repo_path)
        else:
            raise ValueError(
                f"Invalid repo path: '{repo_path}'. "
                "Provide a valid local directory or GitHub URL."
            )

    def _is_github_url(self, path: str) -> bool:
        """Check if the path is a GitHub/Git URL."""
        patterns = [
            r"^https?://github\.com/.+/.+",
            r"^git@github\.com:.+/.+",
            r"^https?://gitlab\.com/.+/.+",
            r"^https?://bitbucket\.org/.+/.+",
            r"^.*\.git$",
        ]
        return any(re.match(p, path) for p in patterns)

    def _inject_token(self, url: str) -> str:
        """
        If a GITHUB_TOKEN env var is set and the URL is HTTPS,
        inject it for authentication (supports private repos).
        e.g. https://github.com/u/r → https://<token>@github.com/u/r
        """
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            return url
        # Only inject into HTTPS URLs
        if url.startswith("https://"):
            # Remove any existing credentials in the URL
            url_no_proto = url[len("https://"):]
            # Strip existing user:pass@ if present
            if "@" in url_no_proto.split("/")[0]:
                url_no_proto = url_no_proto.split("@", 1)[1]
            return f"https://{token}@{url_no_proto}"
        return url

    def _clone_repo(self, url: str) -> str:
        """Clone a git repository to a temp directory."""
        # Clean up the URL for directory naming
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        clone_path = os.path.join(self.config.clone_dir, repo_name)

        # Remove existing clone if present
        if os.path.exists(clone_path):
            shutil.rmtree(clone_path)

        os.makedirs(self.config.clone_dir, exist_ok=True)

        # Inject token for private repo access (if configured)
        clone_url = self._inject_token(url)

        print(f"📥 Cloning repository to {clone_path}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, clone_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print("✅ Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            # Clean up partial clone on failure
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository: {e.stderr}") from e

        return clone_path

    def get_file_tree(self, repo_path: str) -> str:
        """
        Generate a file tree string of the repository for context.
        Only includes relevant files (respects exclude/include rules).
        """
        tree_lines = []
        repo_root = Path(repo_path)

        for root, dirs, files in os.walk(repo_path):
            # Filter out excluded directories (in-place to prevent os.walk descent)
            dirs[:] = [
                d for d in dirs
                if d not in self.config.exclude_dirs
                and not d.startswith(".")
            ]

            level = len(Path(root).relative_to(repo_root).parts)
            indent = "│   " * level
            folder_name = os.path.basename(root)

            if level == 0:
                tree_lines.append(f"{folder_name}/")
            else:
                tree_lines.append(f"{indent}├── {folder_name}/")

            sub_indent = "│   " * (level + 1)
            for f in sorted(files):
                file_path = os.path.join(root, f)
                ext = Path(f).suffix.lower()

                # Include relevant files or common config files
                if ext in self.config.include_extensions or f in (
                    "Makefile", "Dockerfile", "docker-compose.yml",
                    "Procfile", "Gemfile", "Cargo.toml", "go.mod",
                    "package.json", "pom.xml", "build.gradle",
                    "requirements.txt", "setup.py", "pyproject.toml",
                ):
                    size = os.path.getsize(file_path)
                    if size <= self.config.max_file_size:
                        tree_lines.append(f"{sub_indent}├── {f}")

        return "\n".join(tree_lines)

    def collect_files(self, repo_path: str) -> list[dict]:
        """
        Collect all relevant source files and their content.
        Returns list of dicts with 'path' and 'content' keys.
        """
        files = []
        repo_root = Path(repo_path)

        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs
                if d not in self.config.exclude_dirs
                and not d.startswith(".")
            ]

            for fname in sorted(filenames):
                file_path = os.path.join(root, fname)
                ext = Path(fname).suffix.lower()

                is_config_file = fname in (
                    "Makefile", "Dockerfile", "docker-compose.yml",
                    "Procfile", "Gemfile", "Cargo.toml", "go.mod",
                    "package.json", "pom.xml", "build.gradle",
                    "requirements.txt", "setup.py", "pyproject.toml",
                )

                if ext not in self.config.include_extensions and not is_config_file:
                    continue

                size = os.path.getsize(file_path)
                if size > self.config.max_file_size or size == 0:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                except (PermissionError, OSError):
                    continue

                relative_path = str(Path(file_path).relative_to(repo_root))
                files.append({
                    "path": relative_path,
                    "content": content,
                })

        return files

    def collect_files_for_target(self, repo_path: str, target: str) -> list[dict]:
        """
        Collect files scoped to a specific target path (file or directory).

        Uses progressive path stripping to handle cases where the user
        includes the repo directory name in the target (e.g., typing
        'RepoName/src/main' when internal paths are 'src/main/...').

        Args:
            repo_path: Root path of the repository.
            target: Relative path within the repo (e.g., 'src/auth/' or 'core/payment.py').
        """
        all_files = self.collect_files(repo_path)
        target = target.strip("/\\")

        # Strategy 1: Exact file match
        exact = [f for f in all_files if f["path"] == target]
        if exact:
            return exact

        # Strategy 2: Directory prefix match
        prefix = target + "/"
        filtered = [
            f for f in all_files
            if f["path"].startswith(prefix) or f["path"] == target
        ]
        if filtered:
            return filtered

        # Strategy 3: Progressive path stripping — handle "RepoName/src/main"
        # when internal paths are "src/main/..."
        parts = target.split("/")
        for i in range(1, len(parts)):
            sub_target = "/".join(parts[i:])
            if not sub_target:
                continue
            sub_prefix = sub_target + "/"
            sub_filtered = [
                f for f in all_files
                if f["path"].startswith(sub_prefix) or f["path"] == sub_target
            ]
            if sub_filtered:
                return sub_filtered

        # Strategy 4: Case-insensitive prefix match
        target_lower = target.lower()
        ci_filtered = [
            f for f in all_files
            if f["path"].lower().startswith(target_lower + "/")
        ]
        if ci_filtered:
            return ci_filtered

        # Strategy 5: Case-insensitive progressive stripping
        for i in range(1, len(parts)):
            sub_target = "/".join(parts[i:]).lower()
            if not sub_target:
                continue
            ci_sub = [
                f for f in all_files
                if f["path"].lower().startswith(sub_target + "/")
                or f["path"].lower() == sub_target
            ]
            if ci_sub:
                return ci_sub

        # Strategy 6: Fuzzy — target substring appears anywhere in path
        fuzzy = [f for f in all_files if target_lower in f["path"].lower()]
        return fuzzy

    def get_file_tree_for_target(self, repo_path: str, target: str) -> str:
        """
        Generate a file tree scoped to a specific target path.
        Falls back to the full tree if the target is not found.
        """
        target = target.strip("/\\")
        target_full = os.path.join(repo_path, target)

        if os.path.isdir(target_full):
            return self.get_file_tree(target_full)
        elif os.path.isfile(target_full):
            return target
        # Fallback to full tree
        return self.get_file_tree(repo_path)

    def cleanup(self, repo_path: str):
        """Remove cloned repository if it's in our temp directory."""
        if repo_path.startswith(self.config.clone_dir):
            shutil.rmtree(repo_path, ignore_errors=True)
            print("🧹 Cleaned up cloned repository.")

