"""
MongoDB-backed version store for documentation history.

Database structure:
    aidoc (database)
    ├── _projects                 ← registry of all documented projects
    │   ├── { project_id, display_name, repo_path, target_path, collection_name, ... }
    │   └── ...
    ├── v__myproject              ← versions for "myproject" (full repo)
    │   ├── { version: "1.0.0", markdown, full_markdown, ... }
    │   └── { version: "1.1.0", ... }
    └── v__myproject__src_auth    ← versions for "myproject" target "src/auth/"
        ├── { version: "1.0.0", ... }
        └── ...

Semantic versioning:
  - Major bump  (X.0.0) → full regeneration
  - Minor bump  (x.Y.0) → sections added or removed
  - Patch bump  (x.y.Z) → content edits within existing sections
"""

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure

try:
    import certifi
    _CA_FILE = certifi.where()
except ImportError:
    _CA_FILE = None

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

def _version_tuple(version: str) -> tuple[int, int, int]:
    """Parse '1.2.3' → (1, 2, 3)."""
    parts = version.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _version_str(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def _sanitize_collection_name(raw: str) -> str:
    """
    Turn a raw string into a valid MongoDB collection name.
    Rules: lowercase, alphanumeric + underscores, max 80 chars.
    """
    # Lowercase and replace path separators / non-alnum with underscores
    name = raw.lower().strip().rstrip("/")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    # Prefix with v__ to avoid collisions with internal collections
    name = f"v__{name}"
    return name[:80] if len(name) > 80 else name


def derive_project_id(repo_path: str, target_path: str | None = None) -> str:
    """
    Build a unique project identifier from repo path + optional target.
    Examples:
        /Users/me/myproject          → "myproject"
        /Users/me/myproject + core/  → "myproject__core"
        https://github.com/u/repo    → "repo"
        https://github.com/u/repo + src/auth → "repo__src_auth"
    """
    path = repo_path.rstrip("/")
    if "github.com" in path:
        parts = path.split("/")
        base = parts[-1].replace(".git", "") if parts else path
    else:
        base = os.path.basename(path) or "untitled"

    project_id = base
    if target_path and target_path.strip():
        target_clean = re.sub(r"[^a-zA-Z0-9]+", "_", target_path.strip().rstrip("/"))
        project_id = f"{base}__{target_clean}"

    return project_id


# ──────────────────────────────────────────────
# Version Store
# ──────────────────────────────────────────────

class DocVersionStore:
    """
    Manages versioned documentation snapshots in MongoDB.
    Each project (repo + target) gets its own collection.
    A _projects registry collection tracks all known projects.
    """

    REGISTRY = "_projects"

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", db_name: str = "aidoc"):
        self._client: Optional[MongoClient] = None
        self._uri = mongo_uri
        self.db_name = db_name
        self._connected = False
        self._last_error: Optional[str] = None

    # ── connection ────────────────────────────

    def connect(self) -> bool:
        """Try to connect; return True on success."""
        if self._connected:
            return True
        try:
            kwargs: dict = {"serverSelectionTimeoutMS": 5000}
            # Atlas (mongodb+srv://) needs a proper CA bundle for TLS
            if _CA_FILE and "srv" in self._uri.lower():
                kwargs["tlsCAFile"] = _CA_FILE
            self._client = MongoClient(self._uri, **kwargs)
            self._client.admin.command("ping")
            self._connected = True
            self._last_error = None
            return True
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc)
            return False

    @property
    def _db(self):
        return self._client[self.db_name]

    @property
    def _registry(self):
        return self._db[self.REGISTRY]

    def _col(self, project_id: str):
        """Return the version collection for a specific project."""
        col_name = _sanitize_collection_name(project_id)
        return self._db[col_name]

    # ── project registry ─────────────────────

    def register_project(
        self,
        project_id: str,
        display_name: str,
        repo_path: str,
        target_path: str | None = None,
    ) -> dict:
        """
        Register a project in the registry (upsert).
        Returns the project document.
        """
        col_name = _sanitize_collection_name(project_id)
        now = datetime.now(IST)

        self._registry.update_one(
            {"project_id": project_id},
            {
                "$set": {
                    "display_name": display_name,
                    "repo_path": repo_path,
                    "target_path": target_path,
                    "collection_name": col_name,
                    "last_updated": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )
        return self._registry.find_one({"project_id": project_id})

    def list_projects(self) -> list[dict]:
        """Return all registered projects with their version counts, newest first."""
        projects = list(self._registry.find().sort("last_updated", DESCENDING))
        # Enrich with version count
        for p in projects:
            col = self._db[p["collection_name"]]
            p["version_count"] = col.count_documents({})
            # Get latest version string
            latest = col.find_one(
                sort=[("major", DESCENDING), ("minor", DESCENDING), ("patch", DESCENDING)]
            )
            p["latest_version"] = latest["version"] if latest else None
        return projects

    # ── version bumping ──────────────────────

    @staticmethod
    def compute_change_type(
        old_titles: set[str],
        new_titles: set[str],
        old_sections: list[dict],
        new_sections: list[dict],
    ) -> str:
        """
        Determine the kind of change between two document versions.
        Returns one of: sections_added, sections_removed, content_modified.
        """
        added = new_titles - old_titles
        removed = old_titles - new_titles

        if added or removed:
            if added and removed:
                return "sections_added"  # structural change – treat as minor
            return "sections_added" if added else "sections_removed"

        # Check for content-level changes in shared titles
        for title in old_titles & new_titles:
            old_c = next((s["content"] for s in old_sections if s["title"] == title), "")
            new_c = next((s["content"] for s in new_sections if s["title"] == title), "")
            if old_c != new_c:
                return "content_modified"

        return "content_modified"  # fallback

    def bump_version(self, current_version: str, change_type: str) -> str:
        """Return the next version string based on the type of change."""
        major, minor, patch = _version_tuple(current_version)

        if change_type == "regenerated":
            return _version_str(major + 1, 0, 0)
        elif change_type in ("sections_added", "sections_removed"):
            return _version_str(major, minor + 1, 0)
        else:  # content_modified
            return _version_str(major, minor, patch + 1)

    # ── CRUD (per-project collection) ────────

    def save_version(
        self,
        project_id: str,
        version: str,
        markdown: str,
        full_markdown: str,
        description: str,
        change_type: str = "initial",
    ) -> str:
        """Insert a new version document into the project's collection."""
        major, minor, patch = _version_tuple(version)
        doc = {
            "version": version,
            "major": major,
            "minor": minor,
            "patch": patch,
            "markdown": markdown,
            "full_markdown": full_markdown,
            "description": description,
            "change_type": change_type,
            "created_at": datetime.now(IST),
        }
        self._col(project_id).insert_one(doc)

        # Update registry timestamp
        self._registry.update_one(
            {"project_id": project_id},
            {"$set": {"last_updated": datetime.now(IST)}},
        )
        return version

    def get_latest_version(self, project_id: str) -> Optional[dict]:
        """Return the most recent version document for a project."""
        return self._col(project_id).find_one(
            sort=[("major", DESCENDING), ("minor", DESCENDING), ("patch", DESCENDING)],
        )

    def get_version(self, project_id: str, version: str) -> Optional[dict]:
        """Return a specific version document."""
        return self._col(project_id).find_one({"version": version})

    def get_all_versions(self, project_id: str) -> list[dict]:
        """Return all versions for a project, newest first."""
        cursor = self._col(project_id).find(
            sort=[("major", DESCENDING), ("minor", DESCENDING), ("patch", DESCENDING)],
        )
        return list(cursor)

    def delete_project(self, project_id: str) -> int:
        """
        Delete a project entirely: drop its version collection
        and remove its registry entry. Returns version count deleted.
        """
        col = self._col(project_id)
        count = col.count_documents({})
        col.drop()
        self._registry.delete_one({"project_id": project_id})
        return count

    def delete_version(self, project_id: str, version: str) -> bool:
        """Delete a single version. Returns True if a document was removed."""
        result = self._col(project_id).delete_one({"version": version})
        return result.deleted_count > 0
