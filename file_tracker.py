"""
file_tracker.py — FileTracker

Implements the plan in file_tracker_plan.md:
  - SQLite-backed state store (file_path, mtime, sha256, chunk_ids, last_synced)
  - Two-layer change detection: mtime fast-path → SHA-256 confirmation
  - Full sync algorithm: deletions first, then new files, then modified files
  - Manual ingest mode: ingest a single file or recursively walk a directory
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Helper: deterministic, globally unique chunk ID
# ---------------------------------------------------------------------------

def make_chunk_id(file_path: str, chunk_number: int) -> str:
    """
    Produces a human-readable, globally unique, stable chunk ID.

    Format: <filename>_<12-char path-hash>_chunk_<n>

    - Human-readable: filename prefix
    - Globally unique: path hash disambiguates same-named files in different dirs
    - Stable/deterministic: same file always yields the same prefix
    """
    path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:12]
    file_name = os.path.basename(file_path)
    return f"{file_name}_{path_hash}_chunk_{chunk_number}"


# ---------------------------------------------------------------------------
# Helper: content hash
# ---------------------------------------------------------------------------

def _sha256_file(file_path: str) -> str:
    """Returns the SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# FileTracker
# ---------------------------------------------------------------------------

# Minimum file size (bytes) — skip empty / stub files
_MIN_FILE_BYTES = 10

# Directories to exclude when walking the vault
_EXCLUDED_DIRS = {".obsidian", "_templates", ".git"}


class FileTracker:
    """
    Persists per-file state in a local SQLite database and drives incremental
    ChromaDB synchronisation.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (auto-created on first use).
    chroma_manager : ChromaDBManager
        The ChromaDB manager used to upsert / delete vectors.
    vectorizer : Vectorizer
        The vectorizer used to embed chunks.
    excluded_dirs : set[str], optional
        Directory names to skip when walking the vault.
    min_file_bytes : int, optional
        Files smaller than this are skipped entirely.
    """

    def __init__(
        self,
        db_path: str,
        chroma_manager,
        vectorizer,
        excluded_dirs: Optional[set] = None,
        min_file_bytes: int = _MIN_FILE_BYTES,
    ):
        self.db_path = db_path
        self.chroma_manager = chroma_manager
        self.vectorizer = vectorizer
        self.excluded_dirs = excluded_dirs if excluded_dirs is not None else _EXCLUDED_DIRS
        self.min_file_bytes = min_file_bytes
        self._init_db()

    # ------------------------------------------------------------------
    # DB initialisation
    # ------------------------------------------------------------------

    def _init_db(self):
        """Creates the tracker table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_tracker (
                    file_path   TEXT PRIMARY KEY,
                    mtime       REAL NOT NULL,
                    sha256      TEXT NOT NULL,
                    chunk_ids   TEXT NOT NULL,
                    last_synced REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Low-level DB helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> dict:
        """Returns {file_path: row_dict} for every tracked file."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM file_tracker").fetchall()
        return {row["file_path"]: dict(row) for row in rows}

    def _insert_or_update(
        self,
        conn: sqlite3.Connection,
        file_path: str,
        mtime: float,
        sha256: str,
        chunk_ids: List[str],
    ):
        conn.execute(
            """
            INSERT INTO file_tracker (file_path, mtime, sha256, chunk_ids, last_synced)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                mtime       = excluded.mtime,
                sha256      = excluded.sha256,
                chunk_ids   = excluded.chunk_ids,
                last_synced = excluded.last_synced
            """,
            (file_path, mtime, sha256, json.dumps(chunk_ids), time.time()),
        )

    def _update_mtime(self, conn: sqlite3.Connection, file_path: str, mtime: float):
        """Updates mtime only (content didn't change — mtime lie)."""
        conn.execute(
            "UPDATE file_tracker SET mtime = ?, last_synced = ? WHERE file_path = ?",
            (mtime, time.time(), file_path),
        )

    def _delete_rows(self, conn: sqlite3.Connection, file_paths: List[str]):
        conn.executemany(
            "DELETE FROM file_tracker WHERE file_path = ?",
            [(p,) for p in file_paths],
        )

    # ------------------------------------------------------------------
    # Filesystem walk
    # ------------------------------------------------------------------

    def _walk_md_files(self, root: str) -> List[str]:
        """
        Recursively finds all .md files under *root*, excluding configured
        directories and files below the minimum size threshold.
        """
        found = []
        for path in Path(root).rglob("*.md"):
            # Exclude configured directories anywhere in the path
            if any(part in self.excluded_dirs for part in path.parts):
                continue
            abs_path = str(path.resolve())
            if os.path.getsize(abs_path) < self.min_file_bytes:
                continue
            found.append(abs_path)
        return found

    # ------------------------------------------------------------------
    # Ingest helpers
    # ------------------------------------------------------------------

    def _ingest_file(self, file_path: str) -> List[str]:
        """
        Chunks, embeds, and upserts a single file into ChromaDB.

        Returns the list of chunk IDs that were written.
        """
        from chunking import chunk_file

        raw_chunks = chunk_file(file_path)
        if not raw_chunks:
            return []

        # Assign deterministic chunk IDs (overrides the old scheme in vectorize.py)
        chunk_ids = [make_chunk_id(file_path, i + 1) for i in range(len(raw_chunks))]

        # Build the vectorized dataset manually so we control the IDs
        texts = [f"search_document: {c['page_content']}" for c in raw_chunks]
        embeddings = self.vectorizer.model.encode(texts)

        vectorized_data = []
        for i, chunk in enumerate(raw_chunks):
            vectorized_data.append(
                {
                    "id": chunk_ids[i],
                    "document": chunk["page_content"],
                    "embedding": embeddings[i].tolist(),
                    "metadata": chunk["metadata"],
                }
            )

        # Write to ChromaDB first, then commit tracker DB (per plan §8)
        self.chroma_manager.upsert_vectors(vectorized_data)
        return chunk_ids

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_path(self, path: str):
        """
        Manual mode (plan §10): ingest a single file or all .md files under a
        directory, applying two-layer change detection to skip unchanged files.

        Progress is printed to stdout as: [X/Y] Ingesting: <relative_path>
        """
        path = os.path.abspath(path)

        if os.path.isfile(path):
            if not path.endswith(".md"):
                print(f"Skipping non-markdown file: {path}")
                return
            files = [path]
        elif os.path.isdir(path):
            files = sorted(self._walk_md_files(path))
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")

        total = len(files)
        if total == 0:
            print("No markdown files found.")
            return

        tracked = self._load_all()

        new_count = modified_count = skipped_count = 0

        with self._connect() as conn:
            for idx, file_path in enumerate(files, start=1):
                rel = os.path.relpath(file_path, path) if os.path.isdir(path) else os.path.basename(file_path)
                current_mtime = os.path.getmtime(file_path)

                if file_path in tracked:
                    stored = tracked[file_path]

                    # Layer 1: mtime fast-path
                    if current_mtime == stored["mtime"]:
                        skipped_count += 1
                        continue

                    # Layer 2: hash confirmation
                    current_hash = _sha256_file(file_path)
                    if current_hash == stored["sha256"]:
                        # mtime lie — only update mtime record
                        self._update_mtime(conn, file_path, current_mtime)
                        skipped_count += 1
                        continue

                    # Genuinely modified — delete old chunks, re-ingest
                    print(f"[{idx}/{total}] Updating:  {rel}")
                    old_chunk_ids = json.loads(stored["chunk_ids"])
                    if old_chunk_ids:
                        self.chroma_manager.delete_by_ids(old_chunk_ids)

                    chunk_ids = self._ingest_file(file_path)
                    self._insert_or_update(conn, file_path, current_mtime, current_hash, chunk_ids)
                    modified_count += 1

                else:
                    # New file — ingest
                    print(f"[{idx}/{total}] Ingesting: {rel}")
                    chunk_ids = self._ingest_file(file_path)
                    current_hash = _sha256_file(file_path)
                    self._insert_or_update(conn, file_path, current_mtime, current_hash, chunk_ids)
                    new_count += 1

        print(
            f"\nDone. {new_count} new, {modified_count} updated, {skipped_count} unchanged"
            f" (out of {total} total files)."
        )

    def sync(self, vault_path: str):
        """
        Startup sync: walk vault_path and apply the full delta algorithm
        (deletions → new → modified), as defined in plan §5.
        """
        vault_path = os.path.abspath(vault_path)
        current_files = set(self._walk_md_files(vault_path))
        tracked = self._load_all()
        tracked_files = set(tracked.keys())

        deleted = tracked_files - current_files
        new_files = current_files - tracked_files
        candidates = tracked_files & current_files

        with self._connect() as conn:
            # 1. Deletions first (plan §5)
            if deleted:
                print(f"Removing {len(deleted)} deleted file(s) from index...")
                for file_path in deleted:
                    old_chunk_ids = json.loads(tracked[file_path]["chunk_ids"])
                    if old_chunk_ids:
                        self.chroma_manager.delete_by_ids(old_chunk_ids)
                self._delete_rows(conn, list(deleted))

            # 2. New files
            new_list = sorted(new_files)
            total_new = len(new_list)
            for idx, file_path in enumerate(new_list, start=1):
                rel = os.path.relpath(file_path, vault_path)
                print(f"[{idx}/{total_new}] Ingesting: {rel}")
                chunk_ids = self._ingest_file(file_path)
                mtime = os.path.getmtime(file_path)
                sha256 = _sha256_file(file_path)
                self._insert_or_update(conn, file_path, mtime, sha256, chunk_ids)

            # 3. Candidates — two-layer change detection
            modified_count = skipped_count = 0
            for file_path in sorted(candidates):
                stored = tracked[file_path]
                current_mtime = os.path.getmtime(file_path)

                if current_mtime == stored["mtime"]:
                    skipped_count += 1
                    continue

                current_hash = _sha256_file(file_path)
                if current_hash == stored["sha256"]:
                    self._update_mtime(conn, file_path, current_mtime)
                    skipped_count += 1
                    continue

                rel = os.path.relpath(file_path, vault_path)
                print(f"Updating: {rel}")
                old_chunk_ids = json.loads(stored["chunk_ids"])
                if old_chunk_ids:
                    self.chroma_manager.delete_by_ids(old_chunk_ids)
                chunk_ids = self._ingest_file(file_path)
                self._insert_or_update(conn, file_path, current_mtime, current_hash, chunk_ids)
                modified_count += 1

        print(
            f"\nSync complete. {len(deleted)} deleted, {total_new} new, "
            f"{modified_count} updated, {skipped_count} unchanged."
        )
