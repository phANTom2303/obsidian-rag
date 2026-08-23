# File Tracker Plan — Obsidian RAG

> **Goal**: On startup, detect which vault notes have been added, modified, or deleted since the last run, and sync only those changes into ChromaDB — no full re-ingestion.

---

## 1. Why a Custom Tracker (and Not LangChain's `RecordManager`)

LangChain's `RecordManager` is feasible in the narrow sense that it would work, but it is a poor fit here for a few concrete reasons:

| Concern | LangChain `RecordManager` | Custom tracker |
|---|---|---|
| **Dependency weight** | Pulls in the full `langchain` or `langchain_community` package | Zero new dependencies |
| **Storage backend** | Defaults to SQLite via SQLAlchemy — adds a second ORM layer on top of your existing ChromaDB | Plain SQLite via Python's built-in `sqlite3` |
| **ID scheme** | Content-hash of the document text, designed for LangChain `Document` objects | You control the ID scheme (more on this below) |
| **Cleanup strategy** | `cleanup="full"` deletes anything not in the current write batch — doesn't map to "delta-only startup scan" | Custom: compare filesystem state vs stored state, delete orphans precisely |
| **Change detection** | None — it re-hashes every document each run to see if content changed | You can use OS `mtime` as a fast pre-filter before hashing |

**Bottom line**: The *concept* of a record manager is exactly right. The implementation should be native to your stack.

---

## 2. The Two-Layer Change Detection Strategy

Detecting changes accurately requires two layers. Using only one of them has real tradeoffs.

### Layer 1 — `mtime` (modification timestamp)

The OS updates `mtime` whenever a file's content changes. Comparing `mtime` is a single integer comparison — essentially free.

```python
stored_mtime = tracker_db["note.md"]["mtime"]
current_mtime = os.path.getmtime("note.md")

if current_mtime == stored_mtime:
    skip()   # fast path — no I/O needed
```

**Why not `mtime` alone?**
- `mtime` can be spoofed (e.g., `touch -t`, rsync, Git checkouts). A restore from a backup could reset `mtime` without changing content, triggering a needless re-embed.
- `mtime` can fail to update in edge cases (NFS mounts, some copy tools).

### Layer 2 — Content hash (SHA-256)

If `mtime` has changed, read the file and hash it. Only if the hash also changed do we re-chunk and re-embed.

```python
if current_mtime != stored_mtime:
    current_hash = sha256(file_bytes).hexdigest()
    if current_hash == stored_hash:
        update_mtime_only()  # mtime lie — file unchanged, just update mtime record
    else:
        re_ingest()
```

**Why not hash alone?**
- Hashing requires reading the full file. For a 1000-note vault that is ~50 MB of I/O every startup, even if nothing changed.
- `mtime` gives you the fast O(1) path; hashing only runs on the small delta.

**Why SHA-256 and not MD5/xxHash?**
- SHA-256 is in Python's stdlib (`hashlib`) — no dependency.
- For notes the collision risk is academic, but SHA-256 costs nothing meaningful at this scale.

---

## 3. The Tracker State Store — SQLite

The tracker needs to persist state between runs. Options:

| Option | Pros | Cons |
|---|---|---|
| **SQLite** (built-in `sqlite3`) | Zero dependencies, ACID, fast, file-portable | One more file to manage |
| **JSON file** | Dead simple | No atomic writes → corruption on crash mid-write; no index → O(n) lookups |
| **Reuse ChromaDB metadata** | No new file | ChromaDB is not a key-value store; querying metadata by `file_path` requires a `where` filter scan, not an indexed lookup |
| **Redis** | Fast | Massively over-engineered for a local CLI tool |

**Decision: SQLite** — it is the right default for "structured local state that needs to survive crashes."

### Schema

```sql
CREATE TABLE IF NOT EXISTS file_tracker (
    file_path   TEXT PRIMARY KEY,  -- absolute path to the .md file
    mtime       REAL NOT NULL,     -- float from os.path.getmtime()
    sha256      TEXT NOT NULL,     -- hex digest
    chunk_ids   TEXT NOT NULL,     -- JSON array: ["note.md_chunk_1", "note.md_chunk_2", ...]
    last_synced REAL NOT NULL      -- unix timestamp of the last successful ingest
);
```

**Why store `chunk_ids`?**

This is the key insight. When a note is deleted or updated, you need to know *which* chunk IDs to delete from ChromaDB. You have two options:

- **Option A**: Query ChromaDB with a `where={"file_path": path}` filter to find chunks to delete.
- **Option B**: Store chunk IDs in the tracker DB and delete directly by ID.

**Option B is better** for two reasons:
1. ChromaDB metadata filters are linear scans, not indexed. On a large collection this is slow.
2. The chunk count can change on re-ingest (more or fewer chunks if the file changed significantly). You must delete the *old* chunk IDs, not the new ones.

**Why `file_path` as the primary key and not `file_name`?**

Your vault almost certainly has notes with the same name in different folders (e.g., `Math/Vectors.md` and `Physics/Vectors.md`). Absolute path is unique; filename is not.

---

## 4. Chunk ID Scheme — Make It Deterministic and Stable

Currently in `vectorize.py`, chunk IDs are:

```python
"id": f"{file_name}_chunk_{chunk_number}"
```

This has a subtle problem: two files named `Vectors.md` in different directories produce the same chunk ID. ChromaDB will silently overwrite one with the other on upsert.

**Proposed scheme:**

```python
import hashlib, os

def make_chunk_id(file_path: str, chunk_number: int) -> str:
    # Use a short hash of the absolute path for stable uniqueness
    path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:12]
    file_name = os.path.basename(file_path)
    return f"{file_name}_{path_hash}_chunk_{chunk_number}"
```

This is:
- **Human-readable** (still has the filename prefix)
- **Globally unique** (path hash disambiguates same-named files)
- **Stable** (same file → same prefix across runs, so upserts are idempotent)
- **Deterministic** (no UUIDs, no random component — the tracker can reconstruct IDs if needed)

---

## 5. The Full Sync Algorithm

On startup, `FileTracker.sync(vault_path)` runs the following:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Walk vault_path → build set: current_files (abs paths)   │
│ 2. Load tracker DB → build set: tracked_files               │
│                                                             │
│ 3. DELETED = tracked_files - current_files                  │
│    → delete chunk_ids from ChromaDB                         │
│    → remove rows from tracker DB                            │
│                                                             │
│ 4. NEW = current_files - tracked_files                      │
│    → chunk + embed + upsert to ChromaDB                     │
│    → insert row into tracker DB                             │
│                                                             │
│ 5. CANDIDATES = tracked_files ∩ current_files               │
│    for each file in CANDIDATES:                             │
│      if mtime unchanged → skip (fast path)                  │
│      else:                                                  │
│        hash file content                                    │
│        if hash unchanged → update mtime in DB only          │
│        else (MODIFIED):                                     │
│          delete old chunk_ids from ChromaDB                 │
│          chunk + embed + upsert new chunks                  │
│          update tracker DB (new hash, mtime, chunk_ids)     │
└─────────────────────────────────────────────────────────────┘
```

**Why process deletions first?** If you process new/modified files first and then crash before handling deletions, your next run will still clean up correctly. The opposite order (deletions last) could leave stale chunks forever if a crash happens after re-ingestion.

---

## 6. What Files to Walk

Use `pathlib.Path.rglob("*.md")` to find all markdown files recursively.

**Exclusions to consider:**
- `.obsidian/` directory (plugin configs, not notes)
- `_templates/` or similar template folders
- Files smaller than some minimum byte size (e.g., empty or stub files)

These should be configurable, not hardcoded.

---

## 7. Performance Characteristics

For a 1000-note vault (~50 MB of markdown):

| Operation | Cost | Notes |
|---|---|---|
| Walk filesystem | ~10ms | `os.walk` is fast |
| Load tracker DB | ~5ms | Single `SELECT *` |
| `mtime` comparison per unchanged file | ~0.001ms | Pure integer compare |
| SHA-256 hash of changed file | ~1ms per file | Buffered read |
| Embed N changed chunks | Dominates | ~50ms per chunk on CPU |

If only 5 notes changed out of 1000, you pay the embedding cost for those 5 notes only, not all 1000. That is the core win.

---

## 8. Concurrency / Safety

Since this is a startup sync and Obsidian itself is not expected to be writing notes at the same moment as the sync, concurrency is not an immediate concern. However:

- Wrap the ChromaDB upsert and tracker DB update in the same logical "commit" boundary. If ChromaDB upsert succeeds but the tracker DB write fails (crash), the next run will see the file as "new" and re-upsert (idempotent — safe). The reverse is worse: tracker marks it synced but chunks are not in ChromaDB. To avoid this, **write to ChromaDB first, then commit to tracker DB**.

---

## 9. File Layout

```
obsidian-rag/
├── file_tracker.py       ← New: FileTracker class (SQLite state + sync logic)
├── chunking.py           ← Existing (unchanged)
├── vectorize.py          ← Minor: update chunk ID scheme
├── db.py                 ← Minor: add delete_by_ids() method
├── main.py               ← Add startup sync call; expose manual mode entry point
└── tracker.db            ← Auto-created SQLite file (gitignored)
```

---

## 10. Settled Decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | **Vault path configuration** | Stored in `.env` as `VAULT_PATH`. Read at startup via `python-dotenv` (already in use). |
| 2 | **First-run / bootstrap** | No special case. A **manual mode** accepts any file path and recursively ingests/updates everything under it. On first use, run manual mode pointed at the full vault. On subsequent uses, it serves as a targeted re-ingest for any subtree. The tracker DB self-initialises on first write — no setup step needed. |
| 4 | **Progress reporting** | Simple stdout counter: `[X/Y] Ingesting: <filename>`. No extra dependencies. |
| 5 | **Scheduled / startup sync** | Deferred. The ingestion methods will be self-contained and callable from anywhere — startup hook, cron, or file-watcher can be wired up later without changes to the core logic. |
| 6 | Attachment files | Skip any images, pdfs, etc, only ingest markdown files for now |

### Manual Mode Interface

```
python main.py ingest <path>
```

- If `<path>` is a file, ingest that single file.
- If `<path>` is a directory, walk it recursively and ingest all `.md` files found.
- The tracker DB is updated after each file, so a mid-run crash is safe to resume.
- Progress output: `[3/142] Ingesting: Algorithms/DFS.md`
