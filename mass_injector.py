#!/usr/bin/env python3
"""
mass_injector.py — RYAN 6.2.1 Knowledge Injector

Reads a text file, splits it into overlapping 800-char chunks, routes each
chunk to the correct domain ChromaDB collection, and writes a 200-char stub
to the master_index for query routing.

Usage:
    python3 mass_injector.py <path_to_file.txt>

Called automatically by ryan_watcher.py — can also be run standalone.
"""
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Paths  (must match ryan_ssj4_fortress.py MemoryStore and ryan_watcher.py)
# ---------------------------------------------------------------------------
DOMAINS_PATH = "/mnt/core_agency/memory/domains"
MASTER_PATH  = "/mnt/core_agency/memory/tier3_master_index"

# ---------------------------------------------------------------------------
# Domain routing keywords  (mirror of MemoryStore._DOMAIN_KEYWORDS)
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: dict[str, frozenset] = {
    "cosmos":     frozenset({"math", "space", "earth", "physics", "planet", "science", "astronomy", "telescope", "light", "gravity"}),
    "life":       frozenset({"med", "bio", "doctor", "health", "anatomy", "sick", "virus", "heart", "mitochondria", "cell"}),
    "humanities": frozenset({"history", "religion", "god", "culture", "war", "library", "ancient"}),
    "ops":        frozenset({"script", "bash", "python", "ros", "system", "drive", "map", "config"}),
}

CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks to avoid split-boundary loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _route_domain(text: str) -> str:
    """Return the best-matching domain for *text*, defaulting to 'bond'."""
    lowered = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return domain
    return "bond"


def _overlapping_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into fixed-size character chunks with *overlap* between them.

    Overlapping ensures that ideas spanning a chunk boundary appear in both
    adjacent chunks, so retrieval never drops split-boundary context.
    """
    if not text:
        return []
    chunks, step = [], size - overlap
    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
        if i + size >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Main injection routine
# ---------------------------------------------------------------------------
def inject_knowledge(file_path: str) -> None:
    """
    Inject all chunks from *file_path* into ChromaDB.

    Each chunk is written to its domain collection (domain_*) and a 200-char
    stub is written to master_index for fast query routing.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print(f"🧠 Injecting {file_path} into RYAN memory...")
    content = Path(file_path).read_text(encoding="utf-8")

    settings = chromadb.Settings(anonymized_telemetry=False)
    domain_client = chromadb.PersistentClient(path=DOMAINS_PATH, settings=settings)
    master_client = chromadb.PersistentClient(path=MASTER_PATH,  settings=settings)

    domain_cols = {
        name: domain_client.get_or_create_collection(f"domain_{name}")
        for name in list(_DOMAIN_KEYWORDS.keys()) + ["bond"]
    }
    master_col = master_client.get_or_create_collection("master_index")

    chunks  = _overlapping_chunks(content)
    source  = os.path.basename(file_path)
    ts_base = datetime.now().isoformat()

    for i, chunk in enumerate(chunks):
        uid    = str(uuid.uuid4())
        domain = _route_domain(chunk)
        meta   = {"source": file_path, "time": ts_base, "domain": domain, "chunk": i}

        domain_cols[domain].add(
            documents=[chunk],
            ids=[f"{domain}_{uid}"],
            metadatas=[meta],
        )
        master_col.add(
            documents=[chunk[:200]],
            ids=[f"master_{uid}"],
            metadatas=[{"domain": domain, "source": source, "time": ts_base}],
        )

    top_domain = _route_domain(content)
    print(f"✅ {len(chunks)} engrams → domain '{top_domain}' + master_index.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "braininject.txt"
    inject_knowledge(target)
