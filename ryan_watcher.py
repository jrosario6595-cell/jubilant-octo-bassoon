#!/usr/bin/env python3
"""
ryan_watcher.py — RYAN 6.2.1 Knowledge Watcher + Memory Sync

Watches the knowledge_drop zone for .txt files, injects each one via
mass_injector.py, then runs a memory sync inline (no separate master_sync.py
needed — that script is now absorbed here).

Flow:
  DROP_ZONE/*.txt  →  mass_injector.py  →  _sync_master_index()  →  flags log
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Paths  (keep in sync with ryan_ssj4_fortress.py and mass_injector.py)
# ---------------------------------------------------------------------------
DROP_ZONE    = Path("/mnt/core_agency/knowledge_drop")
INJECTOR     = "/mnt/core_agency/scripts/mass_injector.py"
DOMAINS_PATH = "/mnt/core_agency/memory/domains"
MASTER_PATH  = "/mnt/core_agency/memory/tier3_master_index"
FLAGS_PATH   = "/mnt/core_agency/flags/system_flags.jsonl"

_ALL_DOMAINS = ["cosmos", "life", "humanities", "ops", "bond"]


# ---------------------------------------------------------------------------
# Inline sync (replaces master_sync.py)
# ---------------------------------------------------------------------------
def _sync_master_index() -> None:
    """
    Count engrams in every domain collection and the master index,
    then append a milestone entry to system_flags.jsonl.
    """
    print("🛰️  Syncing master memory index...")
    settings = chromadb.Settings(anonymized_telemetry=False)

    domain_client = chromadb.PersistentClient(path=DOMAINS_PATH, settings=settings)
    master_client = chromadb.PersistentClient(path=MASTER_PATH,  settings=settings)

    master_col    = master_client.get_or_create_collection("master_index")
    master_count  = master_col.count()

    domain_counts = {
        name: domain_client.get_or_create_collection(f"domain_{name}").count()
        for name in _ALL_DOMAINS
    }

    print(f"   master_index: {master_count} engrams")
    for name, count in domain_counts.items():
        print(f"   domain_{name}: {count} engrams")
    print(f"   Total domain engrams: {sum(domain_counts.values())}")

    _write_flag({
        "action":        "flag",
        "type":          "milestone",
        "message":       "Master index synchronised.",
        "master_count":  master_count,
        "domain_counts": domain_counts,
        "timestamp":     datetime.now().isoformat(),
    })
    print("✅ Sync complete.")


def _write_flag(entry: dict) -> None:
    """Append a JSON entry to the system flags log."""
    try:
        with open(FLAGS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        print(f"⚠️  Could not write flag: {exc}")


# ---------------------------------------------------------------------------
# Watcher loop
# ---------------------------------------------------------------------------
def watch_and_ingest() -> None:
    """
    Poll DROP_ZONE every 5 seconds. For each .txt file found:
      1. Run mass_injector.py against it.
      2. Delete the file.
    After all files in a batch are injected, sync the master index.
    """
    DROP_ZONE.mkdir(parents=True, exist_ok=True)
    print(f"👀 RYAN Watcher active. Scanning: {DROP_ZONE}")

    while True:
        files = list(DROP_ZONE.glob("*.txt"))
        if files:
            for f in files:
                print(f"📥 Injecting: {f.name}")
                result = subprocess.run(["python3", INJECTOR, str(f)])
                if result.returncode != 0:
                    _write_flag({
                        "action":    "flag",
                        "type":      "error",
                        "message":   f"Injector failed on {f.name} (exit {result.returncode})",
                        "timestamp": datetime.now().isoformat(),
                    })
                f.unlink()

            try:
                _sync_master_index()
            except Exception as exc:
                print(f"❌ Sync error: {exc}")
                _write_flag({
                    "action":    "flag",
                    "type":      "error",
                    "message":   f"Sync failure: {exc}",
                    "timestamp": datetime.now().isoformat(),
                })

        time.sleep(5)


if __name__ == "__main__":
    watch_and_ingest()
