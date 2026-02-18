"""Shared utilities for loading and paths."""

import json
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
ENCODING = "utf-8"


def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file; return list of parsed dicts (skips blank lines)."""
    rows = []
    with path.open("r", encoding=ENCODING) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
