from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"ID file is empty or non-unique: {path}")
    return ids


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                if name in result:
                    raise RuntimeError(f"duplicate FASTA ID: {name}")
                result[name] = "".join(chunks).upper()
            name = line[1:].split()[0]
            chunks = []
        else:
            if name is None:
                raise RuntimeError("FASTA sequence found before first header")
            chunks.append(line)
    if name is not None:
        if name in result:
            raise RuntimeError(f"duplicate FASTA ID: {name}")
        result[name] = "".join(chunks).upper()
    if not result or any(not sequence for sequence in result.values()):
        raise RuntimeError(f"FASTA is empty or contains an empty sequence: {path}")
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
