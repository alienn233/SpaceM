from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_json, read_ids, read_json, sha256


PFAM_NUMERIC = [
    "w10_pfam_hit_fraction", "w10_pfam_domain_count",
    "w10_pfam_diversity", "w10_pfam_missing",
]
EGGNOG_NUMERIC = [
    "w10_eggnog_neighbor_count", "w10_eggnog_hit_fraction",
    "w10_cog_hit_fraction", "w10_go_hit_fraction", "w10_ec_hit_fraction",
    "w10_kegg_hit_fraction", "w10_cazy_hit_fraction",
]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t")


def align_table(path: Path, ids: list[str], required: list[str]) -> pd.DataFrame:
    frame = read_table(path)
    if "protein_id" not in frame:
        raise RuntimeError(f"protein_id missing from {path}")
    frame["protein_id"] = frame.protein_id.astype(str)
    if frame.protein_id.duplicated().any():
        raise RuntimeError(f"duplicate protein_id in {path}")
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise RuntimeError(f"missing columns in {path}: {missing_columns}")
    indexed = frame.set_index("protein_id", drop=False)
    missing_ids = [protein for protein in ids if protein not in indexed.index]
    if missing_ids:
        raise RuntimeError(f"{path} misses {len(missing_ids)} IDs; first={missing_ids[0]}")
    return indexed.loc[ids].reset_index(drop=True)


def load_array(path: Path, shape: tuple[int, int], allowed_dtypes: tuple[np.dtype, ...]) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    if value.shape != shape or value.dtype not in allowed_dtypes:
        raise RuntimeError(f"array contract failed for {path}: shape={value.shape} dtype={value.dtype}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"array contains non-finite values: {path}")
    return value


def fill_binary(matrix: np.ndarray, start: int, texts: pd.Series, vocab: list[str]) -> None:
    lookup = {token: index for index, token in enumerate(vocab)}
    for row, text in enumerate(texts.fillna("")):
        columns = {lookup[token] for token in str(text).split("|") if token in lookup}
        if columns:
            matrix[row, start + np.fromiter(columns, dtype=np.int64)] = 1.0


def make_matrix(
    esm8: np.ndarray,
    esm650: np.ndarray,
    portable: np.ndarray,
    pfam: pd.DataFrame,
    eggnog: pd.DataFrame,
    mag: pd.DataFrame,
    vocab: dict,
) -> np.ndarray:
    if vocab["pfam_numeric_columns"] != PFAM_NUMERIC or vocab["eggnog_numeric_columns"] != EGGNOG_NUMERIC:
        raise RuntimeError("numeric feature order differs from frozen contract")
    if len(vocab["pfam_w10_vocab"]) != 256 or len(vocab["eggnog_w10_vocab"]) != 512:
        raise RuntimeError("fold vocabulary sizes differ from 256/512")
    mag_columns = list(vocab["mag_columns"])
    missing = [column for column in mag_columns if column not in mag]
    if missing:
        raise RuntimeError(f"MAG table misses {len(missing)} frozen columns; first={missing[0]}")
    n = len(pfam)
    matrix = np.empty((n, 2564), dtype=np.float32)
    cursor = 0
    matrix[:, cursor:cursor + 320] = esm8
    cursor += 320
    matrix[:, cursor:cursor + 1280] = esm650
    cursor += 1280
    matrix[:, cursor:cursor + 44] = portable
    cursor += 44
    matrix[:, cursor:cursor + len(PFAM_NUMERIC)] = pfam[PFAM_NUMERIC].to_numpy(np.float32)
    cursor += len(PFAM_NUMERIC)
    matrix[:, cursor:cursor + 256] = 0.0
    fill_binary(matrix, cursor, pfam.w10_neighbor_pfam_tokens, vocab["pfam_w10_vocab"])
    cursor += 256
    matrix[:, cursor:cursor + len(EGGNOG_NUMERIC)] = eggnog[EGGNOG_NUMERIC].to_numpy(np.float32)
    cursor += len(EGGNOG_NUMERIC)
    matrix[:, cursor:cursor + 512] = 0.0
    fill_binary(matrix, cursor, eggnog.w10_neighbor_eggnog_tokens, vocab["eggnog_w10_vocab"])
    cursor += 512
    matrix[:, cursor:cursor + len(mag_columns)] = mag[mag_columns].to_numpy(np.float32)
    cursor += len(mag_columns)
    if cursor != 2564 or vocab["feature_dimension"] != 2564 or np.isinf(matrix).any():
        raise RuntimeError(f"assembled matrix contract failed: cursor={cursor}")

    if not np.isfinite(matrix[:, :2423]).all():
        raise RuntimeError("non-MAG features contain missing or infinite values")
    return matrix


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble separate V3 and V2 fold matrices from frozen feature components.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--esm8", type=Path, required=True)
    parser.add_argument("--esm650", type=Path, required=True)
    parser.add_argument("--portable-context", type=Path, required=True)
    parser.add_argument("--pfam", type=Path, required=True)
    parser.add_argument("--eggnog", type=Path, required=True)
    parser.add_argument("--mag", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    ids = read_ids(args.ids)
    n = len(ids)
    esm8 = load_array(args.esm8, (n, 320), (np.dtype("float16"), np.dtype("float32")))
    esm650 = load_array(args.esm650, (n, 1280), (np.dtype("float16"), np.dtype("float32")))
    portable = load_array(args.portable_context, (n, 44), (np.dtype("float32"),))
    pfam = align_table(args.pfam, ids, [*PFAM_NUMERIC, "w10_neighbor_pfam_tokens"])
    eggnog = align_table(args.eggnog, ids, [*EGGNOG_NUMERIC, "w10_neighbor_eggnog_tokens"])
    mag = align_table(args.mag, ids, [])
    for frame, columns, label in [(pfam, PFAM_NUMERIC, "Pfam"), (eggnog, EGGNOG_NUMERIC, "eggNOG")]:
        if not np.isfinite(frame[columns].to_numpy(np.float32)).all():
            raise RuntimeError(f"{label} numeric features contain missing or infinite values")

    hierarchy = config["hierarchy"]
    families = {
        "v3": Path(hierarchy["v3_clean_hn_gate_root"]),
        "v2": Path(hierarchy["v2_conditional_subtype_root"]),
    }
    records = []
    for family, root in families.items():
        for fold in hierarchy["folds"]:
            vocab_path = root / "fold_checkpoints" / f"fold{fold}" / "vocab.json"
            vocab = read_json(vocab_path)
            matrix = make_matrix(esm8, esm650, portable, pfam, eggnog, mag, vocab)
            destination = args.outdir / family / f"fold{fold}.f32.npy"
            atomic_npy(destination, matrix)
            records.append({
                "family": family,
                "fold": fold,
                "rows": n,
                "columns": int(matrix.shape[1]),
                "vocab_sha256": sha256(vocab_path),
                "matrix_file": str(destination),
                "mag_missing_values": int(np.isnan(matrix[:, 2423:]).sum()),
            })
            print(json.dumps({"event": "matrix_complete", **records[-1]}, sort_keys=True), flush=True)
    manifest = {
        "state": "FOLD_MATRICES_COMPLETE",
        "rows": n,
        "ids": str(args.ids),
        "separate_v3_v2_matrices": True,
        "records": records,
        "inputs": {key: str(value) for key, value in {
            "esm8": args.esm8, "esm650": args.esm650, "portable_context": args.portable_context,
            "pfam": args.pfam, "eggnog": args.eggnog, "mag": args.mag,
        }.items()},
    }
    atomic_json(args.outdir / "MATRIX_MANIFEST.json", manifest)
    (args.outdir / "FOLD_MATRICES_COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
