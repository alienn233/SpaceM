from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import atomic_json, read_fasta, read_ids, read_json, sha256


CLASSES = np.asarray(["ARG", "MRG", "MGE"])
REQUIRED_COORDINATES = ["protein_id", "sample_id", "contig_id", "start", "end", "strand"]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, sep="\t")


def align_proteins(path: Path, ids: list[str]) -> pd.DataFrame:
    frame = read_table(path)
    missing = [column for column in REQUIRED_COORDINATES if column not in frame]
    if missing:
        raise RuntimeError(f"protein table misses required columns: {missing}")
    frame["protein_id"] = frame.protein_id.astype(str)
    if frame.protein_id.duplicated().any():
        raise RuntimeError("protein table contains duplicate protein_id")
    indexed = frame.set_index("protein_id", drop=False)
    absent = [protein for protein in ids if protein not in indexed.index]
    if absent:
        raise RuntimeError(f"protein table misses {len(absent)} IDs; first={absent[0]}")
    aligned = indexed.loc[ids].reset_index(drop=True)
    for column in ["start", "end"]:
        aligned[column] = pd.to_numeric(aligned[column], errors="raise").astype(np.int64)
    if (aligned.end < aligned.start).any() or aligned.contig_id.astype(str).eq("").any():
        raise RuntimeError("invalid protein coordinates")
    return aligned


def load_matrix(path: Path, n: int) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != (n, 2564) or matrix.dtype != np.float32:
        raise RuntimeError(f"matrix contract failed: {path} shape={matrix.shape} dtype={matrix.dtype}")
    if np.isinf(matrix).any() or not np.isfinite(matrix[:, :2423]).all():
        raise RuntimeError(f"matrix finite/missing-value contract failed: {path}")
    return matrix


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def write_fasta(path: Path, ids: list[str], sequences: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein in ids:
            sequence = sequences[protein]
            handle.write(f">{protein}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 CLEAN_HN membership gate plus V2 ARG/MRG/MGE conditional subtype.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--protein-table", type=Path, required=True)
    parser.add_argument("--protein-fasta", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    config = read_json(args.config)
    hierarchy = config["hierarchy"]
    ids = read_ids(args.ids)
    proteins = align_proteins(args.protein_table, ids)
    sequences = read_fasta(args.protein_fasta)
    missing_sequences = [protein for protein in ids if protein not in sequences]
    if missing_sequences:
        raise RuntimeError(f"protein FASTA misses {len(missing_sequences)} IDs; first={missing_sequences[0]}")
    v3_root = Path(hierarchy["v3_clean_hn_gate_root"])
    v2_root = Path(hierarchy["v2_conditional_subtype_root"])
    threshold = float(hierarchy["v3_gate_threshold"])
    v3_result_threshold = float(read_json(v3_root / "RESULT.json")["thresholds"]["target_gate"])
    if threshold != v3_result_threshold:
        raise RuntimeError("configured V3 gate threshold differs from frozen RESULT.json")

    gate_folds: list[np.ndarray] = []
    subtype_folds: list[np.ndarray] = []
    provenance = []
    for fold in hierarchy["folds"]:
        v3_matrix = load_matrix(args.matrix_dir / "v3" / f"fold{fold}.f32.npy", len(ids))
        v2_matrix = load_matrix(args.matrix_dir / "v2" / f"fold{fold}.f32.npy", len(ids))
        v3_checkpoint = v3_root / "fold_checkpoints" / f"fold{fold}"
        v2_checkpoint = v2_root / "fold_checkpoints" / f"fold{fold}"
        gate_path = v3_checkpoint / "gate.model.txt"
        subtype_path = v2_checkpoint / "target_subclass.model.txt"
        gate = lgb.Booster(model_file=str(gate_path))
        subtype = lgb.Booster(model_file=str(subtype_path))
        if gate.num_feature() != 2564 or gate.num_model_per_iteration() != 1:
            raise RuntimeError(f"V3 gate shape changed in fold {fold}")
        if subtype.num_feature() != 2564 or subtype.num_model_per_iteration() != 3:
            raise RuntimeError(f"V2 subtype shape changed in fold {fold}")
        p_gate = np.asarray(gate.predict(v3_matrix, num_threads=args.threads), dtype=np.float64)
        p_subtype = np.asarray(subtype.predict(v2_matrix, num_threads=args.threads), dtype=np.float64)
        if p_gate.shape != (len(ids),) or p_subtype.shape != (len(ids), 3):
            raise RuntimeError(f"prediction shape changed in fold {fold}")
        if not np.isfinite(p_gate).all() or not np.isfinite(p_subtype).all():
            raise RuntimeError(f"non-finite predictions in fold {fold}")
        if (p_gate < 0).any() or (p_gate > 1).any() or (p_subtype < 0).any() or not np.allclose(p_subtype.sum(1), 1.0, atol=1e-6):
            raise RuntimeError(f"probability contract failed in fold {fold}")
        gate_folds.append(p_gate)
        subtype_folds.append(p_subtype)
        provenance.append({
            "fold": fold,
            "v3_gate_sha256": sha256(gate_path),
            "v3_vocab_sha256": sha256(v3_checkpoint / "vocab.json"),
            "v2_subtype_sha256": sha256(subtype_path),
            "v2_vocab_sha256": sha256(v2_checkpoint / "vocab.json"),
        })
        print(json.dumps({"event": "hierarchy_fold_scored", "fold": fold, "rows": len(ids)}), flush=True)

    gate_stack = np.stack(gate_folds)
    subtype_stack = np.stack(subtype_folds)
    p_gate = gate_stack.mean(0)
    p_subtype = subtype_stack.mean(0)
    subtype_call = CLASSES[p_subtype.argmax(1)]
    binary = p_gate >= threshold
    final_class = np.where(binary, subtype_call, "OTHER")
    result = proteins.copy()
    result["gate_model"] = "V3_CLEAN_HN"
    result["gate_score"] = p_gate
    result["gate_score_sd"] = gate_stack.std(0)
    result["gate_threshold"] = threshold
    result["binary_prediction"] = np.where(binary, "YES", "NO")
    result["conditional_model"] = "V2_FORMAL_FOLDSAFE"
    result["conditional_subclass"] = subtype_call
    result["final_class"] = final_class
    result["p_OTHER"] = 1.0 - p_gate
    for index, klass in enumerate(CLASSES):
        result[f"conditional_p_{klass}"] = p_subtype[:, index]
        result[f"conditional_p_{klass}_sd"] = subtype_stack[:, :, index].std(0)
        result[f"p_{klass}"] = p_gate * p_subtype[:, index]
    if not np.allclose(result[["p_OTHER", "p_ARG", "p_MRG", "p_MGE"]].sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("joint probabilities do not sum to one")

    args.outdir.mkdir(parents=True, exist_ok=True)
    atomic_tsv(args.outdir / "HIERARCHY_PREDICTIONS.tsv", result)
    class_counts = {}
    for klass in CLASSES:
        selected = result.loc[result.final_class.eq(klass), "protein_id"].astype(str).tolist()
        write_fasta(args.outdir / "fasta" / f"{klass}.faa", selected, sequences)
        class_counts[str(klass)] = len(selected)
    manifest = {
        "state": "HIERARCHY_COMPLETE",
        "rows": len(ids),
        "yes": int(binary.sum()),
        "no": int((~binary).sum()),
        "class_counts": class_counts,
        "gate": "V3 CLEAN_HN five-fold mean",
        "gate_threshold": threshold,
        "conditional_subtype": "V2 formal fold-safe five-fold mean",
        "v3_subtype_head_used": False,
        "v2_gate_head_used": False,
        "virulence_factor_branch_enabled": False,
        "selection_override_by_user": True,
        "selection_boundary": config["policy"]["v3_override_note"],
        "model_provenance": provenance,
    }
    atomic_json(args.outdir / "HIERARCHY_MANIFEST.json", manifest)
    (args.outdir / "HIERARCHY_COMPLETE").touch()
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
