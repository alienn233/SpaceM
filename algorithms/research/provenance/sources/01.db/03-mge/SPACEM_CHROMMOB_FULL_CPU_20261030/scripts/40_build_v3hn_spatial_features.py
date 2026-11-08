from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = (5_000, 10_000, 20_000, 50_000, 100_000)
LAMBDA = (5_000, 20_000, 50_000)


def summarize_accession(group: pd.DataFrame) -> list[dict]:
    cargo = group.loc[group.final_class.isin(("ARG", "MRG"))]
    mge = group.loc[group.final_class.eq("MGE")].sort_values("start1")
    if cargo.empty:
        return []
    starts = mge.start1.to_numpy(np.int64)
    ends = mge.end1.to_numpy(np.int64)
    mids = (starts + ends) / 2
    strands = mge.strand.to_numpy(str)
    scores = mge.p_MGE.to_numpy(np.float32)
    rows = []
    for item in cargo.itertuples(index=False):

        lo = 0
        hi = np.searchsorted(starts, int(item.end1) + 100_000 + 1, side="right")
        if hi > lo:
            idx = np.arange(lo, hi)
            gap = np.maximum.reduce((starts[idx] - int(item.end1) - 1,
                                     int(item.start1) - ends[idx] - 1,
                                     np.zeros(len(idx), dtype=np.int64)))
            keep = gap <= 100_000
            idx, gap = idx[keep], gap[keep]
        else:
            idx = np.empty(0, dtype=np.int64)
            gap = np.empty(0, dtype=np.int64)
        record = {
            "host_protein_id": item.host_protein_id,
            "accession": item.accession,
            "start1": int(item.start1),
            "end1": int(item.end1),
            "strand": item.strand,
            "cargo_class": item.final_class,
            "p_ARG": float(item.p_ARG),
            "p_MRG": float(item.p_MRG),
            "p_MGE_self": float(item.p_MGE),
            "gate_score": float(item.gate_score),
            "reference_position": item.reference_position,
            "inside_element_uid": item.inside_element_uid,
            "inside_element_type": item.inside_element_type,
            "partition": item.partition,
            "mge_count_100kb": len(idx),
            "mge_count_replicon": len(mge),
            "nearest_mge_edge_gap_bp": int(gap.min()) if len(gap) else np.nan,
            "nearest_mge_p_MGE": float(scores[idx[np.argmin(gap)]]) if len(gap) else np.nan,
        }
        for w in WINDOWS:
            sel = gap <= w
            record[f"mge_count_{w // 1000}kb"] = int(sel.sum())
            record[f"mge_score_sum_{w // 1000}kb"] = float(scores[idx[sel]].sum())
        for lam in LAMBDA:
            record[f"mge_exp_sum_lambda_{lam // 1000}kb"] = float(
                (scores[idx] * np.exp(-gap / lam)).sum())
        if len(idx):
            nearest = idx[np.argmin(gap)]
            cargo_mid = (int(item.start1) + int(item.end1)) / 2
            record["nearest_mge_signed_midpoint_bp"] = float(mids[nearest] - cargo_mid)
            record["nearest_mge_oriented_midpoint_bp"] = float(
                (mids[nearest] - cargo_mid) * (1 if item.strand == "+" else -1))
            record["nearest_mge_same_strand"] = bool(strands[nearest] == item.strand)
            record["mge_fraction_same_strand_100kb"] = float(
                np.mean(strands[idx] == item.strand))
            ordered = idx[np.argsort(starts[idx])]
            if len(ordered) > 1:
                adjacent_gap = np.maximum(starts[ordered[1:]] - ends[ordered[:-1]] - 1, 0)
                record["mge_pair_count_within_5kb"] = int(np.sum(adjacent_gap <= 5_000))
                record["mge_pair_min_gap_bp"] = int(adjacent_gap.min())
                record["mge_pair_span_bp"] = int(ends[ordered[-1]] - starts[ordered[0]] + 1)
            else:
                record["mge_pair_count_within_5kb"] = 0
                record["mge_pair_min_gap_bp"] = np.nan
                record["mge_pair_span_bp"] = 0
        else:
            record.update({"nearest_mge_signed_midpoint_bp": np.nan,
                           "nearest_mge_oriented_midpoint_bp": np.nan,
                           "nearest_mge_same_strand": np.nan,
                           "mge_fraction_same_strand_100kb": np.nan,
                           "mge_pair_count_within_5kb": 0,
                           "mge_pair_min_gap_bp": np.nan,
                           "mge_pair_span_bp": 0})
        record["baseline_predicted_mge_within_5kb"] = bool(len(gap) and gap.min() <= 5_000)
        rows.append(record)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    out = root / "v3hn_spatial"
    out.mkdir(exist_ok=True)
    done = out / "V3HN_SPATIAL_COMPLETE"
    if done.exists():
        print((out / "audit.json").read_text(), flush=True)
        return
    for marker in (root / "inference_v3hn/02_hierarchy/HIERARCHY_COMPLETE",
                   root / "full_cohort_split/CANDIDATE_SPLIT_COMPLETE"):
        if not marker.exists():
            raise RuntimeError(f"required upstream marker missing: {marker}")
    prediction = pd.read_csv(
        root / "inference_v3hn/02_hierarchy/HIERARCHY_PREDICTIONS.tsv", sep="\t",
        usecols=["protein_id", "gate_score", "final_class", "p_ARG", "p_MRG", "p_MGE"])
    positions = pd.read_parquet(root / "full_cohort_split/CANDIDATE_POSITIONS_WITH_SPLIT.parquet",
        columns=["host_protein_id", "accession", "start1", "end1", "strand",
                 "reference_position", "inside_element_uid", "inside_element_type", "partition"])
    if len(prediction) != 1_131_354 or len(positions) != len(prediction):
        raise RuntimeError("candidate locus count changed")
    if prediction.protein_id.duplicated().any() or positions.host_protein_id.duplicated().any():
        raise RuntimeError("duplicate candidate locus identifier")
    merged = positions.merge(prediction, left_on="host_protein_id", right_on="protein_id",
                             how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise RuntimeError("V3HN prediction does not cover all candidate loci")
    merged = merged.drop(columns=["protein_id", "_merge"])
    all_cargo = []
    for _, group in merged.groupby("accession", sort=False):
        all_cargo.extend(summarize_accession(group))
    result = pd.DataFrame.from_records(all_cargo)
    if len(result) != int(merged.final_class.isin(("ARG", "MRG")).sum()):
        raise RuntimeError("predicted cargo lost during spatial aggregation")
    if result.host_protein_id.duplicated().any():
        raise RuntimeError("spatial output has duplicate cargo")
    tmp = out / "CARGO_MGE_SPATIAL.parquet.part"
    result.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out / "CARGO_MGE_SPATIAL.parquet")
    audit = {
        "candidate_positions": len(merged),
        "predicted_ARG": int(merged.final_class.eq("ARG").sum()),
        "predicted_MRG": int(merged.final_class.eq("MRG").sum()),
        "predicted_MGE": int(merged.final_class.eq("MGE").sum()),
        "spatial_cargo_rows": len(result),
        "cargo_with_MGE_within_5kb": int(result.baseline_predicted_mge_within_5kb.sum()),
        "cargo_with_MGE_within_100kb": int(result.mge_count_100kb.gt(0).sum()),
        "fresh_test_cargo_rows_held_for_later": int(result.partition.eq("fresh_test").sum()),
        "inside_reference_cargo_rows": int(result.reference_position.eq("inside").sum()),
        "outside_reference_not_true_negative": True,
        "distance_definition": "CDS edge gap, zero if overlapping",
        "unit": "full host candidate protein locus; ARG/MRG and MGE labels from frozen V3HN",
        "mobility_model_trained": False,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    done.touch()
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
