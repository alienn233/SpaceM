from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
OLD_THRESHOLD = 0.7277917861938477
CENTER_NUMERIC = [
    "old_mlp_oof_score", "protein_length", "cds_length", "center_gc_percent",
    "contig_length", "contig_gc_percent", "center_contig_gc_delta",
    "distance_to_left_end", "distance_to_right_end", "nearest_contig_end_distance",
    "genes_within_5kb", "genes_within_10kb",
]
NB_NUMERIC = [
    "protein_length", "cds_length", "bp_distance_to_center", "gap_or_overlap_to_adjacent",
    "same_strand_as_center", "gc_percent", "gc_delta_from_center", "is_pad",
]
LGB_PARAMS = dict(
    objective="binary", device_type="cpu", learning_rate=0.03, num_leaves=31,
    n_estimators=1000, min_child_samples=100, max_bin=127, random_state=SEED,
    n_jobs=48, verbosity=-1, deterministic=True, force_col_wise=True,
)


def rel_name(value: int) -> str:
    return f"m{abs(value):02d}" if value < 0 else f"p{value:02d}"


def tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    return [x for x in text.split("|") if x]


def build_wide(out: Path, smoke: bool) -> pd.DataFrame:
    stem = "SMOKE_" if smoke else ""
    center_path = out / f"{stem}CENTER_BASE_FEATURES.parquet"
    neighbor_path = out / f"{stem}NEIGHBOR_INDEX.parquet"
    centers = pq.read_table(center_path).to_pandas()
    neighbors = pq.read_table(neighbor_path).to_pandas()
    if centers["protein_id"].duplicated().any():
        raise ValueError("duplicate center protein IDs")
    if len(neighbors) != 20 * len(centers):
        raise ValueError(f"neighbor contract expected {20 * len(centers)}, found {len(neighbors)}")
    neighbors.sort_values(["protein_id", "relative_position"], inplace=True)
    wide = centers.set_index("protein_id", drop=False)
    for col in NB_NUMERIC:
        pivot = neighbors.pivot(index="protein_id", columns="relative_position", values=col)
        pivot.columns = [f"nb_{rel_name(int(x))}_{col}" for x in pivot.columns]
        wide = wide.join(pivot, how="left", validate="one_to_one")
    for window in (2, 5, 10):
        part = neighbors[neighbors["relative_position"].abs() <= window].copy()
        valid = part[~part["is_pad"].astype(bool)].copy()
        grouped = valid.groupby("protein_id", sort=False)
        agg = grouped.agg(
            **{
                f"w{window}_neighbor_count": ("neighbor_id", "size"),
                f"w{window}_mean_protein_length": ("protein_length", "mean"),
                f"w{window}_std_protein_length": ("protein_length", "std"),
                f"w{window}_mean_abs_gc_delta": ("gc_delta_from_center", lambda x: np.nanmean(np.abs(x))),
                f"w{window}_same_strand_fraction": ("same_strand_as_center", "mean"),
                f"w{window}_overlap_count": ("gap_or_overlap_to_adjacent", lambda x: int((x < 0).sum())),
                f"w{window}_mean_gap": ("gap_or_overlap_to_adjacent", "mean"),
                f"w{window}_std_gap": ("gap_or_overlap_to_adjacent", "std"),
                f"w{window}_median_gap": ("gap_or_overlap_to_adjacent", "median"),
                f"w{window}_mean_bp_distance": ("bp_distance_to_center", "mean"),
                f"w{window}_min_bp_distance": ("bp_distance_to_center", "min"),
                f"w{window}_max_bp_distance": ("bp_distance_to_center", "max"),
            }
        )
        pfam_rows = []
        for pid, group in part.groupby("protein_id", sort=False):
            ordered = []
            hit_genes = 0
            domain_count = 0
            domain_set = set()
            valid_genes = 0
            for row in group.itertuples(index=False):
                if bool(row.is_pad):
                    continue
                valid_genes += 1
                values = [x for x in str(row.pfams).split(",") if x and x != "nan"]
                if values:
                    hit_genes += 1
                domain_count += len(values)
                domain_set.update(values)
                ordered.extend(values)
            pfam_rows.append({
                "protein_id": pid,
                f"w{window}_neighbor_pfams": "|".join(ordered),
                f"w{window}_pfam_hit_fraction": hit_genes / valid_genes if valid_genes else 0.0,
                f"w{window}_pfam_domain_count": domain_count,
                f"w{window}_pfam_diversity": len(domain_set),
                f"w{window}_pfam_missing": float(domain_count == 0),
            })
        pfam = pd.DataFrame(pfam_rows).set_index("protein_id")
        wide = wide.join(agg, how="left").join(pfam, how="left")
    wide.reset_index(drop=True, inplace=True)
    path = out / f"{stem}CONTEXT_FEATURES.parquet"
    wide.to_parquet(path, index=False, compression="zstd")
    return wide


def assign_splits(frame: pd.DataFrame, out: Path, smoke: bool) -> pd.DataFrame:
    stem = "SMOKE_" if smoke else ""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = np.full(len(frame), -1, dtype=np.int8)
    y = frame["label"].to_numpy(dtype=np.int8)
    groups = frame["mag_id"].astype(str).to_numpy()
    for idx, (_, valid) in enumerate(splitter.split(np.zeros(len(frame)), y, groups)):
        fold[valid] = idx
    if (fold < 0).any():
        raise ValueError("incomplete MAG split assignment")
    frame["new_mag_fold"] = fold
    if frame.groupby("mag_id")["new_mag_fold"].nunique().max() != 1:
        raise ValueError("MAG leakage across new folds")
    manifest = frame[["protein_id", "label", "environment", "mag_id", "contig_id", "homology_cluster", "old_homology_fold", "new_mag_fold"]].copy()
    cluster_cross = manifest.groupby("homology_cluster")["new_mag_fold"].nunique()
    manifest["old_homology_cluster_crosses_new_mag_folds"] = manifest["homology_cluster"].map(cluster_cross).gt(1)
    manifest.to_csv(out / f"{stem}SPLIT_MANIFEST.tsv", sep="\t", index=False)
    audit = {
        "rows": len(frame), "mags": int(frame.mag_id.nunique()),
        "old_fold0_rows_used": int((frame.old_homology_fold == 0).sum()),
        "mag_groups_crossing_new_folds": int((frame.groupby("mag_id")["new_mag_fold"].nunique() > 1).sum()),
        "old_homology_clusters": int(manifest.homology_cluster.nunique()),
        "old_homology_clusters_crossing_new_mag_folds": int((cluster_cross > 1).sum()),
        "rows_in_crossing_old_homology_clusters": int(manifest.old_homology_cluster_crosses_new_mag_folds.sum()),
        "note": "MAG grouping is enforced. Old MMseqs30 clusters are audited, not used as a second grouping key.",
    }
    if audit["old_fold0_rows_used"] != 0 or audit["mag_groups_crossing_new_folds"] != 0:
        raise ValueError(f"split leakage contract failed: {audit}")
    (out / f"{stem}SPLIT_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return frame


def numeric_columns(window: int, model: str) -> list[str]:
    cols = list(CENTER_NUMERIC)
    if model == "M1":
        return cols
    for pos in list(range(-window, 0)) + list(range(1, window + 1)):
        tag = rel_name(pos)
        cols.extend(f"nb_{tag}_{name}" for name in NB_NUMERIC)
    cols.extend([
        f"w{window}_neighbor_count", f"w{window}_mean_protein_length", f"w{window}_std_protein_length",
        f"w{window}_mean_abs_gc_delta", f"w{window}_same_strand_fraction", f"w{window}_overlap_count",
        f"w{window}_mean_gap", f"w{window}_std_gap", f"w{window}_median_gap",
        f"w{window}_mean_bp_distance", f"w{window}_min_bp_distance", f"w{window}_max_bp_distance",
        f"w{window}_pfam_hit_fraction", f"w{window}_pfam_domain_count",
        f"w{window}_pfam_diversity", f"w{window}_pfam_missing",
    ])
    return cols


def balanced_train_indices(frame: pd.DataFrame, train_idx: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    part = frame.iloc[train_idx]
    pos = train_idx[part["label"].to_numpy() == 1]
    neg = train_idx[part["label"].to_numpy() == 0]
    target = min(len(neg), 3 * len(pos))
    groups = frame.iloc[neg]["negative_selection_group"].fillna("").to_numpy()
    hard = neg[groups == "large_cluster"]
    regular = neg[groups != "large_cluster"]
    hard_n = min(len(hard), target // 2)
    regular_n = min(len(regular), target - hard_n)
    chosen = []
    if hard_n:
        chosen.append(rng.choice(hard, hard_n, replace=False))
    if regular_n:
        chosen.append(rng.choice(regular, regular_n, replace=False))
    used = hard_n + regular_n
    if used < target:
        chosen_flat = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        remaining = np.setdiff1d(neg, chosen_flat, assume_unique=False)
        chosen.append(rng.choice(remaining, target - used, replace=False))
    sampled = np.concatenate([pos] + chosen)
    rng.shuffle(sampled)
    return sampled


def top_pfam_vocab(texts: pd.Series, limit: int = 256) -> list[str]:
    counts = Counter()
    for text in texts.fillna(""):
        counts.update(set(tokens(text)))
    return [x for x, _ in counts.most_common(limit)]


def pfam_sparse(texts: pd.Series, vocab: list[str]) -> sparse.csr_matrix:
    lookup = {x: i for i, x in enumerate(vocab)}
    rows, cols = [], []
    for i, text in enumerate(texts.fillna("")):
        for token in set(tokens(text)):
            j = lookup.get(token)
            if j is not None:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)


def make_matrix(frame: pd.DataFrame, indices: np.ndarray, cols: list[str], model: str, window: int, vocab: list[str] | None, shuffle_context: bool, seed: int) -> sparse.csr_matrix | np.ndarray:
    numeric = frame.iloc[indices][cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0).to_numpy(dtype=np.float32)
    texts = None
    if model in {"M3", "M3_RANDOM"} or shuffle_context:
        texts = frame.iloc[indices][f"w{window}_neighbor_pfams"].fillna("").reset_index(drop=True)
    if shuffle_context:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(indices))
        center_n = len(CENTER_NUMERIC)
        numeric[:, center_n:] = numeric[order, center_n:]
        if texts is not None:
            texts = texts.iloc[order].reset_index(drop=True)
    if model not in {"M3", "M3_RANDOM"}:
        return numeric
    if vocab is None:
        raise ValueError("M3 requires Pfam vocabulary")
    return sparse.hstack([sparse.csr_matrix(numeric), pfam_sparse(texts, vocab)], format="csr")


def train_oof(frame: pd.DataFrame, model: str, window: int, random_control: bool = False) -> tuple[np.ndarray, list[int], list[str]]:
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    best_iterations = []
    last_vocab: list[str] = []
    cols = numeric_columns(window, model)
    for fold in range(5):
        valid = np.flatnonzero(frame["new_mag_fold"].to_numpy() == fold)
        train_all = np.flatnonzero(frame["new_mag_fold"].to_numpy() != fold)
        train = balanced_train_indices(frame, train_all, SEED + 1000 * fold + window)
        vocab = None
        if model in {"M3", "M3_RANDOM"}:
            vocab = top_pfam_vocab(frame.iloc[train][f"w{window}_neighbor_pfams"], 256)
            last_vocab = vocab
        x_train = make_matrix(frame, train, cols, model, window, vocab, random_control, SEED + fold * 17 + 1)
        x_valid = make_matrix(frame, valid, cols, model, window, vocab, random_control, SEED + fold * 17 + 2)
        classifier = lgb.LGBMClassifier(**LGB_PARAMS)
        classifier.fit(
            x_train, frame.iloc[train]["label"].to_numpy(),
            eval_set=[(x_valid, frame.iloc[valid]["label"].to_numpy())],
            eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        scores[valid] = classifier.predict_proba(x_valid, num_iteration=classifier.best_iteration_)[:, 1]
        best_iterations.append(int(classifier.best_iteration_ or LGB_PARAMS["n_estimators"]))
    if not np.isfinite(scores).all():
        raise ValueError(f"nonfinite OOF scores for {model} W{window}")
    return scores, best_iterations, last_vocab


def threshold_max_mcc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)[::-1]
    y_sorted = y[order]
    score_sorted = score[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_p = int((y == 1).sum())
    total_n = int((y == 0).sum())
    fn = total_p - tp
    tn = total_n - fp


    tp_f = tp.astype(np.float64)
    fp_f = fp.astype(np.float64)
    fn_f = fn.astype(np.float64)
    tn_f = tn.astype(np.float64)
    denom = np.sqrt(
        (tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f)
    )
    mcc = np.divide(
        tp_f * tn_f - fp_f * fn_f,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    candidates = np.flatnonzero(np.r_[score_sorted[:-1] != score_sorted[1:], True])
    best = candidates[np.argmax(mcc[candidates])]
    return float(score_sorted[best])


def threshold_for_recall(y: np.ndarray, score: np.ndarray, target: float) -> float:
    positive = np.sort(score[y == 1])[::-1]
    need = max(1, min(len(positive), int(math.ceil(target * len(positive)))))
    return float(positive[need - 1])


def threshold_for_fpr(y: np.ndarray, score: np.ndarray, target: float) -> float:
    negative = np.sort(score[y == 0])[::-1]
    allowed = int(math.floor(target * len(negative)))
    if allowed <= 0:
        return float(np.nextafter(negative[0], np.inf))
    return float(negative[min(allowed - 1, len(negative) - 1)])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
        "positive_prevalence": float(np.mean(y)), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auroc": float(roc_auc_score(y, score)), "auprc": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "fpr": float(fp / (tn + fp)) if tn + fp else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def comparison_metrics(y: np.ndarray, score: np.ndarray, threshold: float, baseline_recall: float) -> dict:
    out = metrics(y, score, threshold)
    matched_t = threshold_for_recall(y, score, baseline_recall)
    matched = metrics(y, score, matched_t)
    out.update({
        "matched_baseline_recall_threshold": matched_t,
        "fpr_at_baseline_recall": matched["fpr"],
        "precision_at_baseline_recall": matched["precision"],
        "actual_recall_at_baseline_recall": matched["recall"],
    })
    for fpr in (0.01, 0.02, 0.04):
        t = threshold_for_fpr(y, score, fpr)
        m = metrics(y, score, t)
        out[f"recall_at_fpr_{int(fpr*100)}pct"] = m["recall"]
        out[f"actual_fpr_at_{int(fpr*100)}pct"] = m["fpr"]
    precision, recall, _ = precision_recall_curve(y, score)
    for target in (0.80, 0.90):
        eligible = recall[precision >= target]
        out[f"recall_at_precision_{int(target*100)}pct"] = float(eligible.max()) if len(eligible) else float("nan")
    return out


def fit_final(frame: pd.DataFrame, model: str, window: int, best_iterations: list[int], threshold: float, out: Path) -> dict:
    all_idx = np.arange(len(frame))
    train = balanced_train_indices(frame, all_idx, SEED + 9999)
    cols = numeric_columns(window, model)
    vocab = top_pfam_vocab(frame.iloc[train][f"w{window}_neighbor_pfams"], 256) if model == "M3" else None
    x = make_matrix(frame, train, cols, model, window, vocab, False, SEED)
    params = dict(LGB_PARAMS)
    params["n_estimators"] = max(1, int(round(float(np.median(best_iterations)))))
    classifier = lgb.LGBMClassifier(**params)
    classifier.fit(x, frame.iloc[train]["label"].to_numpy())
    feature_names = cols + ([f"pfam__{x}" for x in vocab] if vocab else [])
    importance = pd.DataFrame({
        "feature": feature_names,
        "gain": classifier.booster_.feature_importance(importance_type="gain"),
        "split": classifier.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(out / "FEATURE_IMPORTANCE.tsv", sep="\t", index=False)
    bundle = {
        "state": "FINAL_DEV_MODEL_NO_INDEPENDENT_TEST",
        "model_name": model, "window": window, "numeric_columns": cols,
        "pfam_vocabulary": vocab, "threshold_from_grouped_dev_oof": threshold,
        "classifier": classifier, "old_fold0_used": False,
        "training_rows_after_1_to_3_sampling": len(train),
        "training_positive": int(frame.iloc[train].label.sum()),
        "training_negative": int((frame.iloc[train].label == 0).sum()),
        "lightgbm_params": params,
    }
    joblib.dump(bundle, out / "CONTEXT_MODEL.pkl", compress=3)
    return bundle


def bootstrap_ci(frame: pd.DataFrame, score_cols: list[tuple[str, float]], out: Path, reps: int = 100) -> None:
    rng = np.random.default_rng(SEED + 551)
    mags = frame["mag_id"].astype(str).unique()
    mag_codes = pd.Categorical(frame["mag_id"].astype(str), categories=mags).codes
    y = frame["label"].to_numpy(dtype=np.int8)
    rows = []
    for rep in range(reps):
        sampled = rng.integers(0, len(mags), size=len(mags))
        count = np.bincount(sampled, minlength=len(mags)).astype(np.float64)
        weight = count[mag_codes]
        keep = weight > 0
        yy = y[keep]
        ww = weight[keep]
        for name, threshold in score_cols:
            score = frame[name].to_numpy(dtype=float)[keep]
            pred = score >= threshold
            tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1], sample_weight=ww).ravel()
            denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-300))
            rows.append({
                "replicate": rep, "model": name,
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                "mcc": (tp * tn - fp * fn) / denom,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn), "specificity": tn / (tn + fp), "fpr": fp / (tn + fp),
                "auroc": roc_auc_score(yy, score, sample_weight=ww),
                "auprc": average_precision_score(yy, score, sample_weight=ww),
            })
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "MAG_BOOTSTRAP_REPLICATES.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "compresslevel": 1})
    summaries = []
    for (model, metric), values in raw.drop(columns="replicate").set_index("model").stack().groupby(level=[0, 1]):
        arr = values.to_numpy(dtype=float)
        summaries.append({"model": model, "metric": metric, "lower_95": np.quantile(arr, 0.025), "median": np.quantile(arr, 0.5), "upper_95": np.quantile(arr, 0.975), "replicates": reps})
    pd.DataFrame(summaries).to_csv(out / "MAG_BOOTSTRAP_CI.tsv", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    frame = build_wide(out, args.smoke)
    frame = assign_splits(frame, out, args.smoke)
    y = frame["label"].to_numpy(dtype=np.int8)
    oof = pd.DataFrame({
        "protein_id": frame.protein_id, "label": y, "environment": frame.environment,
        "mag_id": frame.mag_id, "contig_id": frame.contig_id, "new_mag_fold": frame.new_mag_fold,
        "old_homology_fold": frame.old_homology_fold, "strict_identity": frame.identity,
        "old_mlp_oof_score": frame.old_mlp_oof_score,
    })
    baseline = metrics(y, frame.old_mlp_oof_score.to_numpy(dtype=float), OLD_THRESHOLD)
    baseline_recall = float(baseline["recall"])
    specs = [("M1", 0, False), ("M2_W2", 2, False), ("M2_W5", 5, False), ("M2_W10", 10, False)]
    iteration_map: dict[str, list[int]] = {}
    comparison = []
    m0 = comparison_metrics(y, oof.old_mlp_oof_score.to_numpy(), OLD_THRESHOLD, baseline_recall)
    m0.update({"model": "M0_FROZEN_MLP128", "window": 0, "feature_contract": "old legal OOF score only", "threshold_source": "original development OOF"})
    comparison.append(m0)
    for name, window, random_control in specs:
        model_type = "M1" if name == "M1" else "M2"
        scores, iterations, _ = train_oof(frame, model_type, window, random_control)
        oof[name] = scores
        iteration_map[name] = iterations
        threshold = threshold_max_mcc(y, scores)
        row = comparison_metrics(y, scores, threshold, baseline_recall)
        row.update({"model": name, "window": window, "feature_contract": "center" if name == "M1" else "center+position+distance+strand+GC", "threshold_source": "grouped development OOF max MCC"})
        comparison.append(row)
    prelim = pd.DataFrame(comparison)
    candidates = prelim[prelim.model.str.startswith("M2_")].sort_values(["fpr_at_baseline_recall", "mcc"], ascending=[True, False])
    main_window = int(candidates.iloc[0].window)
    for name, random_control in [(f"M3_W{main_window}", False), (f"M3_RANDOM_W{main_window}", True)]:
        model_type = "M3_RANDOM" if random_control else "M3"
        scores, iterations, _ = train_oof(frame, model_type, main_window, random_control)
        oof[name] = scores
        iteration_map[name] = iterations
        threshold = threshold_max_mcc(y, scores)
        row = comparison_metrics(y, scores, threshold, baseline_recall)
        row.update({"model": name, "window": main_window, "feature_contract": "M2+fold_train_only_Pfam" if not random_control else "M3 with independently shuffled context", "threshold_source": "grouped development OOF max MCC"})
        comparison.append(row)

    table = pd.DataFrame(comparison)
    selected_name = f"M3_W{main_window}"
    selected = table[table.model == selected_name].iloc[0]
    selected_threshold = float(selected.threshold)
    matched_threshold = float(selected.matched_baseline_recall_threshold)
    decision_fpr_drop = float(m0["fpr"] - selected["fpr_at_baseline_recall"])
    decision_precision_gain = float(selected["precision_at_baseline_recall"] - m0["precision"])
    decision = "MET" if decision_fpr_drop > 0 else "NOT_MET"
    table["selected_main_model"] = table.model.eq(selected_name)
    table["decision"] = np.where(table.model.eq(selected_name), decision, "")
    table.to_csv(out / ("SMOKE_MODEL_COMPARISON.tsv" if args.smoke else "MODEL_COMPARISON.tsv"), sep="\t", index=False)
    oof.to_csv(out / ("SMOKE_OOF_PREDICTIONS.tsv" if args.smoke else "OOF_PREDICTIONS.tsv"), sep="\t", index=False)

    if args.smoke:
        summary = {"state": "SMOKE_TRAIN_COMPLETE", "rows": len(frame), "main_window": main_window, "decision": decision, "old_fold0_rows_used": 0}
        (out / "SMOKE_TRAIN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (out / "SMOKE_TRAIN_COMPLETE").touch()
        print(json.dumps(summary, sort_keys=True))
        return

    bundle = fit_final(frame, "M3", main_window, iteration_map[selected_name], selected_threshold, out)
    env_rows = []
    for env, part in oof.groupby("environment"):
        idx = part.index.to_numpy()
        for name, score_col, threshold in [
            ("M0_FROZEN_MLP128", "old_mlp_oof_score", OLD_THRESHOLD),
            (selected_name, selected_name, selected_threshold),
        ]:
            row = metrics(y[idx], oof.loc[idx, score_col].to_numpy(dtype=float), threshold)
            row.update({"environment": env, "model": name})
            env_rows.append(row)
    pd.DataFrame(env_rows).to_csv(out / "ENVIRONMENT_METRICS.tsv", sep="\t", index=False)

    identity_rows = []
    pos = oof[oof.label == 1].copy()
    pos["identity_bin"] = pd.cut(pos.strict_identity.astype(float), bins=[0, 90, 95, 99, 100.0001], labels=["<=90", "90-95", "95-99", "99-100"], include_lowest=True)
    for identity_bin, part in pos.groupby("identity_bin", observed=True):
        for name, score_col, threshold in [("M0_FROZEN_MLP128", "old_mlp_oof_score", OLD_THRESHOLD), (selected_name, selected_name, selected_threshold)]:
            identity_rows.append({"identity_bin": str(identity_bin), "model": name, "positive_count": len(part), "recall": float((part[score_col] >= threshold).mean()), "threshold": threshold})
    pd.DataFrame(identity_rows).to_csv(out / "IDENTITY_STRATIFIED_RECALL.tsv", sep="\t", index=False)

    old_pred = oof.old_mlp_oof_score.to_numpy() >= OLD_THRESHOLD
    new_pred_matched = oof[selected_name].to_numpy() >= matched_threshold
    neg = y == 0
    posmask = y == 1
    fp_analysis = {
        "old_false_positives": int((old_pred & neg).sum()),
        "old_false_positives_corrected": int((old_pred & neg & ~new_pred_matched).sum()),
        "new_false_positives_introduced": int((~old_pred & neg & new_pred_matched).sum()),
        "old_true_positives": int((old_pred & posmask).sum()),
        "old_true_positives_retained": int((old_pred & posmask & new_pred_matched).sum()),
        "new_true_positives_added": int((~old_pred & posmask & new_pred_matched).sum()),
        "comparison": "new threshold matched to old recall on same grouped-dev OOF centers",
    }
    (out / "FALSE_POSITIVE_FLOW.json").write_text(json.dumps(fp_analysis, indent=2, sort_keys=True) + "\n")

    pred = frame[["protein_id", "center_product", "reference_seed", "identity", "qcov", "scov", "target_class"]].copy()
    pred["old_mlp_oof_score"] = oof.old_mlp_oof_score
    pred["context_oof_score"] = oof[selected_name]
    pred["context_threshold"] = selected_threshold
    pred["final_prediction"] = np.where(pred.context_oof_score >= selected_threshold, "YES", "NO")
    pred["score_status"] = "SCORED_GROUPED_DEV_OOF"
    neighbor_ids = pq.read_table(out / "NEIGHBOR_INDEX.parquet", columns=["protein_id", "relative_position", "neighbor_id"]).to_pandas()
    neighbor_ids.sort_values(["protein_id", "relative_position"], inplace=True)
    neighbor_text = neighbor_ids.groupby("protein_id")["neighbor_id"].agg(lambda x: ",".join(x)).rename("left_to_right_neighbor_ids")
    pred = pred.join(neighbor_text, on="protein_id")
    all_genes = pq.read_table(out / "CONTIG_GENE_STATUS.parquet").to_pandas()
    predictions = all_genes.merge(pred, on="protein_id", how="left", validate="many_to_one")
    predictions["score_status"] = predictions["score_status"].fillna("NOT_SCORED_NO_LEGAL_OLD_OOF")
    predictions["final_prediction"] = predictions["final_prediction"].fillna("NOT_SCORED")
    predictions["context_threshold"] = predictions["context_threshold"].fillna(selected_threshold)
    predictions.to_csv(out / "CONTEXT_CONTIG_PREDICTIONS.tsv", sep="\t", index=False)

    score_cols = [("old_mlp_oof_score", OLD_THRESHOLD), (selected_name, matched_threshold)]
    bootstrap_ci(oof, score_cols, out, reps=args.bootstrap_reps)
    result = {
        "state": "ML_COMPLETE", "evaluation_scope": "strict MAG-grouped development OOF only; no independent test",
        "selected_model": selected_name, "selected_window": main_window,
        "selected_threshold": selected_threshold, "decision": decision,
        "baseline_recall": baseline_recall,
        "baseline_fpr": float(m0["fpr"]), "baseline_precision": float(m0["precision"]),
        "context_matched_recall": float(selected["actual_recall_at_baseline_recall"]),
        "context_fpr_at_matched_recall": float(selected["fpr_at_baseline_recall"]),
        "context_precision_at_matched_recall": float(selected["precision_at_baseline_recall"]),
        "absolute_fpr_reduction": decision_fpr_drop,
        "relative_fpr_reduction": decision_fpr_drop / float(m0["fpr"]) if float(m0["fpr"]) else float("nan"),
        "absolute_precision_gain": decision_precision_gain,
        "old_fold0_rows_used": 0,
        "pfam2vec_status": "SKIPPED_OPTIONAL_NO_GENSIM_IN_AUDITED_LIGHTGBM_ENV",
    }
    (out / "RESULT_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "ML_COMPLETE").touch()
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
