from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WINDOW = 10
M3_MODEL_NAME = "M3_MATCHED_NEGATIVE_CONTEXT_W10"
M4_MODEL_NAME = "M4_MATCHED_NEGATIVE_CONTEXT_W10"


def model_name(args: argparse.Namespace) -> str:
    base = M4_MODEL_NAME if args.add_eggnog else M3_MODEL_NAME
    if args.all_supervised_negatives:
        base = base.replace("_MATCHED_NEGATIVE_", "_ALL_SUPERVISED_NEGATIVES_")
    if args.matched_negative_weight != 1.0:
        return base.replace(
            "_CONTEXT_", f"_WEIGHT{args.matched_negative_weight:g}_CONTEXT_"
        )
    return base


def feature_contract(args: argparse.Namespace) -> str:
    base = "center_ESM2_320+W10_numeric_context+fold_train_only_Pfam"
    if args.add_eggnog:
        return base + "+per_MAG_eggNOG_numeric+fold_train_only_eggNOG_tokens"
    return base


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def matched_flags(frame: pd.DataFrame, neighbor_path: Path) -> pd.DataFrame:
    positive = frame.label.eq(1)
    positive_mags = set(frame.loc[positive, "mag_id"].astype(str))
    positive_contigs = set(
        frame.loc[positive, "mag_id"].astype(str)
        + "\t"
        + frame.loc[positive, "contig_id"].astype(str)
    )
    keys = frame.mag_id.astype(str) + "\t" + frame.contig_id.astype(str)
    flags = pd.DataFrame(index=frame.index)
    flags["negative"] = frame.label.eq(0).to_numpy()
    flags["negative_same_positive_mag"] = (
        frame.label.eq(0) & frame.mag_id.astype(str).isin(positive_mags)
    ).to_numpy()
    flags["negative_same_positive_contig"] = (
        frame.label.eq(0) & keys.isin(positive_contigs)
    ).to_numpy()

    label_map = dict(zip(frame.protein_id.astype(str), frame.label.astype(np.int8)))
    neighbor = pq.read_table(
        neighbor_path,
        columns=["protein_id", "neighbor_id", "is_pad"],
    ).to_pandas()
    neighbor = neighbor[~neighbor.is_pad].copy()
    neighbor["center_label"] = neighbor.protein_id.astype(str).map(label_map)
    neighbor["neighbor_label"] = neighbor.neighbor_id.astype(str).map(label_map)
    direct_ids = set(
        neighbor.loc[
            neighbor.center_label.eq(0) & neighbor.neighbor_label.eq(1),
            "protein_id",
        ].astype(str)
    )
    flags["negative_w10_contains_positive"] = (
        frame.label.eq(0) & frame.protein_id.astype(str).isin(direct_ids)
    ).to_numpy()
    return flags


def select_training_rows(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    train_all: np.ndarray,
    seed: int,
    all_supervised_negatives: bool = False,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    labels = frame.label.to_numpy(dtype=np.int8)
    pos = train_all[labels[train_all] == 1]
    neg = train_all[labels[train_all] == 0]
    if all_supervised_negatives:
        sampled = train_all.copy()
        rng.shuffle(sampled)
        selected_neg = sampled[labels[sampled] == 0]
        return sampled, {
            "positive": len(pos),
            "negative": len(selected_neg),
            "total": len(sampled),
            "negative_same_positive_mag": int(
                flags.negative_same_positive_mag.to_numpy()[selected_neg].sum()
            ),
            "negative_same_positive_contig": int(
                flags.negative_same_positive_contig.to_numpy()[selected_neg].sum()
            ),
            "negative_w10_contains_positive": int(
                flags.negative_w10_contains_positive.to_numpy()[selected_neg].sum()
            ),
            "all_supervised_negatives": True,
            "protein_duplication": False,
        }
    target = min(len(neg), 3 * len(pos))

    same_mag = neg[flags.negative_same_positive_mag.to_numpy()[neg]]
    if len(same_mag) <= target:
        chosen = same_mag.copy()
    else:
        direct = same_mag[flags.negative_w10_contains_positive.to_numpy()[same_mag]]
        same_contig = np.setdiff1d(
            same_mag[flags.negative_same_positive_contig.to_numpy()[same_mag]],
            direct,
            assume_unique=False,
        )
        other = np.setdiff1d(same_mag, np.concatenate([direct, same_contig]))
        blocks = [direct]
        capacity = target - len(direct)
        take_contig = min(capacity, len(same_contig))
        if take_contig:
            blocks.append(rng.choice(same_contig, take_contig, replace=False))
        capacity -= take_contig
        if capacity:
            blocks.append(rng.choice(other, capacity, replace=False))
        chosen = np.concatenate(blocks)

    remaining_needed = target - len(chosen)
    if remaining_needed:
        remaining = np.setdiff1d(neg, chosen, assume_unique=False)
        groups = frame.iloc[remaining].negative_selection_group.fillna("").to_numpy()
        hard = remaining[groups == "large_cluster"]
        regular = remaining[groups != "large_cluster"]
        chosen_groups = frame.iloc[chosen].negative_selection_group.fillna("").to_numpy()
        chosen_hard = int((chosen_groups == "large_cluster").sum())
        desired_hard = target // 2
        hard_n = min(len(hard), remaining_needed, max(0, desired_hard - chosen_hard))
        regular_n = min(len(regular), remaining_needed - hard_n)
        additions: list[np.ndarray] = []
        if hard_n:
            additions.append(rng.choice(hard, hard_n, replace=False))
        if regular_n:
            additions.append(rng.choice(regular, regular_n, replace=False))
        used = hard_n + regular_n
        if used < remaining_needed:
            used_ids = np.concatenate(additions) if additions else np.array([], dtype=np.int64)
            pool = np.setdiff1d(remaining, used_ids, assume_unique=False)
            additions.append(rng.choice(pool, remaining_needed - used, replace=False))
        chosen = np.concatenate([chosen] + additions)

    sampled = np.concatenate([pos, chosen])
    rng.shuffle(sampled)
    selected_neg = sampled[labels[sampled] == 0]
    summary = {
        "positive": len(pos),
        "negative": len(selected_neg),
        "total": len(sampled),
        "negative_same_positive_mag": int(
            flags.negative_same_positive_mag.to_numpy()[selected_neg].sum()
        ),
        "negative_same_positive_contig": int(
            flags.negative_same_positive_contig.to_numpy()[selected_neg].sum()
        ),
        "negative_w10_contains_positive": int(
            flags.negative_w10_contains_positive.to_numpy()[selected_neg].sum()
        ),
        "all_supervised_negatives": False,
        "protein_duplication": False,
    }
    return sampled, summary


def training_weights(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    indices: np.ndarray,
    matched_negative_weight: float,
) -> np.ndarray | None:
    if matched_negative_weight == 1.0:
        return None
    values = np.ones(len(indices), dtype=np.float32)
    labels = frame.label.to_numpy(dtype=np.int8)[indices]
    matched = flags.negative_same_positive_mag.to_numpy()[indices]
    values[(labels == 0) & matched] = np.float32(matched_negative_weight)
    return values


def load_training(args: argparse.Namespace):
    root = Path(args.train_root)
    frame = pq.read_table(root / "CONTEXT_FEATURES.parquet").to_pandas()
    if args.add_eggnog:
        eggnog = pq.read_table(root / "EGGNOG_CONTEXT_FEATURES.parquet").to_pandas()
        frame = frame.merge(eggnog, on="protein_id", how="left", validate="one_to_one")
    split = pd.read_csv(
        root / "SPLIT_MANIFEST.tsv", sep="\t", usecols=["protein_id", "new_mag_fold"]
    )
    frame = frame.merge(split, on="protein_id", how="left", validate="one_to_one")
    if frame.new_mag_fold.isna().any() or set(frame.new_mag_fold.astype(int)) != {0, 1, 2, 3, 4}:
        raise ValueError("MAG-grouped development fold contract failed")
    if int((frame.old_homology_fold == 0).sum()) != 0:
        raise ValueError("sealed old fold0 entered matched-negative training")
    if frame.groupby("mag_id").new_mag_fold.nunique().max() != 1:
        raise ValueError("a MAG crosses outer folds")
    index = pd.read_csv(root / "CENTER_ESM2_INDEX.tsv.gz", sep="\t")
    if not np.array_equal(
        frame.protein_id.astype(str).to_numpy(), index.protein_id.astype(str).to_numpy()
    ):
        raise ValueError("training context/ESM2 row order mismatch")
    esm = np.load(root / "CENTER_ESM2_T6_8M_LAYER6_MEAN.f16.npy", mmap_mode="r")
    if esm.shape != (len(frame), 320) or esm.dtype != np.float16:
        raise ValueError(f"invalid center ESM2 matrix: {esm.shape} {esm.dtype}")
    flags = matched_flags(frame, root / "NEIGHBOR_INDEX.parquet")
    return frame, esm, flags


def train_oof(args: argparse.Namespace, frame, esm, flags, extend_esm2, train_context):
    out = Path(args.output)
    checkpoints = out / "fold_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    iterations: list[int] = []
    sampling: list[dict] = []
    folds = frame.new_mag_fold.to_numpy(dtype=np.int8)
    cols = extend_esm2.numeric_columns(WINDOW, True, args.add_eggnog)
    for fold in range(5):
        score_path = checkpoints / f"fold{fold}.scores.f64.npy"
        index_path = checkpoints / f"fold{fold}.indices.i64.npy"
        meta_path = checkpoints / f"fold{fold}.json"
        if score_path.exists() and index_path.exists() and meta_path.exists():
            valid = np.load(index_path)
            fold_scores = np.load(score_path)
            meta = json.loads(meta_path.read_text())
            if fold_scores.shape != valid.shape or not np.isfinite(fold_scores).all():
                raise ValueError(f"invalid fold {fold} checkpoint")
            scores[valid] = fold_scores
            iterations.append(int(meta["best_iteration"]))
            sampling.append(meta["sampling"])
            continue

        valid = np.flatnonzero(folds == fold)
        train_all = np.flatnonzero(folds != fold)
        train, sample_summary = select_training_rows(
            frame,
            flags,
            train_all,
            train_context.SEED + 31001 + fold * 1000,
            args.all_supervised_negatives,
        )
        vocab = train_context.top_pfam_vocab(
            frame.iloc[train][f"w{WINDOW}_neighbor_pfams"], 256
        )
        eggnog_vocab = (
            train_context.top_pfam_vocab(
                frame.iloc[train][f"w{WINDOW}_neighbor_eggnog_tokens"], 512
            )
            if args.add_eggnog
            else None
        )
        x_train = extend_esm2.make_matrix(
            frame, esm, None, train, cols, WINDOW, True, vocab,
            args.add_eggnog, eggnog_vocab, False, False,
            train_context.SEED + fold * 17 + 901,
        )
        x_valid = extend_esm2.make_matrix(
            frame, esm, None, valid, cols, WINDOW, True, vocab,
            args.add_eggnog, eggnog_vocab, False, False,
            train_context.SEED + fold * 17 + 902,
        )
        params = dict(train_context.LGB_PARAMS)
        params["n_jobs"] = int(args.threads)
        classifier = lgb.LGBMClassifier(**params)
        sample_weight = training_weights(
            frame, flags, train, args.matched_negative_weight
        )
        classifier.fit(
            x_train,
            frame.iloc[train].label.to_numpy(dtype=np.int8),
            sample_weight=sample_weight,
            eval_set=[(x_valid, frame.iloc[valid].label.to_numpy(dtype=np.int8))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        fold_scores = classifier.predict_proba(
            x_valid, num_iteration=classifier.best_iteration_
        )[:, 1].astype(np.float64)
        best_iteration = int(classifier.best_iteration_ or params["n_estimators"])
        atomic_npy(score_path, fold_scores)
        atomic_npy(index_path, valid.astype(np.int64))
        atomic_json(
            meta_path,
            {
                "fold": fold,
                "best_iteration": best_iteration,
                "validation_rows": len(valid),
                "sampling": sample_summary,
                "matched_negative_weight": args.matched_negative_weight,
            },
        )
        scores[valid] = fold_scores
        iterations.append(best_iteration)
        sampling.append(sample_summary)
        print(json.dumps({"event": "fold_complete", "fold": fold, "best_iteration": best_iteration, "sampling": sample_summary}, sort_keys=True), flush=True)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite matched-negative OOF scores")
    return scores, iterations, sampling


def fit_final(args, frame, esm, flags, iterations, threshold, extend_esm2, train_context):
    out = Path(args.output) / "MODEL"
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "CONTEXT_MODEL.pkl"
    if model_path.exists() and (out / "COMPLETE").exists():
        return joblib.load(model_path), model_path

    all_idx = np.arange(len(frame), dtype=np.int64)
    train, sample_summary = select_training_rows(
        frame,
        flags,
        all_idx,
        train_context.SEED + 39999,
        args.all_supervised_negatives,
    )
    cols = extend_esm2.numeric_columns(WINDOW, True, args.add_eggnog)
    vocab = train_context.top_pfam_vocab(frame.iloc[train][f"w{WINDOW}_neighbor_pfams"], 256)
    eggnog_vocab = (
        train_context.top_pfam_vocab(
            frame.iloc[train][f"w{WINDOW}_neighbor_eggnog_tokens"], 512
        )
        if args.add_eggnog
        else None
    )
    x = extend_esm2.make_matrix(
        frame, esm, None, train, cols, WINDOW, True, vocab,
        args.add_eggnog, eggnog_vocab, False, False, train_context.SEED,
    )
    params = dict(train_context.LGB_PARAMS)
    params["n_jobs"] = int(args.threads)
    params["n_estimators"] = max(1, int(round(float(np.median(iterations)))))
    classifier = lgb.LGBMClassifier(**params)
    classifier.fit(
        x,
        frame.iloc[train].label.to_numpy(dtype=np.int8),
        sample_weight=training_weights(
            frame, flags, train, args.matched_negative_weight
        ),
    )
    feature_names = (
        [f"esm2_{i:03d}" for i in range(320)]
        + cols
        + [f"pfam__{token}" for token in vocab]
        + ([f"eggnog__{token}" for token in eggnog_vocab] if eggnog_vocab else [])
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": classifier.booster_.feature_importance(importance_type="gain"),
            "split": classifier.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    importance.to_csv(out / "FEATURE_IMPORTANCE.tsv", sep="\t", index=False)
    bundle = {
        "state": "FINAL_DEV_MODEL_NO_INDEPENDENT_TEST",
        "model_name": model_name(args),
        "window": WINDOW,
        "embedding_dimensions": 320,
        "numeric_columns": cols,
        "pfam_vocabulary": vocab,
        "eggnog_vocabulary": eggnog_vocab,
        "threshold_from_grouped_dev_oof": float(threshold),
        "classifier": classifier,
        "old_fold0_used": False,
        "old_v2_used": False,
        "neighbor_labels_used_as_features": False,
        "sampling_contract": (
            "all supervised negative centers"
            if args.all_supervised_negatives
            else "all same-positive-MAG negative centers, then fill to global 1:3"
        ),
        "matched_negative_weight": args.matched_negative_weight,
        "final_sampling": sample_summary,
        "lightgbm_params": params,
    }
    joblib.dump(bundle, model_path, compress=3)
    atomic_json(
        out / "MODEL_CONTRACT.json",
        {
            "state": (
                "M4_MATCHED_NEGATIVE_CONTEXT_MODEL_COMPLETE"
                if args.add_eggnog
                else "M3_MATCHED_NEGATIVE_CONTEXT_MODEL_COMPLETE"
            ),
            "model_name": model_name(args),
            "feature_contract": feature_contract(args),
            "threshold_from_grouped_dev_oof": float(threshold),
            "development_iterations": iterations,
            "sampling": sample_summary,
            "matched_negative_weight": args.matched_negative_weight,
            "eggnog_used": args.add_eggnog,
            "neighbor_labels_used_as_features": False,
            "old_fold0_used": False,
            "old_v2_used": False,
        },
    )
    (out / "COMPLETE").touch()
    return bundle, model_path


def score_matrix(frame, embedding_root: Path, key: str, bundle, extend_esm2):
    ids = (embedding_root / "ids.txt").read_text().splitlines()
    embeddings = np.load(embedding_root / "embeddings.f16.npy", mmap_mode="r")
    if embeddings.shape != (len(ids), 320) or embeddings.dtype != np.float16:
        raise ValueError(f"invalid external ESM2 matrix: {embedding_root}")
    positions = {protein_id: index for index, protein_id in enumerate(ids)}
    wanted = frame[key].astype(str).tolist()
    if any(protein_id not in positions for protein_id in wanted):
        raise ValueError(f"external ESM2 alignment failed: {embedding_root}")
    aligned = np.asarray(embeddings[[positions[protein_id] for protein_id in wanted]], dtype=np.float16)
    matrix = extend_esm2.make_matrix(
        frame, aligned, None, np.arange(len(frame)), bundle["numeric_columns"],
        WINDOW, True, bundle["pfam_vocabulary"],
        bundle["eggnog_vocabulary"] is not None, bundle["eggnog_vocabulary"],
        False, False, 42,
    )
    return bundle["classifier"].predict_proba(matrix)[:, 1]


def aggregate_accessions(locus: pd.DataFrame, threshold: float) -> pd.DataFrame:
    accession = locus.groupby("accession", as_index=False).agg(
        scored_loci=("protein_id", "size"),
        score_max=("score", "max"),
        score_mean=("score", "mean"),
        score_min=("score", "min"),
        yes_loci=("prediction", lambda values: int((values == "YES").sum())),
    )
    accession["threshold"] = threshold
    accession["yes"] = (accession.score_max >= threshold).astype(np.int8)
    accession["prediction"] = np.where(accession.yes.eq(1), "YES", "NO")
    return accession


def score_externals(args, bundle, threshold, extend_esm2):
    result = {}
    mgh_root = Path(args.mgh_root)
    if getattr(args, "external_name", ""):
        external_name = args.external_name
    elif args.all_supervised_negatives:
        prefix = "M4" if args.add_eggnog else "M3"
        external_name = f"{prefix}_ALL_SUPERVISED_NEGATIVES_CONTEXT"
        if args.matched_negative_weight != 1.0:
            external_name = external_name.replace(
                "_CONTEXT", f"_WEIGHT{args.matched_negative_weight:g}_CONTEXT"
            )
    elif args.matched_negative_weight != 1.0:
        prefix = "M4" if args.add_eggnog else "M3"
        external_name = (
            f"{prefix}_MATCHED_NEGATIVE_WEIGHT{args.matched_negative_weight:g}_CONTEXT"
        )
    else:
        external_name = "M4_MATCHED_NEGATIVE_CONTEXT" if args.add_eggnog else "M3_MATCHED_NEGATIVE_CONTEXT"
    mgh_out = mgh_root / external_name
    mgh_out.mkdir(parents=True, exist_ok=True)
    mgh_frame = pq.read_table(mgh_root / "CONTEXT_FEATURES.parquet").to_pandas()
    if args.add_eggnog:
        mgh_eggnog = pq.read_table(mgh_root / "EGGNOG_CONTEXT_FEATURES.parquet").to_pandas()
        mgh_frame = mgh_frame.merge(
            mgh_eggnog, on="protein_id", how="left", validate="one_to_one"
        )
    mgh_score = score_matrix(mgh_frame, mgh_root / "ESM2_M4", "accession", bundle, extend_esm2)
    mgh_locus = mgh_frame[["protein_id", "accession", "contig_id", "start", "end", "strand", "center_product", "center_pfams"]].copy()
    mgh_locus["score"] = mgh_score
    mgh_locus["prediction"] = np.where(mgh_score >= threshold, "YES", "NO")
    mgh_locus.to_csv(mgh_out / "LOCUS_PREDICTIONS.tsv", sep="\t", index=False)
    mgh_accession = aggregate_accessions(mgh_locus, threshold)
    universe = pd.read_csv(mgh_root / "LOCUS_MAP.tsv", sep="\t").groupby("protein_id", as_index=False).agg(total_loci=("locus_id", "size"))
    universe.rename(columns={"protein_id": "accession"}, inplace=True)
    mgh_accession = universe.merge(mgh_accession, on="accession", how="left", validate="one_to_one")
    mgh_accession["scored_loci"] = mgh_accession.scored_loci.fillna(0).astype(int)
    mgh_accession["yes_loci"] = mgh_accession.yes_loci.fillna(0).astype(int)
    mgh_accession["yes"] = mgh_accession.yes.fillna(0).astype(np.int8)
    mgh_accession["prediction"] = np.where(mgh_accession.scored_loci.eq(0), "NOT_SCORED_SHORT_CONTIG", np.where(mgh_accession.yes.eq(1), "YES", "NO"))
    mgh_accession.to_csv(mgh_out / "ACCESSION_PREDICTIONS.tsv", sep="\t", index=False)
    ref = pd.read_csv(mgh_root / "REFERENCE_POSITIVE_FLAGS.tsv", sep="\t")
    mgh_eval = mgh_accession.merge(ref, on="accession", how="left", validate="one_to_one")
    mgh_metrics = []
    for flag in ["REFERENCE_ANY", "ARG", "MRG", "VF", "MGE"]:
        mask = mgh_eval[flag].eq(1)
        caught = int(mgh_eval.loc[mask, "yes"].sum())
        mgh_metrics.append({"reference_subset": flag, "reference_positives": int(mask.sum()), "detected": caught, "recall": caught / int(mask.sum()) if int(mask.sum()) else None})
    pd.DataFrame(mgh_metrics).to_csv(mgh_out / "REFERENCE_METRICS.tsv", sep="\t", index=False)
    result["MGH78578"] = {"accessions": len(mgh_accession), "scored": int(mgh_accession.scored_loci.gt(0).sum()), "yes": int(mgh_accession.yes.sum()), "no": int(mgh_accession.prediction.eq("NO").sum()), "yes_rate": float(mgh_accession.yes.mean()), "reference_metrics": mgh_metrics}
    atomic_json(mgh_out / "RESULT.json", result["MGH78578"])
    (mgh_out / "COMPLETE").touch()

    va_root = Path(args.va_root)
    va_source = va_root / "M4_FULL_GENOME"
    va_out = va_root / external_name
    va_out.mkdir(parents=True, exist_ok=True)
    va_frame = pq.read_table(va_source / "CONTEXT_FEATURES.parquet").to_pandas()
    if args.add_eggnog:
        va_eggnog = pq.read_table(va_source / "EGGNOG_CONTEXT_FEATURES.parquet").to_pandas()
        va_frame = va_frame.merge(
            va_eggnog, on="protein_id", how="left", validate="one_to_one"
        )
    va_score = score_matrix(va_frame, va_source / "ESM2_ALL", "protein_id", bundle, extend_esm2)
    va_locus = va_frame[["protein_id", "accession", "contig_id", "start", "end", "strand", "center_product", "center_pfams"]].copy()
    va_locus["score"] = va_score
    va_locus["prediction"] = np.where(va_score >= threshold, "YES", "NO")
    va_locus.to_csv(va_out / "LOCUS_PREDICTIONS.tsv", sep="\t", index=False)
    va_accession = aggregate_accessions(va_locus, threshold)
    va_accession.to_csv(va_out / "ACCESSION_PREDICTIONS.tsv", sep="\t", index=False)
    official = pd.read_csv(args.va_official, sep="\t")
    official["OFFICIAL_ANY"] = official[["ARG", "MRG", "VF", "MGE"]].max(axis=1)
    va_eval = va_accession.merge(official, left_on="accession", right_on="protein_id", how="left", validate="one_to_one")
    va_metrics = []
    for flag in ["OFFICIAL_ANY", "ARG", "MRG", "VF", "MGE"]:
        mask = va_eval[flag].fillna(0).eq(1)
        caught = int(va_eval.loc[mask, "yes"].sum())
        va_metrics.append({"reference_subset": flag, "reference_positives": int(mask.sum()), "detected": caught, "recall": caught / int(mask.sum()) if int(mask.sum()) else None})
    pd.DataFrame(va_metrics).to_csv(va_out / "OFFICIAL_POSITIVE_METRICS.tsv", sep="\t", index=False)
    result["VA58522"] = {"accessions": len(va_accession), "yes": int(va_accession.yes.sum()), "no": int((va_accession.yes == 0).sum()), "yes_rate": float(va_accession.yes.mean()), "official_positive_metrics": va_metrics}
    atomic_json(va_out / "RESULT.json", result["VA58522"])
    (va_out / "COMPLETE").touch()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mgh-root", required=True)
    parser.add_argument("--va-root", required=True)
    parser.add_argument("--va-official", required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--add-eggnog", action="store_true")
    parser.add_argument("--matched-negative-weight", type=float, default=1.0)
    parser.add_argument("--all-supervised-negatives", action="store_true")
    args = parser.parse_args()
    if args.matched_negative_weight < 1.0:
        raise ValueError("matched-negative weight must be at least 1.0")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.train_root) / "code"))
    import extend_esm2
    import train_context

    frame, esm, flags = load_training(args)
    flag_summary = {column: int(flags[column].sum()) for column in flags.columns}
    atomic_json(
        out / "MATCHED_NEGATIVE_AUDIT.json",
        {
            "rows": len(frame),
            **flag_summary,
            "matched_negative_weight": args.matched_negative_weight,
            "all_supervised_negatives": args.all_supervised_negatives,
            "neighbor_labels_used_as_features": False,
        },
    )
    scores, iterations, sampling = train_oof(args, frame, esm, flags, extend_esm2, train_context)
    atomic_npy(out / "OOF_SCORES.f64.npy", scores)
    y = frame.label.to_numpy(dtype=np.int8)
    threshold = train_context.threshold_max_mcc(y, scores)
    baseline = json.loads((Path(args.train_root) / "M4_EGGNOG_EVALUATION" / "M4_RESULT_SUMMARY.json").read_text())
    baseline_key = "m4_own_threshold_metrics" if args.add_eggnog else "m3_own_threshold_metrics"
    baseline_recall = float(baseline[baseline_key]["recall"])
    metrics = train_context.comparison_metrics(y, scores, threshold, baseline_recall)
    baseline_score_name = "M4_ESM_EGGNOG_W10.oof.f64.npy" if args.add_eggnog else "M3_ESM_W10.oof.f64.npy"
    baseline_scores = np.load(
        Path(args.train_root) / "M4_EGGNOG_EVALUATION" / baseline_score_name
    )
    baseline_threshold = float(baseline[baseline_key]["threshold"])
    baseline_metrics = train_context.comparison_metrics(y, baseline_scores, baseline_threshold, baseline_recall)
    subgroup_rows = []
    for name, mask in [
        ("NEG_W10_CONTAINS_POSITIVE", flags.negative_w10_contains_positive.to_numpy()),
        ("NEG_SAME_POSITIVE_CONTIG", flags.negative_same_positive_contig.to_numpy()),
        ("NEG_SAME_POSITIVE_MAG", flags.negative_same_positive_mag.to_numpy()),
        ("ALL_NEGATIVE", flags.negative.to_numpy()),
    ]:
        n = int(mask.sum())
        subgroup_rows.append({"subgroup": name, "negative_rows": n, "baseline_model": "M4_ESM_EGGNOG_W10" if args.add_eggnog else "M3_ESM_W10", "baseline_model_fp": int((baseline_scores[mask] >= baseline_threshold).sum()), "baseline_model_fpr": float((baseline_scores[mask] >= baseline_threshold).mean()), "matched_model_fp": int((scores[mask] >= threshold).sum()), "matched_model_fpr": float((scores[mask] >= threshold).mean())})
    pd.DataFrame(subgroup_rows).to_csv(out / "NEGATIVE_SUBGROUP_FPR.tsv", sep="\t", index=False)
    bundle, model_path = fit_final(args, frame, esm, flags, iterations, threshold, extend_esm2, train_context)
    external = score_externals(args, bundle, threshold, extend_esm2)
    summary = {
        "state": (
            "M4_MATCHED_NEGATIVE_CONTEXT_COMPLETE"
            if args.add_eggnog
            else "M3_MATCHED_NEGATIVE_CONTEXT_COMPLETE"
        ),
        "model": model_name(args),
        "feature_contract": feature_contract(args),
        "sampling_contract": (
            "use every available supervised negative center without duplication"
            if args.all_supervised_negatives
            else "force all available negative centers from positive-containing MAGs, including overlapping W10, then fill to 1:3"
        ),
        "matched_negative_weight": args.matched_negative_weight,
        "all_supervised_negatives": args.all_supervised_negatives,
        "matched_negative_audit": flag_summary,
        "fold_sampling": sampling,
        "iterations": iterations,
        "threshold": float(threshold),
        "baseline_model": "M4_ESM_EGGNOG_W10" if args.add_eggnog else "M3_ESM_W10",
        "baseline_model_metrics": baseline_metrics,
        "matched_model_metrics": metrics,
        "external": external,
        "eggnog_used": args.add_eggnog,
        "neighbor_labels_used_as_features": False,
        "old_fold0_used": False,
        "old_v2_used": False,
        "model_path": str(model_path),
    }
    atomic_json(out / "RESULT.json", summary)
    (out / "COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
