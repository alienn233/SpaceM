from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MODEL_NAME = "M4_ALL_SUPERVISED_PLUS_SAFE_BACKGROUND_CONTEXT_W10"


def embedding_contract(args):
    if args.embedding_mode == "8m":
        return {
            "directory": "CENTER_ESM2",
            "matrix": "CENTER_ESM2_T6_8M_LAYER6_MEAN.f16.npy",
            "dimension": 320,
            "model": "facebook/esm2_t6_8M_UR50D",
            "layer": 6,
        }
    if args.embedding_mode == "650m":
        return {
            "directory": "CENTER_ESM2_650M",
            "matrix": "CENTER_ESM2_T33_650M_LAYER33_MEAN.f16.npy",
            "dimension": 1280,
            "model": "facebook/esm2_t33_650M_UR50D",
            "layer": 33,
        }
    raise ValueError(f"unsupported embedding mode: {args.embedding_mode}")


def load(args, base):
    original, original_esm8, flags = base.load_training(args)
    contract = embedding_contract(args)
    root = Path(args.train_root)
    if args.embedding_mode == "8m":
        original_esm = original_esm8
    else:
        original_esm = np.load(root / contract["matrix"], mmap_mode="r")
        if original_esm.shape != (len(original), contract["dimension"]):
            raise ValueError(
                f"invalid supervised {args.embedding_mode} ESM2 matrix: "
                f"{original_esm.shape} {original_esm.dtype}"
            )
        if original_esm.dtype != np.float16:
            raise ValueError("supervised ESM2 matrix must be float16")
    pool = Path(args.pool_root)
    for marker in ["FEATURES_COMPLETE", f"{contract['directory']}/COMPLETE"]:
        if not (pool / marker).exists():
            raise ValueError(f"missing prerequisite: {pool / marker}")
    pseudo = pq.read_table(pool / "CONTEXT_FEATURES.parquet").to_pandas()
    eggnog = pq.read_table(pool / "EGGNOG_CONTEXT_FEATURES.parquet").to_pandas()
    pseudo = pseudo.merge(eggnog, on="protein_id", how="left", validate="one_to_one")
    split = pd.read_csv(pool / "PSEUDO_SPLIT_MANIFEST.tsv", sep="\t")
    required_split = {
        "protein_id",
        "new_mag_fold",
        "within_positive_w5",
        "pseudo_label_weight",
    }
    if not required_split.issubset(split.columns):
        raise ValueError(f"pseudo split is missing columns: {required_split - set(split.columns)}")
    split_lookup = split.set_index("protein_id")["new_mag_fold"]
    expected_fold = pseudo.protein_id.astype(str).map(split_lookup)
    if expected_fold.isna().any() or not np.array_equal(
        pseudo.new_mag_fold.to_numpy(dtype=np.int8),
        expected_fold.to_numpy(dtype=np.int8),
    ):
        raise ValueError("pseudo-negative context/split alignment failed")
    if pseudo.new_mag_fold.isna().any() or pseudo.pseudo_label_weight.isna().any():
        raise ValueError("incomplete pseudo-negative fold or weight mapping")
    if args.pseudo_pool_window == 5:
        keep = set(
            split.loc[split.within_positive_w5.astype(bool), "protein_id"].astype(str)
        )
        pseudo = pseudo[pseudo.protein_id.astype(str).isin(keep)].copy()
    pseudo["pseudo_label_weight"] = float(args.pseudo_negative_weight)
    pseudo["label"] = 0
    pseudo["source"] = "SAFE_W10_PSEUDO_NEGATIVE"
    original["source"] = "SUPERVISED"
    if set(original.protein_id.astype(str)).intersection(pseudo.protein_id.astype(str)):
        raise ValueError("pseudo-negative pool overlaps supervised centers")

    embedding_root = pool / contract["directory"]
    ids = (embedding_root / "ids.txt").read_text().splitlines()
    matrix = np.load(embedding_root / "embeddings.f16.npy", mmap_mode="r")
    if matrix.shape != (len(ids), contract["dimension"]) or matrix.dtype != np.float16:
        raise ValueError(f"invalid pseudo-negative ESM2 matrix: {matrix.shape} {matrix.dtype}")
    if len(set(ids)) != len(ids) or not np.isfinite(matrix).all():
        raise ValueError("pseudo-negative ESM2 IDs or values failed validation")
    position = {protein_id: index for index, protein_id in enumerate(ids)}
    wanted = pseudo.protein_id.astype(str).tolist()
    if any(protein_id not in position for protein_id in wanted):
        raise ValueError("pseudo-negative ESM2 alignment failed")
    pseudo_esm = np.asarray(matrix[[position[x] for x in wanted]], dtype=np.float16)
    combined_esm = np.vstack([np.asarray(original_esm, dtype=np.float16), pseudo_esm])
    frame = pd.concat([original, pseudo], ignore_index=True, sort=False)
    return frame, combined_esm, flags, len(original), len(pseudo)


def weights(
    frame,
    flags,
    indices,
    original_rows,
    matched_weight,
    pseudo_negative_weight,
):
    values = np.ones(len(indices), dtype=np.float32)
    original_mask = indices < original_rows
    original_indices = indices[original_mask]
    original_labels = frame.label.to_numpy(dtype=np.int8)[original_indices]
    matched = flags.negative_same_positive_mag.to_numpy()[original_indices]
    values[original_mask] = np.where(
        (original_labels == 0) & matched,
        np.float32(matched_weight),
        np.float32(1.0),
    )
    pseudo_mask = ~original_mask
    if pseudo_mask.any():
        values[pseudo_mask] = np.float32(pseudo_negative_weight)
    return values


def make_matrix(frame, esm, indices, vocab, eggnog_vocab, extend_esm2, seed):
    return extend_esm2.make_matrix(
        frame,
        esm,
        None,
        indices,
        extend_esm2.numeric_columns(10, True, True),
        10,
        True,
        vocab,
        True,
        eggnog_vocab,
        False,
        False,
        seed,
    )


def train_oof(args, frame, esm, flags, original_rows, extend_esm2, train_context, base):
    out = Path(args.output)
    checkpoints = out / "fold_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    scores = np.full(original_rows, np.nan, dtype=np.float64)
    iterations = []
    sampling = []
    folds = frame.new_mag_fold.to_numpy(dtype=np.int8)
    pseudo_rows = np.arange(original_rows, len(frame), dtype=np.int64)
    for fold in range(5):
        score_path = checkpoints / f"fold{fold}.scores.f64.npy"
        index_path = checkpoints / f"fold{fold}.indices.i64.npy"
        meta_path = checkpoints / f"fold{fold}.json"
        if score_path.exists() and index_path.exists() and meta_path.exists():
            valid = np.load(index_path)
            fold_scores = np.load(score_path)
            meta = json.loads(meta_path.read_text())
            scores[valid] = fold_scores
            iterations.append(int(meta["best_iteration"]))
            sampling.append(meta["sampling"])
            continue
        valid = np.flatnonzero(folds[:original_rows] == fold)
        original_train = np.flatnonzero(folds[:original_rows] != fold)
        pseudo_train = pseudo_rows[folds[pseudo_rows] != fold]
        train = np.concatenate([original_train, pseudo_train])
        vocab = train_context.top_pfam_vocab(
            frame.iloc[train]["w10_neighbor_pfams"], 256
        )
        eggnog_vocab = train_context.top_pfam_vocab(
            frame.iloc[train]["w10_neighbor_eggnog_tokens"], 512
        )
        x_train = make_matrix(
            frame, esm, train, vocab, eggnog_vocab, extend_esm2, 51000 + fold
        )
        x_valid = make_matrix(
            frame, esm, valid, vocab, eggnog_vocab, extend_esm2, 52000 + fold
        )
        params = dict(train_context.LGB_PARAMS)
        params["n_jobs"] = int(args.threads)
        classifier = lgb.LGBMClassifier(**params)
        classifier.fit(
            x_train,
            frame.iloc[train].label.to_numpy(dtype=np.int8),
            sample_weight=weights(
                frame,
                flags,
                train,
                original_rows,
                args.matched_negative_weight,
                args.pseudo_negative_weight,
            ),
            eval_set=[(x_valid, frame.iloc[valid].label.to_numpy(dtype=np.int8))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        fold_scores = classifier.predict_proba(
            x_valid, num_iteration=classifier.best_iteration_
        )[:, 1].astype(np.float64)
        best_iteration = int(classifier.best_iteration_ or params["n_estimators"])
        summary = {
            "original_positive": int(frame.iloc[original_train].label.eq(1).sum()),
            "original_negative": int(frame.iloc[original_train].label.eq(0).sum()),
            "safe_w10_pseudo_negative": len(pseudo_train),
            "total": len(train),
            "protein_duplication": False,
            "pseudo_negative_source_fold_excluded": int(
                (folds[pseudo_rows] == fold).sum()
            ),
        }
        base.atomic_npy(score_path, fold_scores)
        base.atomic_npy(index_path, valid.astype(np.int64))
        base.atomic_json(
            meta_path,
            {
                "fold": fold,
                "best_iteration": best_iteration,
                "validation_rows": len(valid),
                "sampling": summary,
                "matched_negative_weight": args.matched_negative_weight,
                "pseudo_negative_weight": args.pseudo_negative_weight,
                "pseudo_pool_window": args.pseudo_pool_window,
            },
        )
        scores[valid] = fold_scores
        iterations.append(best_iteration)
        sampling.append(summary)
        print(
            json.dumps(
                {
                    "event": "fold_complete",
                    "fold": fold,
                    "best_iteration": best_iteration,
                    "sampling": summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not np.isfinite(scores).all():
        raise ValueError("non-finite augmented OOF scores")
    return scores, iterations, sampling


def fit_final(
    args,
    frame,
    esm,
    flags,
    original_rows,
    iterations,
    threshold,
    extend_esm2,
    train_context,
    base,
):
    out = Path(args.output) / "MODEL"
    out.mkdir(parents=True, exist_ok=True)
    train = np.arange(len(frame), dtype=np.int64)
    vocab = train_context.top_pfam_vocab(frame["w10_neighbor_pfams"], 256)
    eggnog_vocab = train_context.top_pfam_vocab(
        frame["w10_neighbor_eggnog_tokens"], 512
    )
    x = make_matrix(frame, esm, train, vocab, eggnog_vocab, extend_esm2, 42)
    params = dict(train_context.LGB_PARAMS)
    params["n_jobs"] = int(args.threads)
    params["n_estimators"] = max(1, int(round(float(np.median(iterations)))))
    classifier = lgb.LGBMClassifier(**params)
    classifier.fit(
        x,
        frame.label.to_numpy(dtype=np.int8),
        sample_weight=weights(
            frame,
            flags,
            train,
            original_rows,
            args.matched_negative_weight,
            args.pseudo_negative_weight,
        ),
    )
    cols = extend_esm2.numeric_columns(10, True, True)
    contract = embedding_contract(args)
    embedding_prefix = f"esm2_{args.embedding_mode}_"
    feature_names = (
        [f"{embedding_prefix}{index:04d}" for index in range(contract["dimension"])]
        + cols
        + [f"pfam__{token}" for token in vocab]
        + [f"eggnog__{token}" for token in eggnog_vocab]
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
        "model_name": args.model_name,
        "window": 10,
        "embedding_dimensions": contract["dimension"],
        "embedding_model": contract["model"],
        "embedding_layer": contract["layer"],
        "embedding_mode": args.embedding_mode,
        "numeric_columns": cols,
        "pfam_vocabulary": vocab,
        "eggnog_vocabulary": eggnog_vocab,
        "threshold_from_grouped_dev_oof": float(threshold),
        "classifier": classifier,
        "old_fold0_used": False,
        "old_v2_used": False,
        "neighbor_labels_used_as_features": False,
        "sampling_contract": (
            "all supervised rows plus all safe W10 pseudo-negatives; pseudo MAG fold "
            "excluded with its source positive MAG"
        ),
        "matched_negative_weight": float(args.matched_negative_weight),
        "pseudo_negative_weight": float(args.pseudo_negative_weight),
        "pseudo_pool_window": int(args.pseudo_pool_window),
        "lightgbm_params": params,
    }
    model_path = out / "CONTEXT_MODEL.pkl"
    joblib.dump(bundle, model_path, compress=3)
    base.atomic_json(
        out / "MODEL_CONTRACT.json",
        {
            "state": "M4_FULL_BACKGROUND_AUGMENTED_MODEL_COMPLETE",
            "model_name": args.model_name,
            "feature_contract": (
                f"center_ESM2_{contract['model']}_{contract['dimension']}D+"
                "W10_numeric_context+fold_train_only_Pfam+"
                "per_MAG_eggNOG_numeric+fold_train_only_eggNOG_tokens"
            ),
            "threshold_from_grouped_dev_oof": float(threshold),
            "development_iterations": iterations,
            "original_supervised_rows": original_rows,
            "safe_w10_pseudo_negative_rows": len(frame) - original_rows,
            "matched_negative_weight": float(args.matched_negative_weight),
            "pseudo_negative_weight": float(args.pseudo_negative_weight),
            "pseudo_pool_window": int(args.pseudo_pool_window),
            "embedding_mode": args.embedding_mode,
            "embedding_model": contract["model"],
            "embedding_layer": contract["layer"],
            "embedding_dimensions": contract["dimension"],
            "neighbor_labels_used_as_features": False,
            "old_fold0_used": False,
            "old_v2_used": False,
        },
    )
    (out / "COMPLETE").touch()
    return bundle, model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mgh-root", required=True)
    parser.add_argument("--va-root", required=True)
    parser.add_argument("--va-official", required=True)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--matched-negative-weight", type=float, default=2.0)
    parser.add_argument("--pseudo-negative-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-pool-window", type=int, choices=(5, 10), default=10)
    parser.add_argument("--embedding-mode", choices=("8m", "650m"), default="8m")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    args.add_eggnog = True
    args.all_supervised_negatives = True
    if not 0.0 < args.pseudo_negative_weight <= 1.0:
        raise ValueError("pseudo-negative weight must be in (0, 1]")
    args.model_name = (
        f"{MODEL_NAME}_{args.embedding_mode.upper()}_W{args.pseudo_pool_window}_"
        f"PW{args.pseudo_negative_weight:g}"
    )
    args.external_name = args.model_name
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.train_root) / "code"))
    import extend_esm2
    import train_context
    import train_m3_matched_negative_context as base

    frame, esm, flags, original_rows, pseudo_rows = load(args, base)
    scores, iterations, sampling = train_oof(
        args, frame, esm, flags, original_rows, extend_esm2, train_context, base
    )
    base.atomic_npy(out / "OOF_SCORES.f64.npy", scores)
    labels = frame.iloc[:original_rows].label.to_numpy(dtype=np.int8)
    threshold = train_context.threshold_max_mcc(labels, scores)
    baseline_summary = json.loads(
        (
            Path(args.train_root)
            / "M4_EGGNOG_EVALUATION"
            / "M4_RESULT_SUMMARY.json"
        ).read_text()
    )
    baseline_recall = float(
        baseline_summary["m4_own_threshold_metrics"]["recall"]
    )
    metrics = train_context.comparison_metrics(
        labels, scores, threshold, baseline_recall
    )
    comparisons = {}
    for name, path in [
        ("M4_MATCHED_NEGATIVE_WEIGHT2", Path(args.train_root) / "M4_MATCHED_NEGATIVE_WEIGHT2_V1" / "OOF_SCORES.f64.npy"),
        ("M4_ALL_SUPERVISED_NEGATIVES", Path(args.train_root) / "M4_ALL_SUPERVISED_NEGATIVES_V1" / "OOF_SCORES.f64.npy"),
    ]:
        if path.exists():
            values = np.load(path)
            comparison_threshold = train_context.threshold_max_mcc(labels, values)
            comparisons[name] = train_context.comparison_metrics(
                labels, values, comparison_threshold, baseline_recall
            )
    bundle, model_path = fit_final(
        args,
        frame,
        esm,
        flags,
        original_rows,
        iterations,
        threshold,
        extend_esm2,
        train_context,
        base,
    )
    external = (
        {
            "state": "NOT_SCORED",
            "reason": "external 650M matrices are outside this controlled development comparison",
        }
        if args.skip_external
        else base.score_externals(args, bundle, threshold, extend_esm2)
    )
    result = {
        "state": "M4_FULL_BACKGROUND_AUGMENTED_COMPLETE",
        "model": args.model_name,
        "original_supervised_rows": original_rows,
        "safe_w10_pseudo_negative_rows": pseudo_rows,
        "fold_sampling": sampling,
        "iterations": iterations,
        "threshold": float(threshold),
        "metrics_on_original_grouped_oof_rows": metrics,
        "comparisons": comparisons,
        "external": external,
        "matched_negative_weight": float(args.matched_negative_weight),
        "pseudo_negative_weight": float(args.pseudo_negative_weight),
        "pseudo_pool_window": int(args.pseudo_pool_window),
        "embedding_mode": args.embedding_mode,
        "embedding_contract": embedding_contract(args),
        "validation_contains_pseudo_labels": False,
        "neighbor_labels_used_as_features": False,
        "old_fold0_used": False,
        "old_v2_used": False,
        "model_path": str(model_path),
    }
    base.atomic_json(out / "RESULT.json", result)
    (out / "COMPLETE").touch()
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
