from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_outside_rate(score: np.ndarray, y: np.ndarray, rate: float) -> float:
    if not 0 < rate < 1 or not np.any(y == 0):
        raise RuntimeError('invalid outside reference rate')
    return float(np.quantile(score[y == 0], 1 - rate, method='higher'))


def cluster_ci(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray,
               group: np.ndarray, reps: int = 2000) -> tuple[float, float]:
    positive = pd.DataFrame({'group': group[y == 1], 'model': pred[y == 1],
                             'baseline': baseline[y == 1]})
    agg = positive.groupby('group').agg(n=('model', 'size'), model=('model', 'sum'),
                                        baseline=('baseline', 'sum'))
    values = agg[['n', 'model', 'baseline']].to_numpy(np.float64)
    rng = np.random.default_rng(20261031)
    sample = rng.integers(0, len(values), size=(reps, len(values)))
    totals = values[sample].sum(axis=1)
    delta = (totals[:, 1] - totals[:, 2]) / totals[:, 0]
    return tuple(float(x) for x in np.quantile(delta, [.025, .975]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=32)
    args = ap.parse_args()
    root = args.root
    source = root / 'v3hn_spatial'
    out = root / 'reference_localization_dev'
    out.mkdir(exist_ok=True)
    if (out / 'DEV_OOF_COMPLETE').exists():
        print((out / 'dev_audit.json').read_text(), flush=True)
        return
    if not (source / 'CARGO_EVIDENCE_COMPLETE').exists():
        raise RuntimeError('cargo evidence incomplete')


    data = pd.read_parquet(source / 'CARGO_EVIDENCE_FULL.parquet',
                           filters=[('partition', '==', 'development')])
    data = data.loc[data.reference_position.isin(('inside', 'outside'))].copy()
    if not data.partition.eq('development').all():
        raise RuntimeError('protected partition crossed development boundary')
    if not data.loc[data.reference_position.eq('inside'), 'inside_element_type'].isin(('ICE', 'IME')).all():
        raise RuntimeError('positive reference type is not ICE/IME')
    split = pd.read_csv(root / 'full_cohort_split/ICE_IME_FULL_SPLIT.tsv', sep='\t',
                        usecols=['source_accession', 'connected_group', 'partition'])
    split = split.loc[split.partition.eq('development'),
                      ['source_accession', 'connected_group']].drop_duplicates()
    if split.source_accession.duplicated().any():
        raise RuntimeError('accession maps to more than one connected group')
    data = data.merge(split, left_on='accession', right_on='source_accession',
                      how='left', validate='many_to_one')
    if data.connected_group.isna().any():
        raise RuntimeError('missing development connected group')
    y = data.reference_position.eq('inside').to_numpy(np.int8)
    group = data.connected_group.to_numpy(str)
    baseline = data.baseline_predicted_mge_within_5kb.to_numpy(bool)
    baseline_outside_rate = float(baseline[y == 0].mean())
    if not 0 < baseline_outside_rate < 1:
        raise RuntimeError('degenerate baseline outside rate')
    feature_cols = json.loads((source / 'feature_columns.json').read_text())
    spatial = [c for c in feature_cols if c.startswith(('p_', 'gate_score', 'mge_', 'nearest_mge_'))]
    mag = [c for c in feature_cols if c.startswith(('a_', 'b_', 'c_'))]
    variants = {'spatial': spatial, 'spatial_mag': spatial + mag, 'all_evidence': feature_cols}
    for name, columns in variants.items():
        if not columns or len(columns) != len(set(columns)) or set(columns) - set(data.columns):
            raise RuntimeError(f'invalid features for {name}')
        if 'baseline_predicted_mge_within_5kb' in columns:
            raise RuntimeError('fixed baseline leaked into model input')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261031)
    folds = np.full(len(data), -1, np.int8)
    for fold, (_, val) in enumerate(cv.split(np.zeros(len(data)), y, group)):
        folds[val] = fold
    if (folds < 0).any() or pd.DataFrame({'group': group, 'fold': folds}).groupby('group').fold.nunique().max() != 1:
        raise RuntimeError('group CV leakage')
    weight = (1 / np.sqrt(data.groupby('connected_group').connected_group.transform('size'))).to_numpy(np.float32)
    weight = weight / weight.mean()
    weight[y == 0] *= 0.5

    metrics = []
    oof_tables = []
    for name, columns in variants.items():
        x = data[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
        score = np.full(len(data), np.nan, np.float32)
        for fold in range(5):
            fit, val = folds != fold, folds == fold
            params = {'objective': 'binary', 'learning_rate': 0.05,
                'num_leaves': 15, 'max_depth': 5, 'min_data_in_leaf': 40,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                'bagging_freq': 1, 'lambda_l2': 5.0,
                'seed': 20261031 + fold, 'num_threads': args.threads,
                'verbosity': -1}
            train = lgb.Dataset(x[fit], label=y[fit], weight=weight[fit],
                                feature_name=columns)
            model = lgb.train(params, train, num_boost_round=300)
            score[val] = model.predict(x[val], num_threads=args.threads)
            model.save_model(str(out / f'{name}_fold{fold}.txt'))
            print(json.dumps({'variant': name, 'fold': fold,
                              'validation_rows': int(val.sum())}), flush=True)
        if np.isnan(score).any():
            raise RuntimeError(f'incomplete development OOF for {name}')
        threshold = threshold_at_outside_rate(score, y, baseline_outside_rate)
        call = score >= threshold
        ci = cluster_ci(y, call, baseline, group)
        metric = {
            'variant': name,
            'features': len(columns),
            'development_rows': len(data),
            'development_connected_groups': len(set(group)),
            'reference_inside_ICE_IME': int(y.sum()),
            'reference_outside_noisy_controls': int((y == 0).sum()),
            'baseline_5kb_inside_recall': float(baseline[y == 1].mean()),
            'baseline_5kb_outside_alert_rate': baseline_outside_rate,
            'model_threshold_matched_outside_rate': threshold,
            'model_inside_recall_matched_outside_rate': float(call[y == 1].mean()),
            'model_outside_alert_rate': float(call[y == 0].mean()),
            'recall_delta_vs_5kb': float(call[y == 1].mean() - baseline[y == 1].mean()),
            'recall_delta_cluster_ci_low': ci[0],
            'recall_delta_cluster_ci_high': ci[1],
            'auroc_reference_localization': float(roc_auc_score(y, score)),
            'auprc_reference_localization': float(average_precision_score(y, score)),
            'outside_is_not_biological_negative': True,
        }
        for cargo_class in ('ARG', 'MRG'):
            ix = (data.cargo_class == cargo_class).to_numpy() & (y == 1)
            metric[f'{cargo_class}_inside_recall'] = float(call[ix].mean())
            metric[f'{cargo_class}_baseline_inside_recall'] = float(baseline[ix].mean())
        for element_type in ('ICE', 'IME'):
            ix = (data.inside_element_type == element_type).to_numpy() & (y == 1)
            metric[f'{element_type}_inside_recall'] = float(call[ix].mean())
            metric[f'{element_type}_baseline_inside_recall'] = float(baseline[ix].mean())
        metrics.append(metric)
        oof_tables.append(pd.DataFrame({'host_protein_id': data.host_protein_id,
            'accession': data.accession, 'connected_group': group, 'fold': folds,
            'cargo_class': data.cargo_class, 'inside_element_type': data.inside_element_type,
            'reference_position': data.reference_position, 'variant': name,
            'score': score, 'model_call': call, 'baseline_call': baseline}))
        print(json.dumps(metric), flush=True)
    pd.DataFrame(metrics).to_csv(out / 'DEV_OOF_METRICS.tsv', sep='\t', index=False)
    pd.concat(oof_tables, ignore_index=True).to_parquet(out / 'DEV_OOF_PREDICTIONS.parquet', index=False)
    (out / 'feature_families.json').write_text(json.dumps(variants, indent=2) + '\n')
    audit = {'state': 'DEV_OOF_COMPLETE', 'CPU_LightGBM': lgb.__version__,
             'fresh_test_partition_not_read': True,
             'reference_outside_noisy_control_not_true_negative': True,
             'threshold_matched_to_fixed5kb_outside_alert_rate': True,
             'model_selection_and_fresh_test_evaluation': 'pending review of development OOF'}
    (out / 'dev_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'DEV_OOF_COMPLETE').touch()
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
