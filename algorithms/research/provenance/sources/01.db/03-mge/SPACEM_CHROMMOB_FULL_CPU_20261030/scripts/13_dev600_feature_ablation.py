from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE = Path('/ifs1/User/yanyueyang/yyy/gene-finder/01.db/03-mge')
DEV = BASE / 'SPACEM_CHROMMOB_V2_STAGE3_DEV600_20261030'
LOCAL = ['visible_local_bp', 'noncargo_local_cds', 'noncargo_local_codons',
         'local_gc', 'local_gc_change_max_1kb']
GC = ['delta_gc_host', 'abs_delta_gc_host', 'delta_gc3_host', 'abs_delta_gc3_host']
CODON = ['rscu_jsd_host', 'delta_enc_host', 'delta_syn_entropy_host']
TETRA = ['tetra_jsd_host']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--threads', type=int, default=24)
    args = ap.parse_args()
    out = args.root / 'dev600_feature_ablation'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'ABLATION_COMPLETE').exists():
        print((out / 'ablation_metrics.tsv').read_text(), flush=True)
        return
    spec = importlib.util.spec_from_file_location('oof_utils', args.root / 'scripts/04_dev600_oof.py')
    util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(util)
    cargo = pd.read_parquet(DEV / 'training_no_pfam_local_soft/dev_host_random_cargo_local_soft.parquet')
    m0 = pd.read_parquet(DEV / 'stage3_masked/cargo_m0_scores.parquet')
    delta = pd.read_parquet(args.root / 'dev600_delta_unmasked/DEV600_DELTA_FEATURES.parquet')
    prior = pd.read_parquet(args.root / 'dev600_oof_unmasked/DEV600_OOF_PREDICTIONS.parquet',
                             columns=['cargo_protein_id', 'M0R_retrained'])
    data = cargo.merge(m0[['cargo_protein_id', 'p_M0_max_union', 'fixed5kb']],
                       on='cargo_protein_id', validate='one_to_one')
    data = data.merge(delta.drop(columns=['accession']), on='cargo_protein_id', validate='one_to_one')
    data = data.merge(prior, on='cargo_protein_id', validate='one_to_one').reset_index(drop=True)
    if len(data) != 13754:
        raise RuntimeError('ablation input row count mismatch')
    y = data.y.to_numpy(np.int8)
    budget = float(data.loc[data.y.eq(0), 'fixed5kb'].mean())
    base = util.NEUTRAL + ['p_M0_max_union']
    variants = {'M0R_base': data.M0R_retrained.to_numpy(np.float64),
                'local_only': util.oof(data, base + LOCAL, 'local_only', out, args.threads),
                'local_plus_delta_GC': util.oof(data, base + LOCAL + GC, 'local_plus_delta_GC', out, args.threads),
                'local_plus_codon': util.oof(data, base + LOCAL + CODON, 'local_plus_codon', out, args.threads),
                'local_plus_tetra': util.oof(data, base + LOCAL + TETRA, 'local_plus_tetra', out, args.threads),
                'local_plus_all_host_delta': util.oof(data, base + LOCAL + GC + CODON + TETRA,
                                                       'local_plus_all_host_delta', out, args.threads)}
    metrics = []
    for name, score in variants.items():
        threshold = float(np.quantile(score[y == 0], 1 - budget, method='higher'))
        values = util.metrics(y, score, threshold)
        metrics.append({'variant': name, **values})
    result = pd.DataFrame(metrics)
    result.to_csv(out / 'ablation_metrics.tsv', sep='\t', index=False)
    predictions = data[['cargo_protein_id', 'parent_id', 'dev_fold', 'y', 'fixed5kb']].copy()
    for name, score in variants.items():
        predictions[name] = score
    predictions.to_parquet(out / 'ablation_oof_predictions.parquet', index=False, compression='zstd')
    groups = [g.index.to_numpy(np.int32) for _, g in data.groupby('parent_id', sort=False)]
    rng = np.random.default_rng(20261030)
    pairs = []
    reference = variants['local_only']
    for name, score in variants.items():
        if name in ('M0R_base', 'local_only'):
            continue
        difference = float(average_precision_score(y, score) - average_precision_score(y, reference))
        draws = []
        for _ in range(1000):
            ids = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            yy = y[ids]
            if yy.min() == yy.max():
                continue
            draws.append(average_precision_score(yy, score[ids]) - average_precision_score(yy, reference[ids]))
        pairs.append({'variant': name, 'baseline': 'local_only', 'AUPRC_delta': difference,
                      'CI95_low': float(np.quantile(draws, .025)),
                      'CI95_high': float(np.quantile(draws, .975))})
    pd.DataFrame(pairs).to_csv(out / 'paired_vs_local.tsv', sep='\t', index=False)
    audit = {'cohort': 'dev600 grouped five-fold OOF, exploratory feature ablations',
             'reference_outside_budget_from_fixed5kb': budget,
             'no_new_test_scores_read': True, 'background_uses_Botelho_boundaries': False,
             'feature_selection_warning': 'do not claim independent gains from development ablation alone'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (out / 'ABLATION_COMPLETE').touch()
    print(result.to_string(index=False), flush=True)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
