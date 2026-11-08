from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20261101
MODEL_SEED = 20261031
MISSING_GAP = 101001.0
SCALES_BP = (1000, 2000, 5000, 10000)
VARIANT = 'near_mge_monotone_full_replicon'
DROP_FEATURES = ('nearest_mge_signed_midpoint_bp', 'nearest_mge_oriented_midpoint_bp')
PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=15,
              max_depth=5, min_data_in_leaf=40, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=5.,
              verbosity=-1, device_type='cpu', monotone_constraints_method='basic')


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def digest_strings(values) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode()); h.update(b'\n')
    return h.hexdigest()


def augment(frame: pd.DataFrame, original_features: list[str]):
    columns = [c for c in original_features if c not in DROP_FEATURES]
    x = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    gap = pd.to_numeric(x.nearest_mge_edge_gap_bp, errors='coerce').to_numpy(float)

    finite = np.isfinite(gap)
    if np.any(gap[finite] < 0) or np.any(gap[finite] > 100000):
        raise RuntimeError('nearest-MGE distance outside upstream 0..100kb contract')
    gap = np.where(finite, gap, MISSING_GAP)
    x['nearest_mge_edge_gap_bp'] = gap
    p = np.nan_to_num(x.nearest_mge_p_MGE.to_numpy(float), nan=0.)
    if np.any((p < 0) | (p > 1)):
        raise RuntimeError('nearest MGE probabilities outside [0,1]')
    for scale in SCALES_BP:
        proximity = np.exp(-gap/scale)


        proximity[gap == MISSING_GAP] = 0.
        x[f'near_mge_proximity_{scale//1000}kb'] = proximity
        x[f'near_mge_score_proximity_{scale//1000}kb'] = p * proximity
    x['near_mge_present_5kb'] = (gap <= 5000).astype(np.float32)
    columns = list(x.columns)
    constraints = {}
    for c in columns:
        if c == 'nearest_mge_edge_gap_bp':
            constraints[c] = -1
        elif c.startswith(('mge_count_', 'mge_score_sum_', 'mge_exp_sum_lambda_',
                           'near_mge_proximity_', 'near_mge_score_proximity_')):
            constraints[c] = 1
        elif c in ('near_mge_present_5kb', 'nearest_mge_p_MGE', 'mge_pair_count_within_5kb'):
            constraints[c] = 1
        else:
            constraints[c] = 0
    return x.to_numpy(np.float32), columns, constraints


def delta_ci(y, call, comparator, group, target: int, reps=2000):
    part = pd.DataFrame({'group': group[y == target],
                         'a': call[y == target].astype(float),
                         'b': comparator[y == target].astype(float)})
    agg = part.groupby('group').agg(n=('a', 'size'), a=('a', 'sum'), b=('b', 'sum'))
    vals = agg[['n', 'a', 'b']].to_numpy(float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(SEED + target)
    samples = rng.integers(0, len(vals), (reps, len(vals)))
    totals = vals[samples].sum(axis=1)
    diffs = (totals[:, 1]-totals[:, 2])/totals[:, 0]
    return [float(v) for v in np.quantile(diffs, [.025, .975])]


def monotone_audit(model, x, columns, constraints, seed, threads):

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), min(len(x), 2048), replace=False)
    sample = x[idx].copy()
    tol = 1e-10
    pred_threads = min(8, threads)
    per_column = []
    for c, direction in constraints.items():
        if not direction:
            continue
        j = columns.index(c)
        finite = x[:, j][np.isfinite(x[:, j])]
        if len(finite) == 0:
            per_column.append({'feature': c, 'direction': direction, 'grid': [],
                               'rows': len(sample), 'violations': 0})
            continue
        grid = np.unique(np.quantile(finite, [0, .25, .5, .75, 1]))
        last = None; violations = 0
        for value in grid:
            z = sample.copy(); z[:, j] = value
            pred = model.predict(z, num_threads=pred_threads)
            if last is not None:
                violations += int(np.sum(direction*(pred-last) < -tol))
            last = pred
        per_column.append({'feature': c, 'direction': direction,
                           'grid': [float(v) for v in grid], 'rows': len(sample),
                           'violations': violations})
    jgap = columns.index('nearest_mge_edge_gap_bp')
    jp = columns.index('nearest_mge_p_MGE')
    p = np.nan_to_num(sample[:, jp], nan=0.)
    grid = (0., 500., 1000., 2000., 5000., 5001., 10000., 20000., 50000., 100000., MISSING_GAP)
    last = None; coherent_violations = 0
    for gap in grid:
        z = sample.copy(); z[:, jgap] = gap
        for scale in SCALES_BP:
            proximity = 0. if gap == MISSING_GAP else float(np.exp(-gap/scale))
            z[:, columns.index(f'near_mge_proximity_{scale//1000}kb')] = proximity
            z[:, columns.index(f'near_mge_score_proximity_{scale//1000}kb')] = p*proximity
        z[:, columns.index('near_mge_present_5kb')] = float(gap <= 5000)
        pred = model.predict(z, num_threads=pred_threads)
        if last is not None:
            coherent_violations += int(np.sum(pred-last > tol))
        last = pred
    total = coherent_violations + sum(r['violations'] for r in per_column)
    return {'sample_rows': len(sample), 'tolerance': tol,
            'finite_grid_audit_not_exhaustive': True,
            'conditional_on_other_features_fixed': True,
            'coherent_nearest_gap_grid_bp': list(grid),
            'coherent_nearest_gap_violations': coherent_violations,
            'individual_feature_audits': per_column, 'total_violations': total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--threads', type=int, default=64)
    args = ap.parse_args()
    root = args.root
    out = root / 'near_mge_monotone_dev_20261101'
    out.mkdir(exist_ok=True)
    old = root / 'strict_reference_localization_dev'
    data = pd.read_parquet(root/'strict_v3hn_spatial/CARGO_EVIDENCE_FULL.parquet',
                           filters=[('partition', '==', 'development')])
    data = data.loc[data.reference_position.isin(('inside', 'outside'))].copy()
    if not data.partition.eq('development').all() or data.host_protein_id.duplicated().any():
        raise RuntimeError('invalid development-only input contract')
    original_features = json.loads((root/'strict_v3hn_spatial/feature_columns.json').read_text())
    prior = pd.read_parquet(old/'DEV_OOF_PREDICTIONS.parquet')
    prior = prior.loc[prior.variant.eq('all_evidence')].copy()
    if prior.host_protein_id.duplicated().any() or set(prior.host_protein_id) != set(data.host_protein_id):
        raise RuntimeError('frozen OOF and current development cohort differ')
    if not set(prior.fold) == set(range(5)):
        raise RuntimeError('expected original five development folds')
    prior = prior.set_index('host_protein_id').loc[data.host_protein_id].reset_index()
    for c in ('accession', 'cargo_class', 'reference_position'):
        if not np.array_equal(data[c].to_numpy(), prior[c].to_numpy()):
            raise RuntimeError(f'original development OOF metadata mismatch: {c}')
    folds = prior.fold.to_numpy(np.int8)
    group = prior.connected_group.to_numpy(str)
    if pd.DataFrame({'group':group, 'fold':folds}).groupby('group').fold.nunique().max() != 1:
        raise RuntimeError('original connected-group fold contract violated')
    y = data.reference_position.eq('inside').to_numpy(np.int8)
    baseline = data.baseline_predicted_mge_within_5kb.to_numpy(bool)
    if not np.array_equal(baseline, prior.baseline_call.to_numpy(bool)):
        raise RuntimeError('5kb baseline changed since original OOF')
    old_metrics = pd.read_csv(old/'DEV_OOF_METRICS.tsv', sep='\t').set_index('variant').loc['all_evidence']
    rate = float(baseline[y == 0].mean())
    if len(data) != int(old_metrics.development_rows) or not np.isclose(rate, old_metrics.baseline_5kb_outside_alert_rate):
        raise RuntimeError('original development aggregate contract changed')
    oldscore = prior.score.to_numpy(float)
    oldcall = oldscore >= float(old_metrics.model_threshold_matched_outside_rate)
    x, columns, constraints = augment(data, original_features)
    protocol = {'variant':VARIANT, 'created_date':'2026-11-01',
                'input_partition':'development only', 'development_rows':len(data),
                'development_cargo_ids_digest':digest_strings(data.host_protein_id),
                'original_oof_folds_digest':digest_strings(folds),
                'original_oof_groups_digest':digest_strings(group),
                'base_features':len(original_features), 'features':columns,
                'removed_features':list(DROP_FEATURES), 'constraints':constraints,
                'parameters':PARAMS, 'rounds':300, 'model_seed':MODEL_SEED,
                'audit_seed':SEED,
                'script_digest':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'baseline_5kb_outside_alert_rate':rate,
                'no_mge_gap_sentinel_bp':MISSING_GAP,
                'nearest_distance_weight_scales_bp':list(SCALES_BP),
                'explicit_5kb_indicator_reinforces_baseline_geometry':True,
                'no_old_fresh_test_read_or_evaluation':True,
                'outside_reference_is_unverified_control':True,
                'selection':'candidate only; parent selects across prespecified development candidates',
                'direction_claim':'MGE distance influence only with other predictors fixed',
                'audit_pass_required_for_final_candidate_lock':True}
    protocol_path = out/'PROTOCOL.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('existing candidate protocol differs; use a new output directory')
    else:
        atomic_json(protocol_path, protocol)
    if (out/'DEV_MONOTONE_COMPLETE').exists():
        print((out/'DEV_OOF_METRICS.json').read_text(), flush=True)
        return
    weight = 1/np.sqrt(pd.Series(group).groupby(pd.Series(group)).transform('size').to_numpy(float))
    weight = (weight/weight.mean()).astype(np.float32)
    weight[y == 0] *= .5
    constraint_list = [constraints[c] for c in columns]
    score = np.full(len(data), np.nan)
    audits = []
    for fold in range(5):
        fit, val = folds != fold, folds == fold
        model_path = out/f'{VARIANT}_fold{fold}.txt'
        cache_path = out/f'fold{fold}_score.npz'
        if model_path.exists() and cache_path.exists():
            cached = np.load(cache_path)
            if not np.array_equal(cached['index'], np.flatnonzero(val)):
                raise RuntimeError('resume score validation indices differ')
            score[val] = cached['score']
            model = lgb.Booster(model_file=str(model_path))
        else:
            params = {**PARAMS, 'seed':MODEL_SEED+fold, 'num_threads':args.threads,
                      'monotone_constraints':constraint_list}
            train = lgb.Dataset(x[fit], label=y[fit], weight=weight[fit], feature_name=columns)
            model = lgb.train(params, train, num_boost_round=300)
            score[val] = model.predict(x[val], num_threads=args.threads)
            model.save_model(str(model_path))
            np.savez_compressed(cache_path, index=np.flatnonzero(val), score=score[val])
        if model.feature_name() != columns:
            raise RuntimeError('resumed model feature contract differs')
        audit = monotone_audit(model, x[val], columns, constraints, SEED+fold, args.threads)
        audits.append({'fold':fold, **audit})
        atomic_json(out/f'fold{fold}_monotone_audit.json', audits[-1])
        if audit['total_violations']:
            raise RuntimeError('direction audit failed; no candidate lock written')
        print(json.dumps({'fold':fold,'validation_rows':int(val.sum()),
                          'monotonic_violations':audit['total_violations']}), flush=True)
    if not np.isfinite(score).all():
        raise RuntimeError('incomplete development OOF')
    threshold = float(np.quantile(score[y == 0], 1-rate, method='higher'))
    call = score >= threshold
    metrics = {'variant':VARIANT, 'features':len(columns), 'development_rows':len(data),
               'development_connected_groups':len(set(group)),
               'reference_inside_ICE_IME':int(y.sum()),
               'reference_outside_unverified_controls':int((y == 0).sum()),
               'model_threshold_matched_outside_rate':threshold,
               'baseline_5kb_inside_recall':float(baseline[y == 1].mean()),
               'baseline_5kb_outside_alert_rate':rate,
               'model_inside_recall_matched_outside_rate':float(call[y == 1].mean()),
               'model_outside_alert_rate':float(call[y == 0].mean()),
               'old_model_inside_recall':float(oldcall[y == 1].mean()),
               'old_model_outside_alert_rate':float(oldcall[y == 0].mean()),
               'recall_delta_vs_5kb':float(call[y == 1].mean()-baseline[y == 1].mean()),
               'recall_delta_vs_old':float(call[y == 1].mean()-oldcall[y == 1].mean()),
               'recall_delta_vs_5kb_group_ci':delta_ci(y,call,baseline,group,1),
               'recall_delta_vs_old_group_ci':delta_ci(y,call,oldcall,group,1),
               'outside_alert_delta_vs_5kb_group_ci':delta_ci(y,call,baseline,group,0),
               'auroc_reference_localization':float(roc_auc_score(y,score)),
               'auprc_reference_localization':float(average_precision_score(y,score)),
               'old_model_auroc':float(roc_auc_score(y,oldscore)),
               'old_model_auprc':float(average_precision_score(y,oldscore)),
               'monotonic_audit_total_violations':sum(a['total_violations'] for a in audits),
               'outside_is_not_biological_negative':True,
               'old_fresh_test_not_read':True, 'external_evaluation_pending':True}
    for cargo_class in ('ARG','MRG'):
        ix = (data.cargo_class == cargo_class).to_numpy() & (y == 1)
        metrics[f'{cargo_class}_inside_n'] = int(ix.sum())
        for name,pred in [('model',call),('old_model',oldcall),('baseline',baseline)]:
            metrics[f'{cargo_class}_{name}_inside_recall'] = float(pred[ix].mean()) if ix.any() else None
    for element_type in ('ICE','IME'):
        ix = (data.inside_element_type == element_type).to_numpy() & (y == 1)
        metrics[f'{element_type}_inside_n'] = int(ix.sum())
        for name,pred in [('model',call),('old_model',oldcall),('baseline',baseline)]:
            metrics[f'{element_type}_{name}_inside_recall'] = float(pred[ix].mean()) if ix.any() else None
    prediction = pd.DataFrame({'host_protein_id':data.host_protein_id.to_numpy(),
        'accession':data.accession.to_numpy(),'connected_group':group,'fold':folds,
        'cargo_class':data.cargo_class.to_numpy(), 'inside_element_type':data.inside_element_type.to_numpy(),
        'reference_position':data.reference_position.to_numpy(),'variant':VARIANT,
        'score':score,'model_call':call,'old_model_score':oldscore,
        'old_model_call':oldcall,'baseline_call':baseline})
    prediction.to_parquet(out/'DEV_OOF_PREDICTIONS.parquet',index=False)
    atomic_json(out/'DEV_OOF_METRICS.json', metrics)
    pd.DataFrame([metrics]).to_csv(out/'DEV_OOF_METRICS.tsv',sep='\t',index=False)
    final_params = {**PARAMS,'seed':MODEL_SEED,'num_threads':args.threads,
                    'monotone_constraints':constraint_list}
    final = lgb.train(final_params,lgb.Dataset(x,label=y,weight=weight,feature_name=columns),num_boost_round=300)
    final_audit = monotone_audit(final,x,columns,constraints,SEED+10,args.threads)
    atomic_json(out/'FINAL_MONOTONE_AUDIT.json',final_audit)
    if final_audit['total_violations']:
        raise RuntimeError('final-model audit failed; candidate lock not written')
    final_path = out/f'{VARIANT}_all_development.txt'
    final.save_model(str(final_path))
    pd.DataFrame({'feature':columns,'gain':final.feature_importance('gain'),
                  'splits':final.feature_importance('split'),
                  'monotone_direction':constraint_list}).sort_values('gain',ascending=False).to_csv(
                      out/'FEATURE_IMPORTANCE.tsv',sep='\t',index=False)
    atomic_json(out/'CANDIDATE_LOCK.json',{'candidate_variant':VARIANT,
        'candidate_model_path':str(final_path),'threshold_development_oof':threshold,
        'feature_columns':columns,'constraints':constraints,'parameters':final_params,
        'num_boost_round':300,'final_model_monotone_violations':0,
        'candidate_only_not_final_selected_across_variants':True,
        'old_fresh_test_not_read':True,'external_evaluation_pending':True})
    (out/'DEV_MONOTONE_COMPLETE').touch()
    print(json.dumps(metrics,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
