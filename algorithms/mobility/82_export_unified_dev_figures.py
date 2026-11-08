from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def downsample(x, y, threshold, maximum=5000):
    indices = np.unique(np.rint(np.linspace(0, len(x)-1, min(maximum,len(x)))).astype(int))
    return pd.DataFrame({'x':x[indices], 'y':y[indices],
                         'threshold':threshold[indices]})


def as_json(value):
    if isinstance(value, dict):
        return {str(k):as_json(v) for k,v in value.items()}
    if isinstance(value, list):
        return [as_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-dir',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    root=args.model_dir
    if not (root/'MODEL_COMPLETE').exists():
        raise RuntimeError('unified model is not complete; keep stable training running')
    lock=json.loads((root/'CANDIDATE_LOCK.json').read_text())
    if not lock.get('one_model_and_one_cutoff_across_all_lengths') or not lock.get('previous_fresh_test_and_external_not_read'):
        raise RuntimeError('unified development-only lock contract differs')
    if len(lock['features']) != 248:
        raise RuntimeError('expected unified248 candidate')
    out=args.out
    data=out/'data'
    data.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(root/'DEV_OOF_PREDICTIONS.parquet',columns=[
        'host_protein_id','connected_group','fold','length_kb','placement','y',
        'baseline_call','score','cohort_weight','per_length_eval_weight'])
    if not np.isfinite(d.score).all() or d.groupby('connected_group').fold.nunique().max()!=1:
        raise RuntimeError('incomplete or ungrouped development OOF predictions')
    cutoff=float(lock['threshold'])
    curves=[]
    points=[]
    metrics=[]
    for length in (0,20):
        g=d.loc[d.length_kb.eq(length)]
        y=g.y.to_numpy(bool)
        score=g.score.to_numpy(float)
        w=g.per_length_eval_weight.to_numpy(float)
        scope='FULL_REPLICON' if length==0 else '20kb'
        fpr,tpr,threshold=roc_curve(y,score,sample_weight=w,drop_intermediate=True)
        roc=downsample(fpr,tpr,threshold)
        roc['curve']='ROC';roc['scope']=scope;curves.append(roc)
        precision,recall,threshold=precision_recall_curve(y,score,sample_weight=w)
        pr=downsample(recall,precision,np.r_[threshold,np.nan])
        pr['curve']='PR';pr['scope']=scope;curves.append(pr)
        for method,pred in [('Unified248_locked',score>=cutoff),
                            ('Fixed_5kb',g.baseline_call.to_numpy(bool))]:
            inside=float(np.average(pred[y],weights=w[y]))
            outside=float(np.average(pred[~y],weights=w[~y]))
            p=float(np.average(y[pred],weights=w[pred])) if pred.any() else np.nan
            points.extend([{'scope':scope,'curve':'ROC','method':method,'x':outside,'y':inside},
                           {'scope':scope,'curve':'PR','method':method,'x':inside,'y':p}])
        metrics.append({'scope':scope,'windows':len(g),'unique_cargo':g.host_protein_id.nunique(),
                        'inside_windows':int(y.sum()),'outside_windows':int((~y).sum()),
                        'inside_cargo':g.loc[y,'host_protein_id'].nunique(),
                        'outside_cargo':g.loc[~y,'host_protein_id'].nunique(),
                        'auroc':roc_auc_score(y,score,sample_weight=w),
                        'auprc':average_precision_score(y,score,sample_weight=w),
                        'reference_inside_prevalence':np.average(y,weights=w),
                        'threshold':cutoff})
    pd.concat(curves,ignore_index=True).replace([np.inf,-np.inf],np.nan).to_csv(data/'ROC_PR_CURVES.tsv',sep='\t',index=False)
    pd.DataFrame(points).to_csv(data/'ROC_PR_POINTS.tsv',sep='\t',index=False)
    pd.DataFrame(metrics).to_csv(data/'ROC_PR_METRICS.tsv',sep='\t',index=False)
    for name in ('DEV_BY_SCOPE.tsv','FEATURE_IMPORTANCE.tsv','COHORT_BY_LENGTH.tsv'):
        if (root/name).exists():
            shutil.copy2(root/name,data/name)
    importance=pd.read_csv(data/'FEATURE_IMPORTANCE.tsv',sep='\t')
    importance['gain_percent']=100*importance.gain/importance.gain.sum()
    importance.to_csv(data/'FEATURE_IMPORTANCE.tsv',sep='\t',index=False)
    for name in ('CANDIDATE_LOCK.json','DEV_AUDIT.json','WEIGHT_AUDIT.json','TRAIN_PROTOCOL.json'):
        shutil.copy2(root/name,data/name)
    scopes=pd.read_csv(data/'DEV_BY_SCOPE.tsv',sep='\t').set_index('scope')
    def pct(value):return f'{100*value:.2f}%'
    def sentence(scope):
        row=scopes.loc[scope]
        label='Full replicon' if scope=='FULL_REPLICON' else '20 kb fragment'
        return (f'{label}: inside-reference recall was {pct(row.model_inside_recall)} for SpaceM '
                f'and {pct(row.baseline_inside_recall)} for the 5 kb rule; outside-reference alert rates '
                f'were {pct(row.model_outside_alert)} and {pct(row.baseline_outside_alert)}, respectively.')
    explanations=[
      {'figure':'U01','title':'ROC and precision-recall curves for full replicons and 20 kb fragments',
       'paragraphs':[
        'Both contexts use the same 248-feature specification, grouped five-fold design and fixed threshold. Each validation fold is predicted by a model fitted to the other four folds.',
        'ROC axes show the outside-reference alert rate and inside-reference recall. Precision-recall axes show recall and the fraction of predicted-positive loci within reference ICE/IME intervals. Blue curves represent model scores, blue circles the fixed threshold, and orange diamonds the 5 kb rule.',
        sentence('FULL_REPLICON'),sentence('20kb'),
        'Outside-reference 20 kb backgrounds were sampled at 1/16 and receive inverse-sampling weight 16. Retained placements of each gene receive equal weights. AUROC and AP use all scores; displayed curves are thinned to at most 5,000 points.',
        'Results use development out-of-fold predictions and a threshold selected on those predictions. Outside-reference loci constitute unverified background.']},
      {'figure':'U02','title':'Performance across visible context lengths',
       'paragraphs':[
        'The horizontal axis shows 2, 5, 10, 20, 50 and 100 kb fragments followed by full replicons. SpaceM is blue and the 5 kb rule is orange. The model specification and threshold are shared across lengths.',
        'The panels show inside-reference recall and outside-reference alert rate, respectively.',
        'Threshold selection matches the baseline alert budget across the predefined mixture of full and fragment contexts; individual lengths can have different alert rates.',
        'Evaluation includes completely visible ARG/MRG loci and MGE CDS. Left, central and right placements of one gene are treated as repeated contexts. Eligible gene counts can vary with fragment length.',
        'Features are calculated within the visible context at each length. These are the six fragment lengths included in the development design.']},
      {'figure':'U03','title':'Leading predictive features in the unified model',
       'paragraphs':[
        'The plot shows the 15 features with the largest split gain in the model fitted to the complete development set. Percentages use total gain across all 248 features.',
        'Gain summarizes improvement in the training objective. Correlated predictors can share importance.',
        'The feature set combines DNA composition, neighborhood organization and function, oriT evidence, and multiscale cargo-MGE spatial information. Context-specific rows retain the same column order.',
        'Protein classes and scores reuse frozen V3HN annotations from the full host. Context features are recalculated over the visible interval.']},
    ]
    (out/'FIGURE_EXPLANATIONS.json').write_text(json.dumps(explanations,ensure_ascii=False,indent=2)+'\n')
    audit={'model':'UNIFIED248','only_development_OOF':True,'external_data_read':False,
           'one_model_specification_all_scopes':True,'one_cutoff':cutoff,
           'ROC_PR_scope_weights':'per_length_eval_weight; placement-equal; outsideIPW16',
           'curve_display_max_points':5000,'curve_areas_computed_before_display_thinning':True,
           'rows_read':len(d),'feature_count':248,'models_fitted':0,'GPU_used':False,
           'metrics':metrics,'expected_figures':['U01','U02','U03']}
    (data/'EXPORT_AUDIT.json').write_text(json.dumps(as_json(audit),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(as_json(audit),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
