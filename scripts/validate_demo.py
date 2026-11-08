import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();src=a.inputs.parent
h=pd.read_csv(a.out/'GENE_CLASSIFICATION.tsv',sep='\t').set_index('protein_id')
expected=pd.read_csv(src/'EXPECTED_HIERARCHY.tsv',sep='\t').set_index('protein_id').loc[h.index]
cols=['gate_score','p_ARG','p_MRG','p_MGE','p_OTHER','conditional_p_ARG','conditional_p_MRG','conditional_p_MGE']
assert h.final_class.tolist()==expected.final_class.tolist()
delta=float(np.max(np.abs(h[cols].to_numpy()-expected[cols].to_numpy())))
assert delta<1e-7,delta
keys=['host_protein_id','length_kb','placement'];x=pd.read_parquet(a.out/'FEATURES248.parquet').set_index(keys);e=pd.read_parquet(src/'EXPECTED_FEATURES248.parquet').set_index(keys).loc[x.index]
features=json.loads((src/'FEATURE_COLUMNS248.json').read_text());assert len(features)==248
assert np.allclose(x[features],e[features],rtol=2e-4,atol=1e-5,equal_nan=True)
for klass,n in [('ARG',15),('MRG',23),('MGE',7)]:
    f=pd.read_csv(a.out/f'03_fine/{klass}_8M_FINE.tsv',sep='\t');assert set(f.sequence_id)==set(h.index[h.final_class==klass]);assert f.n_models.eq(15).all()
    assert f.probabilities.str.split(';').map(len).eq(n).all()
m=pd.read_parquet(a.out/'04_mobility/MOBILITY_UNIFIED248.parquet');assert len(m)==8 and not m.duplicated(keys).any()
reference=Path(__file__).resolve().parents[1]/'results/demo'
if a.out.resolve()!=reference.resolve():
    baseline=pd.read_parquet(reference/'04_mobility/MOBILITY_UNIFIED248.parquet').set_index(keys)
    current=m.set_index(keys).loc[baseline.index]
    assert np.allclose(current.unified248_score,baseline.unified248_score,rtol=1e-7,atol=1e-8)
    for klass in ['ARG','MRG','MGE']:
        current=pd.read_csv(a.out/f'03_fine/{klass}_8M_FINE.tsv',sep='\t').set_index('sequence_id')
        baseline=pd.read_csv(reference/f'03_fine/{klass}_8M_FINE.tsv',sep='\t').set_index('sequence_id').loc[current.index]
        assert current.predicted_labels.fillna('').tolist()==baseline.predicted_labels.fillna('').tolist()
        def probabilities(values):return np.array([[float(x.rsplit('=',1)[1]) for x in s.split(';')] for s in values])
        assert np.allclose(probabilities(current.probabilities),probabilities(baseline.probabilities),rtol=2e-5,atol=2e-6)
hits=pd.read_csv(a.out/'05_alignment/PREDICTED_TARGET_REFERENCE_HITS.tsv',sep='\t')
assert len(hits)==50 and hits.protein_id.nunique()==8
assert set(hits.protein_id)==set(h.index[h.final_class.isin(['ARG','MRG','MGE'])])
report=(a.out/'report.html').read_text()
assert '20 kb' not in report and '<html lang="en">' in report
j=pd.read_csv(a.out/'ALL_STAGES.tsv',sep='\t');assert j[j.final_class.isin(['MGE','OTHER'])].unified248_score.isna().all()
assert j[j.final_class.isin(['ARG','MRG'])].fine_n_models.eq(15).all()
for stem in ['00_workflow','01_hierarchy','02_fine_functions','03_mobility_context','04_genomic_neighborhoods']:
    for ext in ['png','pdf','svg']:assert (a.out/f'figures/{stem}.{ext}').stat().st_size>1000
audit={'state':'PASS','proteins':len(h),'hierarchy_max_abs_difference':delta,'feature_reconstruction':'all 248 locked model features match archived visible-context features','nonmodel_metadata_note':'raw reconstruction does not fill all redundant host/fragment coordinate columns; joined protein coordinates and example context tables provide positions','fine_models_per_task':15,'mobility_contexts':len(m),'joined_rows':len(j),'figure_sets':5,'scope':'software reproduction on development examples; not independent accuracy'}
(a.out/'VALIDATION.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
