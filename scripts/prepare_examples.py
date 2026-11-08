import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();d=a.out;d.mkdir(parents=True,exist_ok=True)
cases=pd.read_csv(d/'BOTELHO_CASES.tsv',sep='\t');genes=pd.read_csv(d/'BOTELHO_CASE_GENES.tsv',sep='\t')
rows=[]
for c in cases.itertuples():
    rows += [(c.case_id,c.cargo_gene_id,'cargo'),(c.case_id,c.nearest_mge_id,'MGE_neighbor')]
    bg=genes[(genes.case_id==c.case_id)&(genes.v3hn_class=='OTHER')].copy()
    if len(bg):
        bg['distance']=abs((bg.start1+bg.end1)/2-(c.cargo_start1+c.cargo_end1)/2)
        rows.append((c.case_id,bg.sort_values('distance').gene_id.iloc[0],'background'))
selection=pd.DataFrame(rows,columns=['case_id','protein_id','example_role']).drop_duplicates('protein_id')
ids=selection.protein_id.tolist();wanted=set(ids);L=a.source;inp=L/'strict_v3hn_inputs'
h=pd.read_csv(L/'strict_v3hn_inference/02_hierarchy/HIERARCHY_PREDICTIONS.tsv',sep='\t')
h=h.set_index('protein_id',drop=False).loc[ids].reset_index(drop=True)
assert set(h.final_class)=={'ARG','MRG','MGE','OTHER'}
selection.to_csv(d/'SELECTION.tsv',sep='\t',index=False)
(d/'ids.txt').write_text('\n'.join(ids)+'\n')
h.to_csv(d/'EXPECTED_HIERARCHY.tsv',sep='\t',index=False)
h[['protein_id','sample_id','contig_id','start','end','strand']].to_parquet(d/'proteins.parquet',index=False)
allids=(inp/'ids.txt').read_text().splitlines();ix={v:i for i,v in enumerate(allids)};ix=[ix[v] for v in ids]
for src,out in [('ESM8_CANDIDATE_LOCUS.f16.npy','esm8.npy'),('ESM650_CANDIDATE_LOCUS.f16.npy','esm650.npy'),('PORTABLE_CONTEXT_FULL.f32.npy','portable.npy')]:
    np.save(d/out,np.load(inp/src,mmap_mode='r')[ix])
for src,out in [('PFAM_FEATURES_HOST.tsv','pfam.parquet'),('EGGNOG_FEATURES_HOST.tsv','eggnog.parquet')]:
    parts=[x[x.protein_id.isin(wanted)] for x in pd.read_csv(inp/src,sep='\t',chunksize=20000)]
    pd.concat(parts).to_parquet(d/out,index=False)
magfiles=sorted(set(inp.glob('*MAG*.parquet'))|set(inp.glob('*ABC*.parquet')))
assert len(magfiles)==1
mag=pd.read_parquet(magfiles[0]);mag[mag.protein_id.isin(wanted)].to_parquet(d/'mag.parquet',index=False)
seq={};pid=None
with (inp/'CANDIDATE_LOCUS.faa').open() as f:
    for line in f:
        if line.startswith('>'):
            pid=line[1:].split()[0]
            if pid in wanted:seq[pid]=[]
        elif pid in wanted:seq[pid].append(line.strip())
(d/'proteins.faa').write_text(''.join('>'+pid+'\n'+''.join(seq[pid])+'\n' for pid in ids))
keys=['host_protein_id','length_kb','placement']
dev=pd.read_parquet(L/'unified_visible_dev_20261102/DEVELOPMENT_FEATURES.parquet')
sel=[]
for c in cases.itertuples():
    part=dev[(dev.host_protein_id==c.cargo_gene_id)&((dev.length_kb==0)|((dev.length_kb==20)&np.isclose(dev.placement,c.placement)))]
    assert len(part)==2,(c.cargo_gene_id,len(part));sel.append(part)
expected=pd.concat(sel,ignore_index=True)
expected.to_parquet(d/'EXPECTED_FEATURES248.parquet',index=False)
full=pd.read_parquet(L/'strict_v3hn_spatial/CARGO_EVIDENCE_FULL.parquet')
full=full[full.host_protein_id.isin(cases.cargo_gene_id)].copy();full['length_kb']=0;full['placement']=.5
frags=[]
for f in sorted((L/'fragment_full_near_monotone_dev_20261101').glob('batch*/FRAGMENT_FULL_CONTEXT.parquet')):
    part=pd.read_parquet(f);part=part.merge(expected[keys],on=keys,validate='one_to_one')
    if len(part):frags.append(part)
    if sum(map(len,frags))==4:break
base=pd.concat([full,*frags],ignore_index=True);assert len(base)==8
base.to_parquet(d/'base229.parquet',index=False)
local=pd.read_parquet(L/'fragment_near_monotone_dev_20261101/DEVELOPMENT_FEATURES.parquet')
expected[keys].merge(local,on=keys,validate='one_to_one').to_parquet(d/'local44.parquet',index=False)
oof=pd.read_parquet(L/'unified_visible_dev_20261102/DEV_OOF_PREDICTIONS.parquet')
oof=expected[keys].merge(oof,on=keys,validate='one_to_one')
oof[[c for c in oof if 'sha256' not in c]].to_csv(d/'ARCHIVED_MOBILITY_OOF.tsv',sep='\t',index=False)
config={'ids':'ids.txt','protein_table':'proteins.parquet','protein_fasta':'proteins.faa','esm8':'esm8.npy','esm650':'esm650.npy','portable_context':'portable.npy','pfam':'pfam.parquet','eggnog':'eggnog.parquet','mag':'mag.parquet','mobility_base229':'base229.parquet','mobility_local44':'local44.parquet','case_table':'BOTELHO_CASES.tsv','case_genes':'BOTELHO_CASE_GENES.tsv','selection':'SELECTION.tsv','archived_mobility_oof':'ARCHIVED_MOBILITY_OOF.tsv','prediction_context':'development_refit_not_independent_evaluation'}
(d/'inputs.json').write_text(json.dumps(config,indent=2)+'\n')
(d/'PROVENANCE.json').write_text(json.dumps({'selection':'four previously analysed internal explanatory cases; not random and not independent evaluation','proteins':len(h),'class_counts':h.final_class.value_counts().to_dict(),'mobility_rows':8,'context':'full host and 20 kb; frozen full-host upstream annotations reused','references_used_for_features':False},indent=2)+'\n')
print('PREPARED',len(h),'proteins and',len(base),'mobility contexts',flush=True)
