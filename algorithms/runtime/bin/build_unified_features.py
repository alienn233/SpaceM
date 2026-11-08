import argparse,json,runpy
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def build(base,local):
    old=json.loads((ROOT/'config/base229_feature_columns.json').read_text())
    union=runpy.run_path(str(ROOT/'lib/79_build_unified_visible_features.py'))['construct_unified']
    augment=runpy.run_path(str(ROOT/'lib/58_train_near_monotone_dev.py'))['augment']
    keys=['host_protein_id','length_kb','placement']
    for d in [base,local]:
        if not set(keys)<=set(d) or d.duplicated(keys).any():raise ValueError('Missing/duplicate feature keys')
    order=base[keys].merge(local,on=keys,how='left',validate='one_to_one',indicator=True)
    if not order._merge.eq('both').all():raise ValueError('local44 does not cover base229 keys')
    x,cols,_,_,_=union(base,order,old,augment)
    expected=json.loads((ROOT/'models/unified248/feature_columns.json').read_text())
    if cols!=expected:raise ValueError('Locked248 feature order changed')
    meta=base[[c for c in base if c not in old and c not in cols]].reset_index(drop=True)
    return pd.concat([meta,x],axis=1)
def main():
    p=argparse.ArgumentParser(description='Reproduce locked 248-feature union from aligned visible 229 and local44 tables.');p.add_argument('--base229',type=Path,required=True);p.add_argument('--local44',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Output already exists')
    result=build(pd.read_parquet(a.base229),pd.read_parquet(a.local44));a.out.parent.mkdir(parents=True,exist_ok=True);result.to_parquet(a.out,index=False,compression='zstd')
if __name__=='__main__':main()
