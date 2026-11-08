import argparse,json,os,re,shutil,sys,subprocess
from pathlib import Path
OLD='/ifs1/User/yanyueyang/yyy/gene-finder'
def main():
    ap=argparse.ArgumentParser(description='Create a local source-layout view and relocate archived research script paths.')
    ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--score-python',default=sys.executable)
    ap.add_argument('--fine-python',required=True)
    a=ap.parse_args();package=a.package.resolve();out=a.out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty reproduction folder')
    out.mkdir(parents=True,exist_ok=True)
    mapped={}
    for line in (package/'SOURCE_MAP.jsonl').read_text().splitlines():
        item=json.loads(line);dest=package/item['destination']
        if not dest.is_file():continue
        for key in ['source','source_realpath']:
            src=item.get(key,'')
            if src.startswith(OLD+'/'):mapped[src.removeprefix(OLD+'/')]=dest

    for p in (package/'training/fine/artifacts').glob('*/*.py'):
        mapped['model_vault/MULTICLASS_20260904/artifacts/'+str(p.relative_to(package/'training/fine/artifacts'))]=p
    unresolved=set();rewritten=[]
    pinned={'diamond':('version','2.2.5'),'mmseqs':('version','18.8cc5c'),'blastn':('-version','2.17.0'),'cmsearch':('-h','1.1.5'),'prodigal':('-v','2.6.3')}
    tool_cache={}
    env_re=re.compile(r'/ifs1/User/yanyueyang/(?:\.conda/envs/[^/]+|anaconda)/bin/([A-Za-z0-9_.+-]+)')
    def tool(match):
        name=match.group(1)
        if name in ['python','python3']:
            return a.fine_python if 'protein_feature_core' in match.group(0) else a.score_python
        if match.group(0) in tool_cache:return tool_cache[match.group(0)]
        for path in dict.fromkeys([match.group(0),shutil.which(name)]):
            if not path or not Path(path).is_file():continue
            if name in pinned:
                flag,version=pinned[name]
                try:
                    p=subprocess.run([path,flag],capture_output=True,text=True,timeout=20)
                    if version not in p.stdout+p.stderr:continue
                except (OSError,subprocess.TimeoutExpired):continue
            elif path!=match.group(0):continue
            tool_cache[match.group(0)]=path;return path
        unresolved.add(match.group(0));return match.group(0)
    for relative,src in sorted(mapped.items()):
        dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix in ['.py','.sh','.json'] and src.stat().st_size<3000000:
            try:text=src.read_text()
            except UnicodeError:os.symlink(os.path.relpath(src,dest.parent),dest);continue
            new=text.replace(OLD,str(out)).replace('/ifs1/User/yanyueyang/database/eggnog',str(package/'resources/eggnog'))
            new=new.replace(str(out/'01.db/pfam_v1_1'),str(package/'resources/pfam'))
            new=new.replace(str(out/'01.db/03-mge/mobility_dna/db/orit'),str(package/'resources/orit'))
            new=env_re.sub(tool,new);dest.write_text(new);dest.chmod(src.stat().st_mode & 0o777)
            if new!=text:rewritten.append(relative)
        else:os.symlink(os.path.relpath(src,dest.parent),dest)
    audit={'files_mapped':len(mapped),'scripts_or_configs_relocated':rewritten,'unresolved_external_tools':sorted(unresolved),
           'resources_extracted':(package/'resources/eggnog/eggnog.db').exists(),'training_started':False,
           'note':'Paths are rewritten only in new copies; feature generation and fitting remain explicit commands.'}
    (out/'REPRODUCTION_VIEW.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='scripts_or_configs_relocated'},indent=2))
if __name__=='__main__':main()
