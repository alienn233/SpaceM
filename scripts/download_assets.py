import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='Download and install SpaceM models and reference libraries.')
    parser.add_argument('--only', choices=['all', 'models', 'reference'], default='all')
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--base-url')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'assets/ASSET_MANIFEST.json').read_text())
    cache = ROOT / 'assets/downloads'
    cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        if args.only != 'all' and asset['kind'] != args.only:
            continue
        name = asset['file']
        archive = args.local_dir / name if args.local_dir else cache / name
        if not archive.exists():
            if args.local_dir:
                raise FileNotFoundError(archive)
            url = (args.base_url or manifest['base_url']).rstrip('/') + '/' + name
            partial = archive.with_name(name + '.part')
            print('Downloading', name, flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'SpaceM/0.1.0'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(archive)
        if archive.stat().st_size != asset['bytes'] or checksum(archive) != asset['sha256']:
            raise ValueError('Asset integrity mismatch: ' + name)
        print('Installing', name, flush=True)
        if asset['kind'] == 'reference':
            dest = ROOT / asset['destination']
            dest.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(archive, 'rb') as source, dest.open('wb') as out:
                shutil.copyfileobj(source, out)
            if 'sequences' in asset:
                with dest.open() as handle:
                    count = sum(line.startswith('>') for line in handle)
                if count != asset['sequences']:
                    raise ValueError('FASTA count mismatch: ' + name)
        else:
            destination = ROOT / 'assets'
            with tarfile.open(archive, 'r:gz') as tar:
                for member in tar.getmembers():
                    target = (destination / member.name).resolve()
                    if destination.resolve() not in target.parents or not member.isfile():
                        raise ValueError('Unexpected archive member: ' + member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open('wb') as out:
                        shutil.copyfileobj(source, out)
    if args.only != 'reference':
        shutil.copytree(ROOT / 'algorithms/runtime', ROOT / 'assets/runtime', dirs_exist_ok=True)
    print('ASSETS_READY', flush=True)

if __name__ == '__main__':
    main()
