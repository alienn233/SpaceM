import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def environment_path(manager, name, expression):
    result = subprocess.run([manager, 'run', '-n', name, 'python', '-c', expression], check=True, text=True, capture_output=True)
    value = result.stdout.strip().splitlines()[-1]
    if not Path(value).is_file():
        raise FileNotFoundError(value)
    return value

def main():
    parser = argparse.ArgumentParser(description='Run the supplied SpaceM Botelho example.')
    parser.add_argument('--manager', choices=['conda', 'mamba'], default='conda')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/my_demo')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fine-python')
    parser.add_argument('--diamond')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    env['SPACEM_RUNTIME'] = str(ROOT / 'assets/runtime')
    env['SPACEM_FINE_PYTHON'] = args.fine_python or environment_path(args.manager, 'spacem-fine', 'import sys;print(sys.executable)')
    env['SPACEM_DIAMOND'] = args.diamond or environment_path(args.manager, 'spacem-tools', 'import shutil;print(shutil.which("diamond"))')
    subprocess.run([sys.executable, str(ROOT / 'scripts/workflow.py'), '--inputs', str(ROOT / 'examples/botelho/inputs.json'), '--out', str(args.out.resolve()), '--threads', str(args.threads)], env=env, check=True)

if __name__ == '__main__':
    main()
