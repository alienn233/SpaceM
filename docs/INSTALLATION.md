# Installation and reproduction

## Platform and environment manager

The CPU environment definitions target Linux x86-64, as used on B5. Install [Miniforge](https://github.com/conda-forge/miniforge) if Conda or Mamba is not already available:

```bash
curl -L --fail -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate base
conda --version
mamba --version
```

Download and extract `SpaceM_source_examples_20261108.zip`, then run the commands below from its `SpaceM/` directory.

## Create the inference environments

Using Conda:

```bash
conda env create -f envs/score.yml
conda env create -f envs/fine-cpu.yml
conda env create -f envs/tools.yml
```

Using Mamba instead:

```bash
mamba env create -f envs/score.yml
mamba env create -f envs/fine-cpu.yml
mamba env create -f envs/tools.yml
```

The fine environment installs the PyTorch 2.5.1 CPU build from the [official PyTorch package index](https://pytorch.org/get-started/previous-versions/). Package versions are pinned in the environment files. The scoring environment includes scikit-learn and SciPy because the mobility feature builder imports the corresponding research modules.

Check imports and DIAMOND:

```bash
conda run -n spacem-score python -c 'import numpy,pandas,pyarrow,lightgbm,sklearn,scipy; print("Scoring environment ready")'
conda run -n spacem-fine python -c 'import torch,transformers,matplotlib,pyarrow; print("Fine environment ready", torch.__version__)'
conda run -n spacem-tools diamond version
```

## Install trained models and positive libraries

```bash
conda run -n spacem-score python scripts/download_assets.py
```

The installer retrieves the assets listed in `assets/ASSET_MANIFEST.json` from Release v0.1.0, checks their integrity, and installs the runtime plus the three positive libraries. The resulting layout is:

```text
assets/
  ASSET_MANIFEST.json
  downloads/
  runtime/
    bin/
    lib/
    config/
    models/v3_gate/
    models/v2_conditional/
    models/unified248/
    vault/models/esm2_t6_8M_UR50D/
    vault/artifacts/
  references/
    ARG/ARG_POS_ALL.faa
    ARG/ARG_POS_ALL.tsv
    MRG/PROTEIN_POSITIVES.faa
    MRG/PROTEIN_POSITIVE_METADATA.tsv
    MGE/POS_7CLASS.faa
    MGE/POS_7CLASS.tsv
```

Reference paths are already configured in `examples/botelho/references.json`. For installations with locally downloaded release files:

```bash
conda run -n spacem-score python scripts/download_assets.py --local-dir /path/to/release_assets
```

To download through an authenticated GitHub CLI session and install the same verified files:

```bash
mamba install -n spacem-tools -c conda-forge gh
conda run -n spacem-tools gh auth login
conda run -n spacem-tools gh release download v0.1.0 --repo alienn233/SpaceM --dir assets/downloads
conda run -n spacem-score python scripts/download_assets.py --local-dir assets/downloads
```

## Run the complete example

```bash
conda run -n spacem-score python scripts/run_demo.py --threads 4 --out results/my_demo
conda run -n spacem-score python scripts/validate_demo.py --inputs examples/botelho/inputs.json --out results/my_demo
```

`run_demo.py` locates the fine-inference Python and DIAMOND in their named environments, then executes the complete prepared-feature workflow. It writes the HTML report to `results/my_demo/report.html`. With Mamba, use `mamba run` and add `--manager mamba` to `run_demo.py`.

The demonstration inputs include precomputed ESM2, Pfam, eggNOG and DNA-context features. These inputs are sufficient to reproduce its model predictions and reference alignments.

The [Reproduce example workflow](../.github/workflows/reproduce-example.yml) executes these installation, prediction and validation stages on a fresh GitHub runner. Fork the repository and select **Actions > Reproduce example > Run workflow**. Download `SpaceM-reproduced-results` from the completed run to obtain its HTML report, figures, tables and validation record.

## Generate features for additional genomes

Additional genome analysis uses nucleotide assemblies, protein sequences and matching CDS coordinates. The source snapshots and the column contracts are described in [ALGORITHM.md](ALGORITHM.md) and [INPUTS_OUTPUTS.md](INPUTS_OUTPUTS.md).

Install the additional annotation tools:

```bash
mamba env create -f envs/annotation.yml
mamba create -n spacem-eggnog -c conda-forge -c bioconda --strict-channel-priority python=3.10 eggnog-mapper=2.1.15
```

| Resource | Release used | Preparation |
|---|---|---|
| ESM2-t6-8M-UR50D | Named pretrained checkpoint | Included with the core runtime for fine inference; also used for frozen 320-dimensional embeddings |
| ESM2-t33-650M-UR50D | Named pretrained checkpoint | Download the upstream model for frozen 1,280-dimensional embeddings |
| Pfam-A | 38.2 | Obtain the corresponding HMM release and build HMMER indexes |
| eggNOG | 5.0.2 with mapper 2.1.15 | Install the matching database and configure `EGGNOG_DATA_DIR` |
| oriT | Frozen experimental reference | Use the archived oriT sequences and metadata to reproduce nucleotide evidence |
| ARG/MRG/MGE positive libraries | Frozen SpaceM exports | Installed by `download_assets.py` |

For ESM2-650M, download the upstream Hugging Face checkpoint using the fine environment:

```bash
conda run -n spacem-fine python -c 'from huggingface_hub import snapshot_download; snapshot_download("facebook/esm2_t33_650M_UR50D", local_dir="assets/pretrained/esm2_t33_650M_UR50D", allow_patterns=["config.json","model.safetensors","tokenizer_config.json","special_tokens_map.json","vocab.txt"])'
```

Place the Pfam release under `assets/annotation/pfam/`, then index it:

```bash
conda run -n spacem-annotation hmmpress assets/annotation/pfam/Pfam-A.hmm
```

Install the eggNOG database with the version-matched mapper environment:

```bash
mkdir -p assets/annotation/eggnog
conda run -n spacem-eggnog download_eggnog_data.py -y --data_dir assets/annotation/eggnog
export EGGNOG_DATA_DIR="$PWD/assets/annotation/eggnog"
```

Pfam and eggNOG versions must match the frozen feature definitions when applying the released models. Coordinate and sequence changes require regeneration of the dependent context features.

## Upstream documentation

[Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html), [Miniforge](https://github.com/conda-forge/miniforge), [PyTorch CPU packages](https://pytorch.org/get-started/previous-versions/), [eggNOG-mapper](https://github.com/eggnogdb/eggnog-mapper), and [ESM2-650M](https://huggingface.co/facebook/esm2_t33_650M_UR50D).
