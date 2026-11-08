# Software and environments

For Conda/Mamba setup and asset installation, see [INSTALLATION.md](INSTALLATION.md).

## Verified B5 inference environment

The deployed workflow uses two Python environments. The versions below were read from the installed environment or its retained run record; they are not claims about current upstream releases.

| Component | Scoring / alignment | Fine inference / reporting |
|---|---|---|
| Python | 3.14.7 | 3.10.21 |
| NumPy | 2.5.3 | 2.2.6 |
| pandas | 3.0.5 | 2.3.3 |
| PyArrow | 25.0.0 | 25.0.1 |
| LightGBM | 4.7.0 | — |
| scikit-learn | 1.9.1 | — |
| SciPy | 1.18.1 | — |
| PyTorch | — | 2.5.1 (CPU build on B5) |
| Transformers | — | 4.57.6 |
| Matplotlib | — | 3.10.9 |
| DIAMOND | 2.2.6 for the current reference-alignment run | — |

Install Python packages from `requirements-score.txt` and `requirements-fine.txt`. For example, after creating the corresponding environments:

```bash
python -m pip install -r requirements-score.txt
/path/to/fine-environment/bin/python -m pip install -r requirements-fine.txt
```

The full upstream annotation workflow additionally uses:

| Software | Role | Recorded version |
|---|---|---|
| HMMER (`hmmscan`, `hmmpress`) | Pfam domain annotation | 3.4 in the scoring-tools environment |
| eggNOG-mapper | Orthology/function annotation | 2.1.15 paired with DB 5.0.2 |
| DIAMOND bundled with eggNOG-mapper | eggNOG sequence search | 2.0.11 in the retained installation report |
| MMseqs2 | Sequence clustering and dataset construction | 18.8cc5c in the inspected tools environment; historical clustering runs retain their own records |
| CD-HIT | Reference-sequence redundancy reduction where used by the source builder | Exact historical executable version must be taken from the respective construction record |
| argNorm | ARG label harmonization | 1.1.0 |
| IntegronFinder | MGE source curation | 2.0.6 |
| BLAST+ | Nucleotide oriT reference searches | Use the archived search settings; no version inferred here |
| scikit-learn, SciPy | Grouped splits, statistics and training utilities | Stage-specific training environment |
| SHAP | Research-stage model interpretation | Stage-specific research environment |

Tools used to construct a training database are not all required to run a prepared-feature inference example. Conversely, installing the small inference requirements files does not recreate all raw genome annotation dependencies.

## Run inference

After installing the trained runtime and positive reference assets:

```bash
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SPACEM_RUNTIME=/absolute/path/to/runtime
export SPACEM_FINE_PYTHON=/absolute/path/to/fine-environment/bin/python
export SPACEM_DIAMOND=/absolute/path/to/diamond

/absolute/path/to/score-environment/bin/python scripts/workflow.py \
  --inputs examples/botelho/inputs.json \
  --out results/new_run --threads 4
```

Use a new output directory. `SPACEM_PLOT_PYTHON` can independently select the reporting environment; otherwise it uses `SPACEM_FINE_PYTHON`. A modern browser can open the generated HTML offline; no JavaScript package installation or external CDN is required.

The complete demonstration previously executed all ten hierarchy classifiers, 45 fine classifiers and the mobility model. Examples have 10 proteins: two predicted ARG, two MRG, four MGE and two OTHER. Reference alignments are displayed only for the eight predicted targets. Four ARG/MRG loci have full-context mobility scores. These are deployment examples, not independent performance estimates.
