# Trained models and large assets

The Git repository contains source code, small examples, documentation and figures. Full trained weights and database archives remain separate from Git source history. **A fresh clone does not yet contain all assets required for model inference. No public asset download URL has been published in this release.**

The verified B5 archive contains:

| Archive | Compressed bytes | Purpose |
|---|---:|---|
| `01_algorithm_models.tar.zst` | 2,838,274,818 | Frozen hierarchy, fine models and runtime |
| `02_training_data.tar.zst` | 17,822,996,859 | Training-data and provenance archive |
| `03_botelho_results_and_sources.tar.zst` | 22,459,159,195 | Context datasets, results and source scripts |
| `04_annotation_resources.tar.zst` | 12,598,820,248 | Pfam, eggNOG and oriT assets |
| `05_preprocessing_references.tar.zst` | 22,112,049 | Additional target-alignment and preprocessing references |

Each archive was stream-validated when built. Internal integrity manifests accompany the B5 distribution. These sizes describe existing archives, not files claimed to be hosted in this repository.

The runtime expects:

```text
runtime/
  bin/ and lib/
  config/runtime.json
  models/v3_gate/fold_checkpoints/fold0..4/
  models/v2_conditional/fold_checkpoints/fold0..4/
  models/unified248/
  vault/                         # three tasks, 15 fine checkpoints per task
```

Point `SPACEM_RUNTIME` to the extracted complete runtime, not the source-only `algorithms/runtime` directory. Place the positive FAA libraries and metadata under `assets/references/` or adjust `examples/botelho/references.json`. The example HTML can be viewed without any model or database download.

Source snapshots in `algorithms/research` retain stage-specific training entry points. They require the matching archived input tables, labels, split definitions and dependencies. They are not interchangeable one-command retraining recipes, and should not be run against a new dataset without adapting the input contract.

Third-party resources retain their original redistribution terms. This repository does not apply a new license to those resources.
