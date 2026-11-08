# Models and positive references

[Release v0.1.0](https://github.com/alienn233/SpaceM/releases/tag/v0.1.0) contains the trained inference models and reference libraries. Run `python scripts/download_assets.py` from the extracted SpaceM source directory to download, verify and install them.

| Archive | Installed contents |
|---|---|
| `SpaceM_core_models.tar.gz` | Binary and conditional LightGBM checkpoints, fold vocabularies, UNIFIED248 model/feature lock and the ESM2-8M backbone |
| `SpaceM_ARG_fine_models.tar.gz` | ARG checkpoints for three seeds and five folds |
| `SpaceM_MRG_fine_models.tar.gz` | MRG checkpoints for three seeds and five folds |
| `SpaceM_MGE_fine_models.tar.gz` | MGE checkpoints for three seeds and five folds |
| Six reference `.gz` files | ARG/MRG/MGE positive FAA files and matching metadata TSV files |

The asset manifest records archive sizes, integrity checks and FAA sequence counts. The installer combines the weights with the source runtime under `assets/runtime/`. Positive references are placed under `assets/references/ARG`, `MRG` and `MGE`, matching `examples/botelho/references.json`.

The example supplies frozen ESM2-650M embeddings. Generating embeddings for additional proteins uses the upstream [ESM2-t33-650M-UR50D checkpoint](https://huggingface.co/facebook/esm2_t33_650M_UR50D).

Research/training source snapshots are in `algorithms/research/`. Full training-data archives and stage-specific split tables are maintained separately from this inference release.
