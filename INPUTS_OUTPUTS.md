# Inputs and outputs

`inputs.json` resolves relative paths from its own directory. Every array row must follow `ids.txt`; TSV/parquet IDs must match the same proteins.

| Input key | Contract |
|---|---|
| `ids` | Unique protein IDs in matrix row order |
| `protein_fasta` | Amino-acid sequences with matching first-token IDs |
| `protein_table` | `protein_id`, `sample_id`, `contig_id`, `start`, `end`, `strand`; 1-based inclusive coordinates |
| `esm8`, `esm650` | Frozen protein embeddings, N×320 and N×1280 |
| `portable_context` | N×44 coordinate-safe organization features |
| `pfam`, `eggnog` | Numeric annotations and W10 tokens consumed by the saved fold vocabularies |
| `mag` | ID-aligned 141-feature DNA-context table |
| `mobility_base229`, `mobility_local44` | Matching visible-context tables keyed by protein, context length and placement |
| `mobility_features` | Alternative prevalidated table with the 248 locked model columns |
| `positive_references` | Optional ARG/MRG/MGE FASTA and metadata configuration |
| `case_table`, `case_genes` | Optional demonstration coordinates for neighborhood figures |

Do not fill absent context with arbitrary zeros. A boundary-related missing neighbor must follow the feature builder's original missingness convention. Missing annotation is not the same as negative function.

| Output | Content |
|---|---|
| `GENE_CLASSIFICATION.tsv` | One row per protein; target gate, conditional/joint class probabilities and fine labels |
| `ALL_STAGES.tsv` | Classification joined to available mobility contexts; raw analysis output |
| `02_hierarchy/` | Coarse predictions and routed target FASTA files |
| `03_fine/` | ARG/MRG multilabel and MGE single-label probabilities, thresholds and selected labels |
| `04_mobility/` | Model score, fixed threshold and call, with prediction-context metadata |
| `05_alignment/PREDICTED_TARGET_REFERENCE_HITS.tsv` | Displayed positive-library hits for predicted ARG/MRG/MGE only |
| `PREDICTIONS_WITH_REFERENCE.tsv` | Per-protein summary; OTHER reference fields are blank; target reference disagreements retained |
| `report.html` | Offline interactive report: filter, sort, inspect, download |
| `RESULTS_中文.txt` | Chinese example explanations and figure legends |
| `figures/` | Workflow, classification, fine functions, full-context mobility and neighborhoods |

MGE/OTHER cargo mobility is `NOT_APPLICABLE`. ARG/MRG without supplied context is `CONTEXT_NOT_PROVIDED`. Predicted OTHER proteins remain in the classification table, but their positive-reference alignments are not displayed or included in the target-hit download. This display restriction is not an accuracy claim about excluded proteins.

Within a predicted target, strong hits to multiple reference classes remain visible and are flagged for review. The report does not silently overwrite model output with homology output. The mobility section displays full available contexts only; archived fragment analysis files are not a report panel.
