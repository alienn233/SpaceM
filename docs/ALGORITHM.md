# SpaceM algorithm

SpaceM links protein identity, functional profiles and mobile genetic context through four stages: V3HN target recognition, V2 conditional classification, task-specific ESM2-8M fine models and UNIFIED248 mobility association.

## Reference data and genomic backgrounds

The frozen positive protein libraries contain 28,543 ARG, 55,397 MRG and 85,818 MGE sequences. Source membership, processing and database dependencies are documented in [DATABASES.md](DATABASES.md). The genome collection comprises 39,144 MGnify catalogue assemblies from gut, marine and soil resources and 2,000 NCBI reference assemblies. Sequence IDs, host and contig IDs, coordinates, strand and quality metadata connect each protein to its genomic context.

Earlier database-background experiments used Bagging-PU to select candidate negatives from unlabeled UniProt/Swiss-Prot proteins. V3HN uses native MAG backgrounds: candidates with target homology or enriched target-associated annotations are filtered, remaining sequences are clustered and sampled, and proteins lacking target evidence in real target neighborhoods are added. Annotation enrichment uses one-sided hypergeometric tests and Benjamini-Hochberg adjustment. Unknown-function proteins can enter the screened background; annotation absence alone does not assign a negative label.

## Inputs and feature representation

Inference takes protein sequences and prepared, ID-aligned sequence/context features. Protein IDs must agree across the FASTA, coordinate table and feature matrices. Coordinates are 1-based inclusive. The center protein and ten CDS on either side define the W10 neighborhood.

| Feature group | Biological information | Representation |
|---|---|---|
| Protein sequence | Evolutionary, structural and functional patterns | Frozen ESM2-8M and ESM2-650M embeddings, 320 and 1,280 dimensions |
| Neighborhood function | Domains, orthology and local functional composition | Pfam/eggNOG numerical features and fold-specific vocabularies |
| Gene organization | Protein length, local arrangement and visible neighbors | 44 coordinate-derived features with boundary missingness |
| DNA context | Nucleotide composition, local deviations and context completeness | 141 MAG DNA features |

Each hierarchy classifier receives 2,564 features. The binary and conditional stages have separate saved fold vocabularies and therefore separate matrices.

## Hierarchical classification

Five V3HN LightGBM models produce the mean target probability, followed by five V2 LightGBM models producing conditional ARG/MRG/MGE probabilities. Training uses class-balanced binary and multiclass log losses, respectively. For target probability g and conditional class probability q_k, the joint probabilities are P(OTHER) = 1 - g and P(k) = g q_k.

The fixed target threshold is 0.737447579539902, selected by maximum MCC on development out-of-fold predictions. Proteins passing the gate receive the class with the highest conditional probability. OTHER proteins retain their classification output and do not enter target functional profiling.

Evaluation groups proteins and their neighborhoods by genome, with StratifiedGroupKFold for MGnify samples. Fold-specific reference isolation removes validation positives and identical sequences before assigning propagated training labels to reference-assembly candidates.

## Functional profiling

Three independent branches fine-tune ESM2-t6-8M-UR50D end to end. Mean pooling over valid residues feeds LayerNorm, dropout (0.2) and a linear head.

| Branch | Output | Activation and loss |
|---|---|---|
| ARG | 15 drug labels | Independent sigmoid; weighted binary cross-entropy |
| MRG | 23 metal labels | Independent sigmoid; weighted binary cross-entropy |
| MGE | 7 mutually exclusive functions | Softmax; weighted multiclass cross-entropy |

Long proteins use 1,022-residue windows at stride 512 with a terminal window. Window logits are length-weighted before activation. AdamW training uses learning rate 5e-5, weight decay 0.01 and 5% warmup. Internal validation selects epochs and label thresholds; outer homology-cluster groups provide evaluation partitions.

Each deployed task averages 15 models from seeds 11, 22 and 33 across five folds. ARG/MRG apply averaged label thresholds to averaged probabilities. MGE selects one maximum-probability function. Frozen embeddings support target recognition, while fine-tuned embeddings resolve functions within target classes.

## Mobility association

UNIFIED248 scores ARG/MRG loci using upstream predictions, multiscale MGE spatial evidence, neighborhood function, DNA context, oriT and context visibility. Reference ICE/IME intervals provide supervision and evaluation labels. The input contains the fixed 248-feature whitelist; the supplied builder derives it from matching base229 and local44 tables.

The 0-1 score measures association with reference ICE/IME genetic environments. Its fixed decision threshold is 0.1396680282651571. The 5 kb proximity rule provides a separate comparator. Development analyses include full and truncated contexts; the example report presents full-context scores.

## Reference alignment

Predicted ARG/MRG/MGE proteins are searched against all three positive FAA libraries using DIAMOND sensitive mode, E <= 1e-5 and up to five hits per library. Outputs include reference IDs, names, source databases, labels, identity, query/reference coverage, E-value and bitscore. Strict support requires identity >90%, both coverages >90% and E <1e-5.

Model calls and reference evidence are reported separately. Strong cross-class hits are flagged for review. Reference-alignment rows are restricted to predicted targets.

## Outputs and interpretation

The HTML report supports search, filtering, sorting, functional probabilities, reference hits, mobility scores, neighborhood diagrams and CSV export. Unassessed neighboring proteins are displayed separately from OTHER.

Feature ablation and TreeSHAP support research-stage interpretation. The supplied Botelho examples are development cases scored by the deployed models; archived out-of-fold scores are retained separately. Input contracts and software dependencies are documented in [INPUTS_OUTPUTS.md](INPUTS_OUTPUTS.md) and [SOFTWARE.md](SOFTWARE.md).
