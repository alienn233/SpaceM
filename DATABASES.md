# Database and resource dependencies

This inventory describes the frozen SpaceM release, rather than the newest upstream database releases. Reference-library counts and fine-classification training counts are different quantities.

## Positive protein references

| Library | Included evidence and processing | Final protein FAA records |
|---|---|---:|
| ARG | DeepARG v2 feature proteins; MEGARes 3.00 and ResFinder 2.6.0 CDS translated to proteins; ARGminer arg_v5 records marked NCBI; normalized metadata and exact-sequence deduplication | 28,543 |
| MRG | BacMet experimental proteins (484), AMRFinderPlus curated proteins (47), and BacMet predicted proteins (54,866) in the frozen protein export | 55,397 |
| MGE | mobileOG-db Beatrix 1.6, ICEberg CDS, TnCentral CDS, and IntegronFinder-derived integron integrases; curated seven-function mapping and exact-sequence deduplication | 85,818 |

The final MGE representative-source counts are mobileOG 83,559, ICEberg 1,910, TnCentral 322 and IntegronFinder 27. These are representative assignments after deduplication, not all overlapping source memberships.

The MRG frozen export also contains 578 MetalResistance records whose sequences are nucleotide/CDS rather than amino acids. They are retained separately as nucleotide records and excluded from the 55,397-protein FAA library. BacMet predicted evidence is homology-derived, not equivalent to per-record experimental validation.

### CARD and argNorm

CARD-derived provenance is present in 1,036 final ARG records, including inherited provenance in integrated source databases. This does **not** mean that the independently inventoried CARD 2026-08 FASTA was directly concatenated into this frozen training library. That standalone CARD collection was retained in the source inventory. The integrated ARG source memberships overlap: DeepARG 12,276; MEGARes 7,758; ResFinder 3,101; NCBI-marked ARGminer 17,251. Their sum is not the final deduplicated count.

argNorm **1.1.0** standardizes ARG annotation labels using its mapping resources and ARO terminology. It is a label-normalization tool, not a separate sequence database. Unmapped labels must be represented explicitly rather than described as experimental absence of resistance.

## Genome and mobile-element context

| Resource | Frozen role |
|---|---|
| MGnify human-gut v3.0, marine v2.0, soil v1.0 | Diverse genome assemblies, native background proteins and gene neighborhoods; 39,144 catalogue assemblies in the source collection |
| NCBI reference assemblies / Pathogen Detection selection | 2,000 pathogen-related reference assemblies; not an unselected population sample |
| Botelho ICE/IME reference cohort | Reference intervals and host-genome contexts for development, comparisons and examples |
| ICEberg | Curated mobile-element proteins and descriptive external reference evaluation; distinct uses must be tracked separately |

Reference intervals supply supervision/evaluation and figure annotations; they are not permitted as prediction features. Genome inventory counts, quality-filtered cohorts and per-analysis eligible subsets should be reported separately.

## Annotation and pretrained resources

| Resource | Verified version or snapshot | Required for |
|---|---|---|
| ESM2-t6-8M-UR50D | Named pretrained checkpoint | Frozen 320D embeddings and the three fine-tuning branches |
| ESM2-t33-650M-UR50D | Named pretrained checkpoint | Frozen 1,280D protein embeddings |
| Pfam-A | **38.2**, dated 2026-01, 30,134 families in the local version file | Protein-domain and neighborhood annotation |
| eggNOG | **5.0.2** | Functional annotation and neighborhood features |
| oriT experimental reference | Frozen `oriT_exp.clean.fasta` and accompanying metadata | Nucleotide homology evidence for mobility features |
| IntegronFinder covariance/annotation resources | IntegronFinder **2.0.6** used during MGE resource construction | Recovery of integron-integrase evidence; not a mandatory extra model at every inference run |

The eggNOG snapshot was verified with eggNOG-mapper **2.1.15**. A separately present v3 beta installation targets a different database generation and must not silently replace this pair. No additional public version identifier is inferred for frozen resources lacking a recorded release.

## Provisioning

The positive FAA libraries, metadata and trained checkpoints are distributed in [Release v0.1.0](https://github.com/alienn233/SpaceM/releases/tag/v0.1.0). Run `python scripts/download_assets.py` to install them at the paths configured in `examples/botelho/references.json`. Each ARG/MRG/MGE entry requires a protein `fasta` and matching `metadata` TSV. Relative paths resolve from the configuration file's directory.

Reference annotations must retain stable IDs, labels and source information; reusing a different release without rebuilding the relevant features can break the trained vocabulary and feature contract. Third-party databases retain their respective licenses and citation requirements.
