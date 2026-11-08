from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .io import fasta_dict, open_text, parse_attributes, reverse_complement
from .schema import A_MISSING, A_ORIGINAL, BASES, B_COLUMNS, C_COLUMNS, DINUCS, POSITIONS


STOP_BY_TABLE = {
    4: {"TAA", "TAG"},
    11: {"TAA", "TAG", "TGA"},
    25: {"TAA", "TAG"},
}
ALL_CODONS = tuple(a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG")
TETRA_KEYS = tuple(
    sorted({min(kmer, reverse_complement(kmer)) for kmer in (a + b + c + d for a in "ACGT" for b in "ACGT" for c in "ACGT" for d in "ACGT")})
)


@dataclass
class Segment:
    start0: int
    end0: int
    phase: int


@dataclass
class Gene:
    raw_id: str
    contig_id: str
    start0: int
    end0: int
    strand: str
    segments: list[Segment] = field(default_factory=list)
    transl_table: int = 11
    partial: bool = False
    sequence: str = ""
    protein_sequence: str = ""
    motif_raw: np.ndarray | None = None
    motif_z: np.ndarray | None = None
    codon_counts: np.ndarray | None = None
    valid_codon: bool = False
    has_ambiguous_nt: bool = False


def _protein_id(attributes: dict[str, str]) -> str | None:
    value = (
        attributes.get("protein_id")
        or attributes.get("ID")
        or attributes.get("locus_tag")
        or attributes.get("Name")
    )
    if value and value.startswith("cds-"):
        value = value[4:]
    return value


def _normalize_gff_protein_id(
    raw_id: str,
    contig_id: str,
    proteins: dict[str, str],
) -> tuple[str, bool]:

    if raw_id in proteins:
        return raw_id, False
    prefix, separator, suffix = raw_id.partition("_")
    if separator and prefix.isdigit() and suffix:
        candidate = f"{contig_id}_{suffix}"
        if candidate in proteins:
            return candidate, True
    return raw_id, False


def parse_gff(resource: str, contigs: dict[str, str], proteins: dict[str, str]) -> tuple[dict[str, list[Gene]], dict]:
    grouped: dict[tuple[str, str, str], list[tuple[int, int, int, dict[str, str]]]] = defaultdict(list)
    skipped = Counter()
    with open_text(resource) as handle:
        for raw in handle:
            if not raw or raw.startswith("#"):
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 9 or parts[2] != "CDS":
                continue
            contig_id = parts[0]
            if contig_id not in contigs:
                skipped["gff_contig_absent_from_fna"] += 1
                continue
            attributes = parse_attributes(parts[8])
            if "pseudo" in attributes or "pseudogene" in attributes:
                skipped["pseudogene"] += 1
                continue
            raw_id = _protein_id(attributes)
            if not raw_id:
                skipped["missing_protein_id"] += 1
                continue
            raw_id, remapped = _normalize_gff_protein_id(raw_id, contig_id, proteins)
            if remapped:
                skipped["prodigal_id_contig_remapped"] += 1
            start0 = int(parts[3]) - 1
            end0 = int(parts[4])
            if start0 < 0 or end0 <= start0 or end0 > len(contigs[contig_id]):
                skipped["invalid_coordinates"] += 1
                continue
            strand = parts[6]
            if strand not in {"+", "-"}:
                skipped["invalid_strand"] += 1
                continue
            try:
                phase = int(parts[7]) if parts[7] != "." else 0
            except ValueError:
                phase = 0
            grouped[(contig_id, raw_id, strand)].append((start0, end0, phase, attributes))

    by_contig: dict[str, list[Gene]] = defaultdict(list)
    duplicate_raw_ids = Counter(key[1] for key in grouped)
    duplicated_across_locations = {key for key, count in duplicate_raw_ids.items() if count > 1}
    for (contig_id, raw_id, strand), rows in grouped.items():
        if raw_id in duplicated_across_locations:
            skipped["protein_id_multiple_locations"] += 1
            continue
        rows.sort(key=lambda row: row[0], reverse=(strand == "-"))
        attributes = rows[0][3]
        segments = [Segment(row[0], row[1], row[2]) for row in rows]
        pieces = []
        for segment in segments:
            piece = contigs[contig_id][segment.start0:segment.end0]
            if strand == "-":
                piece = reverse_complement(piece)
            if segment.phase:
                piece = piece[segment.phase:]
            pieces.append(piece)
        sequence = "".join(pieces).upper()
        try:
            transl_table = int(attributes.get("transl_table", "11"))
        except ValueError:
            transl_table = 11
        partial_text = "|".join(
            str(attributes.get(key, ""))
            for key in ("partial", "start_range", "end_range", "pseudo", "exception")
        ).lower()
        partial = any(token in partial_text for token in ("true", "partial", "<", ">", "frameshift"))
        gene = Gene(
            raw_id=raw_id,
            contig_id=contig_id,
            start0=min(row[0] for row in rows),
            end0=max(row[1] for row in rows),
            strand=strand,
            segments=segments,
            transl_table=transl_table,
            partial=partial,
            sequence=sequence,
            protein_sequence=proteins.get(raw_id, ""),
            has_ambiguous_nt=any(base not in "ACGT" for base in sequence),
        )
        by_contig[contig_id].append(gene)
    for genes in by_contig.values():
        genes.sort(key=lambda gene: (gene.start0, gene.end0, gene.raw_id))
    return by_contig, dict(skipped)


def motif_vector(sequence: str) -> np.ndarray:
    length = len(sequence)
    if not length:
        return np.full(21, np.nan, dtype=np.float64)
    values = [(sequence.count("G") + sequence.count("C")) / length]
    values.extend(sequence.count(base) / length for base in BASES)
    values.extend(sequence.count(motif) / length for motif in DINUCS)
    return np.asarray(values, dtype=np.float64)


def zscore_motifs(genes: Iterable[Gene]) -> dict:
    genes = list(genes)
    matrix = np.stack([motif_vector(gene.sequence) for gene in genes])
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    valid_std = np.isfinite(std) & (std > 0)
    scaled = np.full_like(matrix, np.nan)
    scaled[:, valid_std] = (matrix[:, valid_std] - mean[valid_std]) / std[valid_std]
    for index, gene in enumerate(genes):
        gene.motif_raw = matrix[index]
        gene.motif_z = scaled[index]
    return {
        "motif_zero_variance_n": int(np.sum(~valid_std)),
        "motif_gene_n": len(genes),
    }


def gc_fraction(sequence: str) -> float:
    valid = sum(sequence.count(base) for base in "ACGT")
    if valid == 0:
        return float("nan")
    return (sequence.count("G") + sequence.count("C")) / valid


def codon_vector(sequence: str, table: int) -> tuple[np.ndarray, bool]:
    stops = STOP_BY_TABLE.get(table)
    if stops is None:
        return np.zeros(64, dtype=np.int64), False
    counts = np.zeros(64, dtype=np.int64)
    lookup = {codon: index for index, codon in enumerate(ALL_CODONS)}
    valid = False
    for start in range(0, len(sequence) - 2, 3):
        codon = sequence[start:start + 3]
        if codon in lookup and codon not in stops:
            counts[lookup[codon]] += 1
            valid = True
    for stop in stops:
        counts[lookup[stop]] = 0
    return counts, valid


def tetra_vector(sequence: str) -> np.ndarray:
    lookup = {word: index for index, word in enumerate(TETRA_KEYS)}
    result = np.zeros(len(TETRA_KEYS), dtype=np.int64)
    for start in range(max(0, len(sequence) - 3)):
        word = sequence[start:start + 4]
        if len(word) == 4 and all(base in "ACGT" for base in word):
            result[lookup[min(word, reverse_complement(word))]] += 1
    return result


def js_divergence(left: np.ndarray, right: np.ndarray, alpha: float, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        left = left[mask]
        right = right[mask]
    p = left.astype(np.float64) + alpha
    q = right.astype(np.float64) + alpha
    p /= p.sum()
    q /= q.sum()
    midpoint = (p + q) / 2.0
    left_term = np.sum(np.where(p > 0, p * np.log2(p / midpoint), 0.0))
    right_term = np.sum(np.where(q > 0, q * np.log2(q / midpoint), 0.0))
    return float((left_term + right_term) / 2.0)


def _relative_map(genes: list[Gene], center_index: int) -> dict[int, Gene | None]:
    center = genes[center_index]
    result: dict[int, Gene | None] = {}
    for relative in range(-2, 3):
        physical = center_index + relative if center.strand == "+" else center_index - relative
        result[relative] = genes[physical] if 0 <= physical < len(genes) else None
    return result


def _gap(left: Gene | None, right: Gene | None, original: bool) -> float:
    if left is None or right is None:
        return -200.0 if original else float("nan")
    low, high = sorted((left, right), key=lambda gene: (gene.start0, gene.end0))
    value = high.start0 - low.end0
    return float(value + 1 if original else value)


def _a_features(relative: dict[int, Gene | None], original: bool) -> dict[str, float]:
    prefix = "a_orig" if original else "a"
    result: dict[str, float] = {}
    for relative_position, position in POSITIONS:
        gene = relative[relative_position]
        vector = gene.motif_z if gene is not None else None
        names = [f"{prefix}_gc_z_{position}"]
        names.extend(f"{prefix}_nt_{base}_z_{position}" for base in BASES)
        names.extend(f"{prefix}_di_{motif}_z_{position}" for motif in DINUCS)
        for index, name in enumerate(names):
            result[name] = float(vector[index]) if vector is not None and np.isfinite(vector[index]) else (1.1 if original else float("nan"))
    for relative_position, position in POSITIONS:
        gene = relative[relative_position]
        result[f"{prefix}_aa_len_{position}"] = float(len(gene.protein_sequence)) if gene is not None and gene.protein_sequence else (0.0 if original else float("nan"))
    for relative_position, position in POSITIONS:
        gene = relative[relative_position]
        result[f"{prefix}_codir_{position}"] = float(gene.strand == relative[0].strand) if gene is not None else (2.0 if original else float("nan"))
    for left_rel, right_rel, left_name, right_name in (
        (-2, -1, "m2", "m1"), (-1, 0, "m1", "0"), (0, 1, "0", "p1"), (1, 2, "p1", "p2")
    ):
        result[f"{prefix}_gap_{left_name}_{right_name}"] = _gap(relative[left_rel], relative[right_rel], original)
    expected = A_ORIGINAL if original else A_MISSING
    if list(result) != expected:
        raise RuntimeError("A119 column ordering drift")
    return result


def _translation_identity(cds: str, protein: str, table: int) -> float:
    if not cds or not protein or table not in STOP_BY_TABLE:
        return float("nan")
    standard = {
        "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S","TAT":"Y","TAC":"Y","TGT":"C","TGC":"C","TGG":"W",
        "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
        "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T","AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
        "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A","GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    }
    translated = "".join(standard.get(cds[i:i+3], "X") for i in range(0, len(cds) - 2, 3))
    translated = translated.rstrip("*")
    n = min(len(translated), len(protein))
    if n == 0:
        return float("nan")
    matches = sum(a == b for a, b in zip(translated[:n], protein[:n]))
    if translated[0:1] != protein[0:1] and protein.startswith("M"):
        matches += 1
    return matches / max(len(translated), len(protein))


def build_genome_features(
    genome: dict,
    centers: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict], dict]:
    contigs = fasta_dict(genome["genome_fna"])
    proteins_raw = fasta_dict(genome["proteins_faa"])
    dataset = genome["dataset"]
    genome_id = genome["genome_id"]
    if dataset == "PRIMARY2000":
        proteins = {
            identifier.split("|", 1)[1] if identifier.startswith(genome_id + "|") else identifier: sequence
            for identifier, sequence in proteins_raw.items()
        }
    else:
        proteins = proteins_raw
    genes_by_contig, skipped = parse_gff(genome["gff"], contigs, proteins)
    all_genes = [gene for genes in genes_by_contig.values() for gene in genes]
    motif_qc = zscore_motifs(all_genes) if all_genes else {"motif_zero_variance_n": 21, "motif_gene_n": 0}
    gene_lookup = {gene.raw_id: (gene, genes, index) for genes in genes_by_contig.values() for index, gene in enumerate(genes)}

    contig_gc = {contig: gc_fraction(sequence) for contig, sequence in contigs.items()}
    valid_bases = {contig: sum(sequence.count(base) for base in "ACGT") for contig, sequence in contigs.items()}
    mag_valid = sum(valid_bases.values())
    mag_gc = (
        sum((sequence.count("G") + sequence.count("C")) for sequence in contigs.values()) / mag_valid
        if mag_valid else float("nan")
    )
    codon_totals = np.zeros(64, dtype=np.int64)
    codon_valid_genes = 0
    contig_codon: dict[str, np.ndarray] = {}
    contig_codon_valid: dict[str, int] = {}
    for contig_id, genes in genes_by_contig.items():
        total = np.zeros(64, dtype=np.int64)
        valid_n = 0
        for gene in genes:
            counts, valid = codon_vector(gene.sequence, gene.transl_table)
            gene.codon_counts = counts
            gene.valid_codon = valid
            if valid:
                total += counts
                valid_n += 1
        contig_codon[contig_id] = total
        contig_codon_valid[contig_id] = valid_n
        codon_totals += total
        codon_valid_genes += valid_n
    contig_tetra = {contig: tetra_vector(sequence) for contig, sequence in contigs.items()}
    tetra_total = sum(contig_tetra.values(), np.zeros(len(TETRA_KEYS), dtype=np.int64))

    rows: list[dict] = []
    qc_rows: list[dict] = []
    center_found = 0
    translation_values: list[float] = []
    context_counts: list[int] = []
    codon_missing = 0
    tetra_missing = 0
    for center in centers:
        raw_id = center["raw_protein_id"]
        base = {
            "row_index": int(center["row_index"]),
            "protein_id": center["protein_id"],
            "dataset": dataset,
            "genome_id": genome_id,
        }
        located = gene_lookup.get(raw_id)
        if located is None:
            row = dict(base)
            row.update({"protein_key": f"{genome_id}|MISSING|NA|NA|NA|{center['protein_id']}", "contig_id": "", "start0": np.nan, "end0": np.nan, "strand": ""})
            row.update({column: np.nan for column in A_ORIGINAL + A_MISSING + B_COLUMNS + C_COLUMNS})
            rows.append(row)
            qc_rows.append({
                **base, "protein_key": row["protein_key"], "contig_id": "", "gff_match_ok": False,
                "is_partial_cds": np.nan, "has_ambiguous_nt": np.nan, "window_context_count": 0,
                "translation_identity": np.nan, "missing_reason_A": "CENTER_NOT_IN_GFF",
                "missing_reason_C": "CENTER_NOT_IN_GFF", "circular_status": "UNKNOWN",
            })
            codon_missing += 1
            tetra_missing += 1
            continue
        gene, genes, index = located
        center_found += 1
        relative = _relative_map(genes, index)
        row = dict(base)
        protein_key = f"{genome_id}|{gene.contig_id}|{gene.start0}|{gene.end0}|{gene.strand}|{center['protein_id']}"
        row.update({"protein_key": protein_key, "contig_id": gene.contig_id, "start0": gene.start0, "end0": gene.end0, "strand": gene.strand})
        row.update(_a_features(relative, True))
        row.update(_a_features(relative, False))

        contig_length = len(contigs[gene.contig_id])
        upstream = sum(relative[position] is not None for position in (-2, -1))
        downstream = sum(relative[position] is not None for position in (1, 2))
        b_values = [
            contig_length, len(genes), 1000.0 * len(genes) / contig_length,
            gene.start0, contig_length - gene.end0,
            min(gene.start0, contig_length - gene.end0),
            min(gene.start0, contig_length - gene.end0) / contig_length,
            index, len(genes) - 1 - index,
            upstream, downstream, upstream + downstream,
            int(relative[-2] is not None), int(relative[-1] is not None),
            int(relative[1] is not None), int(relative[2] is not None),
        ]
        row.update(dict(zip(B_COLUMNS, map(float, b_values))))

        neighbor_gc = [gc_fraction(relative[position].sequence) for position in (-2, -1, 1, 2) if relative[position] is not None]
        center_gc = gc_fraction(gene.sequence)
        gc_min = int(config.get("gc_min_valid_bases", 100))
        center_valid_bases = sum(gene.sequence.count(base) for base in "ACGT")
        neighbor_gc_valid = [
            gc_fraction(relative[position].sequence)
            for position in (-2, -1, 1, 2)
            if relative[position] is not None
            and sum(relative[position].sequence.count(base) for base in "ACGT") >= gc_min
        ]
        c_values = [
            center_gc - contig_gc[gene.contig_id] if center_valid_bases >= gc_min and valid_bases[gene.contig_id] >= gc_min and np.isfinite(center_gc) and np.isfinite(contig_gc[gene.contig_id]) else np.nan,
            contig_gc[gene.contig_id] - mag_gc if valid_bases[gene.contig_id] >= gc_min and mag_valid >= gc_min and np.isfinite(contig_gc[gene.contig_id]) and np.isfinite(mag_gc) else np.nan,
            center_gc - float(np.mean(neighbor_gc_valid)) if center_valid_bases >= gc_min and np.isfinite(center_gc) and len(neighbor_gc_valid) >= 2 and np.isfinite(neighbor_gc_valid).all() else np.nan,
            np.nan, np.nan, np.nan,
        ]
        missing_c: list[str] = []
        if len(gene.sequence) >= int(config["codon_min_nt"]) and gene.valid_codon:
            mag_other = codon_totals - gene.codon_counts
            if codon_valid_genes - 1 >= int(config["codon_mag_bg_min_genes"]) and mag_other.sum() > 0:
                stops = STOP_BY_TABLE[gene.transl_table]
                sense_mask = np.asarray([codon not in stops for codon in ALL_CODONS])
                c_values[3] = js_divergence(gene.codon_counts, mag_other, float(config["pseudocount"]), sense_mask)
            else:
                missing_c.append("MAG_CODON_BACKGROUND_TOO_SMALL")
            contig_other = contig_codon[gene.contig_id] - gene.codon_counts
            if contig_codon_valid[gene.contig_id] - 1 >= int(config["codon_contig_bg_min_genes"]) and contig_other.sum() > 0:
                stops = STOP_BY_TABLE[gene.transl_table]
                sense_mask = np.asarray([codon not in stops for codon in ALL_CODONS])
                c_values[4] = js_divergence(gene.codon_counts, contig_other, float(config["pseudocount"]), sense_mask)
            else:
                missing_c.append("CONTIG_CODON_BACKGROUND_TOO_SMALL")
        else:
            missing_c.append("CENTER_CODON_INVALID_OR_SHORT")
        other_tetra = tetra_total - contig_tetra[gene.contig_id]
        other_length = sum(len(sequence) for contig, sequence in contigs.items() if contig != gene.contig_id)
        if contig_length >= int(config["tetra_min_contig_bp"]) and other_length >= int(config["tetra_min_rest_mag_bp"]) and other_tetra.sum() > 0:
            c_values[5] = js_divergence(contig_tetra[gene.contig_id], other_tetra, float(config["pseudocount"]))
        else:
            missing_c.append("TETRA_BACKGROUND_TOO_SMALL")
        row.update(dict(zip(C_COLUMNS, map(float, c_values))))
        rows.append(row)

        translation_identity = _translation_identity(gene.sequence, gene.protein_sequence, gene.transl_table)
        if np.isfinite(translation_identity):
            translation_values.append(translation_identity)
        context_count = upstream + downstream
        context_counts.append(context_count)
        codon_missing += int(not np.isfinite(c_values[3]) or not np.isfinite(c_values[4]))
        tetra_missing += int(not np.isfinite(c_values[5]))
        qc_rows.append({
            **base, "protein_key": protein_key, "contig_id": gene.contig_id, "gff_match_ok": True,
            "is_partial_cds": gene.partial, "has_ambiguous_nt": gene.has_ambiguous_nt,
            "window_context_count": context_count, "translation_identity": translation_identity,
            "missing_reason_A": "ZERO_VARIANCE" if any(not np.isfinite(value) for value in gene.motif_z) else "",
            "missing_reason_C": "|".join(missing_c), "circular_status": "UNKNOWN",
        })

    total_bp = sum(len(sequence) for sequence in contigs.values())
    ambiguous = sum(sum(base not in "ACGT" for base in sequence) for sequence in contigs.values())
    qc_mag = {
        "dataset": dataset,
        "genome_id": genome_id,
        "n_contigs": len(contigs),
        "total_bp": total_bp,
        "n_CDS": len(all_genes),
        "genome_gc": mag_gc,
        "missing_gff_fraction": 1.0 - (len(all_genes) / max(1, len(proteins))),
        "protein_id_match_rate": sum(bool(gene.protein_sequence) for gene in all_genes) / max(1, len(all_genes)),
        "mean_translation_identity": float(np.mean(translation_values)) if translation_values else np.nan,
        "mean_neighbor_observed": float(np.mean(context_counts)) if context_counts else np.nan,
        "fraction_center_with_4_neighbors": float(np.mean(np.asarray(context_counts) == 4)) if context_counts else np.nan,
        "fraction_codon_js_missing": codon_missing / max(1, len(centers)),
        "fraction_tetra_js_missing": tetra_missing / max(1, len(centers)),
        "ambiguous_base_rate": ambiguous / max(1, total_bp),
        "centers_requested": len(centers),
        "centers_mapped": center_found,
        "coordinate_source": genome["coordinate_source"],
        "gff_skipped": skipped,
        **motif_qc,
    }
    return rows, qc_rows, qc_mag
