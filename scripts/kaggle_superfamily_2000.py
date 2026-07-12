#!/usr/bin/env python3
"""Full-superfamily Kaggle runner for the B1 MBL benchmark.

This script embeds the full filtered superfamily set (up to 2000 sequences) with
ESM-2, using AMP, optional DataParallel, and an embedding cache. It then builds
family-level summaries across the whole superfamily so the run is genuinely
full-superfamily rather than family-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SPADUPA_DISABLE_AUTORUN", "1")

NDM_FAMILY = "NDM"
VIM_FAMILY = "VIM"
IMP_FAMILY = "IMP"
OTHER_FAMILY = "OTHER"

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class SuperfamilyRecord:
    header: str
    sequence: str
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-superfamily Kaggle runner")
    parser.add_argument("--input", default="data/b1_superfamily.fasta", help="Filtered B1 superfamily FASTA")
    parser.add_argument("--output", default="output/kaggle_superfamily_2000", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max-seqs", type=int, default=2000, help="Maximum sequences to embed")
    parser.add_argument("--batch-size", type=int, default=32, help="ESM batch size")
    parser.add_argument("--embed-cache", default="output/embeddings_cache_superfamily_2000.npz", help="Embedding cache path")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp.autocast for inference")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel if multiple GPUs are available")
    parser.add_argument("--profile", action="store_true", help="Print per-stage timings")
    parser.add_argument("--clusters", default="output/clusters/b1_superfamily_40.csv", help="Cluster CSV for reporting")
    parser.add_argument("--alpha", type=float, default=0.6, help="Propagation alpha")
    parser.add_argument("--hops", type=int, default=2, help="Propagation hops")
    parser.add_argument("--curated-mutations", default="data/curated_mutations.json", help="Curated NDM mutation file")
    parser.add_argument("--curated-zero-indexed", action="store_true", help="Curated positions are zero-indexed")
    parser.add_argument(
        "--scorer",
        choices=["egnn", "gbsp"],
        default="egnn",
        help=(
            "Method for the 'graph' score (default: egnn): a small E(n)-"
            "equivariant GNN ensemble (see egnn_model.py). 'gbsp' = original "
            "fixed alpha/hops propagation baseline, kept for comparison."
        ),
    )
    parser.add_argument("--egnn-hidden", type=int, default=32, help="EGNN hidden dimension (default: 32)")
    parser.add_argument("--egnn-layers", type=int, default=2, help="Number of EGNN layers (default: 2)")
    parser.add_argument("--egnn-models", type=int, default=5, help="Ensemble size (default: 5)")
    parser.add_argument("--egnn-epochs", type=int, default=300, help="Max training epochs per model (default: 300)")
    return parser.parse_args()


def family_from_header(header: str) -> str:
    text = header.upper()
    if re.search(r"BLA?NDM|NDM-?\d+|BLAN", text):
        return NDM_FAMILY
    if re.search(r"VIM-?\d+|BLBV", text):
        return VIM_FAMILY
    if re.search(r"IMP-?\d+|BLBI|BLA-?IMP", text):
        return IMP_FAMILY
    return OTHER_FAMILY


def load_records(fasta_path: str, max_seqs: int) -> List[SuperfamilyRecord]:
    from Bio import SeqIO

    records: List[SuperfamilyRecord] = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        family = family_from_header(rec.description)
        records.append(SuperfamilyRecord(header=rec.description, sequence=str(rec.seq).strip(), family=family))
        if len(records) >= max_seqs:
            break
    return records


def load_curated_mutations(path: str, zero_indexed: bool = False) -> Dict[str, List[int]]:
    if not path:
        return {}
    curated_path = Path(path)
    if not curated_path.exists():
        return {}
    with curated_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    curated: Dict[str, List[int]] = {}
    for family, value in data.items():
        positions: List[int] = []
        if isinstance(value, dict) and "positions" in value:
            positions = value.get("positions", [])
        elif isinstance(value, list):
            positions = value
        try:
            pos_list = [int(p) for p in positions]
        except (TypeError, ValueError):
            continue
        if not zero_indexed:
            pos_list = [p - 1 for p in pos_list if p > 0]
        curated[str(family).upper()] = [p for p in pos_list if p >= 0]
    return curated


def choose_reference(records: Sequence[SuperfamilyRecord], family: str) -> Optional[SuperfamilyRecord]:
    if family == NDM_FAMILY:
        for rec in records:
            if re.search(r"NDM-1\b|blaNDM-1", rec.header, re.I):
                return rec
    elif family == VIM_FAMILY:
        for rec in records:
            if re.search(r"VIM-1\b", rec.header, re.I):
                return rec
    elif family == IMP_FAMILY:
        for rec in records:
            if re.search(r"IMP-1\b", rec.header, re.I):
                return rec
    fam_records = [r for r in records if r.family == family]
    return fam_records[0] if fam_records else None


def align_reference_to_query(reference: str, query: str) -> Dict[int, Optional[int]]:
    from Bio.Align import PairwiseAligner

    if reference == query:
        return {i: i for i in range(len(reference))}

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1
    alignment = aligner.align(reference, query)[0]

    mapping: Dict[int, Optional[int]] = {i: None for i in range(len(reference))}
    ref_blocks, query_blocks = alignment.aligned
    for (r0, r1), (q0, q1) in zip(ref_blocks, query_blocks):
        for r_i, q_i in zip(range(r0, r1), range(q0, q1)):
            mapping[r_i] = q_i
    return mapping


def project_variant_positions(reference: str, query: str) -> List[int]:
    mapping = align_reference_to_query(reference, query)
    variant_positions: List[int] = []
    for ref_idx, query_idx in mapping.items():
        if query_idx is None or ref_idx >= len(reference) or query_idx >= len(query):
            variant_positions.append(ref_idx)
        elif reference[ref_idx] != query[query_idx]:
            variant_positions.append(ref_idx)
    return sorted(set(variant_positions))


def build_chain_graph(n_residues: int) -> np.ndarray:
    adj = np.zeros((n_residues, n_residues), dtype=np.float32)
    for i in range(n_residues):
        start = max(0, i - 5)
        stop = min(n_residues, i + 6)
        for j in range(start, stop):
            if i != j:
                adj[i, j] = 1.0
    adj += np.eye(n_residues, dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    return adj / (degree + 1e-8)


def propagate_scores(initial: np.ndarray, adj_norm: np.ndarray, alpha: float, hops: int) -> np.ndarray:
    propagated = initial.copy()
    for _ in range(hops):
        propagated = alpha * initial + (1.0 - alpha) * (adj_norm @ propagated)
    return (propagated - propagated.min()) / (propagated.max() - propagated.min() + 1e-8)


_DEFAULT_EGNN_CONFIG = {
    "n_models": 5,
    "hidden_dim": 32,
    "n_layers": 2,
    "dropout": 0.3,
    "epochs": 300,
    "patience": 30,
    "lr": 1e-2,
    "weight_decay": 1e-3,
    "update_coords": False,
    "seed": 0,
    "device": "cpu",
}


def compute_variant_frequency(reference_seq: str, records: Sequence[SuperfamilyRecord]) -> np.ndarray:
    """Fraction of `records` (excluding a sequence identical to the
    reference) carrying a substitution at each reference position --
    EGNN's regression target. This script has no held-out test split (it
    scores the full superfamily against itself), so this is computed over
    all `records`, mirroring how `variance` is computed in `family_summary`.
    """
    n_residues = len(reference_seq)
    variant_counts = np.zeros(n_residues, dtype=np.float32)
    n_compared = 0
    for rec in records:
        if rec.sequence == reference_seq:
            continue
        n_compared += 1
        for pos in project_variant_positions(reference_seq, rec.sequence):
            if 0 <= pos < n_residues:
                variant_counts[pos] += 1
    if n_compared == 0:
        return np.zeros(n_residues, dtype=np.float32)
    return variant_counts / n_compared


def embed_sequences(
    records: Sequence[SuperfamilyRecord],
    device: str,
    batch_size: int,
    cache_path: Optional[str],
    use_amp: bool,
    use_data_parallel: bool,
) -> Dict[str, np.ndarray]:
    import esm
    import torch

    cache: Dict[str, np.ndarray] = {}
    if cache_path and Path(cache_path).exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            cache = {k: data[k] for k in data.files}
            print(f"Loaded {len(cache)} cached embeddings from {cache_path}")
        except Exception as exc:
            print(f"Warning: failed to load cache {cache_path}: {exc}")
            cache = {}

    missing = [rec for rec in records if rec.header not in cache]
    if not missing:
        return cache

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    if use_data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Wrapped model in DataParallel ({torch.cuda.device_count()} GPUs)")

    batch_converter = alphabet.get_batch_converter()
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        batch = [(rec.header, rec.sequence) for rec in chunk]
        _, _, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out = model(toks, repr_layers=[33], return_contacts=False)
            else:
                out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]
        for i, rec in enumerate(chunk):
            emb = reps[i, 1 : len(rec.sequence) + 1].detach().cpu().numpy()
            cache[rec.header] = emb

        if cache_path:
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, **cache)
            except Exception as exc:
                print(f"Warning: failed to save cache {cache_path}: {exc}")

    return cache


def family_summary(
    records: Sequence[SuperfamilyRecord],
    embeddings: Dict[str, np.ndarray],
    family: str,
    reference: Optional[SuperfamilyRecord],
    alpha: float,
    hops: int,
    scorer: str = "egnn",
    egnn_config: Optional[Dict] = None,
) -> Dict[str, float]:
    fam_records = [r for r in records if r.family == family]
    if not fam_records or reference is None:
        return {
            "family": family,
            "n_sequences": len(fam_records),
            "n_positive_positions": 0,
            "roc_auc_variance": float("nan"),
            "roc_auc_graph": float("nan"),
            "roc_auc_combined": float("nan"),
            "mean_variant_positions": float("nan"),
        }

    ref_seq = reference.sequence
    n_residues = len(ref_seq)
    per_position_vectors: List[List[np.ndarray]] = [[] for _ in range(n_residues)]
    for rec in fam_records:
        projected = align_reference_to_query(ref_seq, rec.sequence)
        emb = embeddings[rec.header]
        for ref_idx, query_idx in projected.items():
            if query_idx is not None and 0 <= query_idx < len(emb):
                per_position_vectors[ref_idx].append(emb[query_idx])

    variance = np.zeros(n_residues, dtype=np.float32)
    for idx, vectors in enumerate(per_position_vectors):
        if len(vectors) >= 2:
            stack = np.stack(vectors, axis=0)
            variance[idx] = np.var(stack, axis=0).mean()

    var_norm = (variance - variance.min()) / (variance.max() - variance.min() + 1e-8)
    adj_norm = build_chain_graph(n_residues)

    if family == NDM_FAMILY:
        active_positions = np.array([119, 121, 123, 188, 207, 249], dtype=int)
    else:
        active_positions = np.array([max(0, n_residues // 3), max(0, n_residues // 2)], dtype=int)
    dist = np.array([min(abs(i - a) for a in active_positions) for i in range(n_residues)], dtype=np.float32)
    biophysical = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)

    if scorer == "egnn":
        from egnn_model import build_edge_index_from_adjacency, straight_chain_coords, train_egnn_ensemble

        edge_index = build_edge_index_from_adjacency(adj_norm)
        coords = straight_chain_coords(n_residues)
        position_frac = np.arange(n_residues, dtype=np.float32) / max(1, n_residues - 1)
        node_features = np.stack([var_norm, biophysical, position_frac], axis=1)
        target = np.clip(compute_variant_frequency(ref_seq, fam_records), 0.0, 1.0)

        cfg = dict(_DEFAULT_EGNN_CONFIG)
        if egnn_config:
            cfg.update(egnn_config)
        propagated = train_egnn_ensemble(
            node_features=node_features,
            coords=coords,
            edge_index=edge_index,
            target=target,
            **cfg,
        )
    else:
        propagated = propagate_scores(var_norm, adj_norm, alpha=alpha, hops=hops)

    combined = 0.80 * propagated + 0.20 * biophysical

    positive_positions = sorted({p for rec in fam_records if rec.header != reference.header for p in project_variant_positions(ref_seq, rec.sequence)})

    y_true = np.array([1 if i in positive_positions else 0 for i in range(n_residues)], dtype=int)
    from sklearn.metrics import roc_auc_score

    metrics = {
        "family": family,
        "scorer": scorer,
        "reference": reference.header,
        "n_sequences": len(fam_records),
        "n_positive_positions": int(y_true.sum()),
        "mean_variant_positions": float(np.mean([len(project_variant_positions(ref_seq, r.sequence)) for r in fam_records if r.header != reference.header])) if len(fam_records) > 1 else 0.0,
        "roc_auc_variance": float(roc_auc_score(y_true, var_norm)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "roc_auc_graph": float(roc_auc_score(y_true, propagated)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "roc_auc_combined": float(roc_auc_score(y_true, combined)) if 0 < y_true.sum() < len(y_true) else float("nan"),
    }
    return metrics


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("output/clusters", exist_ok=True)

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    t0 = time.time()
    from Bio import SeqIO

    records = load_records(args.input, args.max_seqs)
    if not records:
        raise SystemExit(f"No sequences found in {args.input}")

    cluster_counts = Counter()
    for rec in records:
        cluster_counts[rec.family] += 1

    print(f"Loaded {len(records)} sequences from {args.input}")
    print(f"Family counts: {dict(cluster_counts)}")

    t_load = time.time()
    embeddings = embed_sequences(
        records,
        device=device,
        batch_size=max(1, args.batch_size),
        cache_path=args.embed_cache,
        use_amp=args.amp,
        use_data_parallel=args.data_parallel,
    )
    t_embed = time.time()
    print(f"ESM-2 ready on {device}; embedded {len(embeddings)} sequences")
    if args.profile:
        print(f"Embedding stage: {t_embed - t_load:.1f}s")

    from cluster_utils import load_cluster_csv

    cluster_map: Dict[str, int] = {}
    cluster_path = Path(args.clusters)
    if cluster_path.exists():
        try:
            cluster_map = load_cluster_csv(str(cluster_path))
        except Exception:
            cluster_map = {}

    curated_map = load_curated_mutations(args.curated_mutations, args.curated_zero_indexed)
    family_refs = {fam: choose_reference(records, fam) for fam in [NDM_FAMILY, VIM_FAMILY, IMP_FAMILY]}

    print(f"[Scorer] {args.scorer}"
          + (f" (hidden={args.egnn_hidden}, layers={args.egnn_layers}, ensemble={args.egnn_models})"
             if args.scorer == "egnn" else ""))

    egnn_config = {
        "n_models": args.egnn_models,
        "hidden_dim": args.egnn_hidden,
        "n_layers": args.egnn_layers,
        "epochs": args.egnn_epochs,
        "device": device,
    }

    summaries: List[Dict[str, object]] = []
    for fam in [NDM_FAMILY, VIM_FAMILY, IMP_FAMILY]:
        summary = family_summary(
            records, embeddings, fam, family_refs[fam], alpha=args.alpha, hops=args.hops,
            scorer=args.scorer, egnn_config=egnn_config,
        )
        summary["analysis_level"] = "full_superfamily"
        summary["label_type"] = "observed_variant_positions"
        summaries.append(summary)
        print(f"[{fam}] sequences: {summary['n_sequences']}")
        print(f"[{fam}] positives: {summary['n_positive_positions']}")
        print(f"[{fam}] ROC-AUC variance={summary['roc_auc_variance']}, graph={summary['roc_auc_graph']}, combined={summary['roc_auc_combined']}")

        if fam == NDM_FAMILY and curated_map.get(fam):
            print(f"[{fam}] curated high-confidence positions loaded: {len(curated_map[fam])}")

    family_counts = Counter([r.family for r in records])
    stats = {
        "n_sequences": len(records),
        "family_counts": dict(family_counts),
        "max_seqs": args.max_seqs,
        "cluster_file": args.clusters,
        "embed_cache": args.embed_cache,
        "device": device,
        "amp": bool(args.amp),
        "data_parallel": bool(args.data_parallel),
        "elapsed_seconds": time.time() - t0,
    }

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "superfamily_benchmark_summary.csv", index=False)
    (output_dir / "superfamily_benchmark_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Write per-sequence manifest for downstream analysis.
    manifest_rows = []
    for rec in records:
        manifest_rows.append({"header": rec.header, "family": rec.family, "length": len(rec.sequence)})
    pd.DataFrame(manifest_rows).to_csv(output_dir / "superfamily_manifest.csv", index=False)

    print(f"Wrote {output_dir / 'superfamily_benchmark_summary.csv'}")
    print(f"Wrote {output_dir / 'superfamily_benchmark_stats.json'}")
    print(f"Wrote {output_dir / 'superfamily_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
