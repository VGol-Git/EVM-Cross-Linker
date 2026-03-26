"""
set_ops.py — Active/Passive address set operations (Phase 4).

Computes pairwise and triple set intersections of active and passive
address lists across chains, together with Jaccard similarity scores and
a comparison to the expected random-overlap baseline (which is effectively
zero given the 2^160 EVM address space).

Typical usage
-------------
    from set_ops import build_full_set_intersection_report, save_set_report

    report = build_full_set_intersection_report(
        interim_dir="data/interim",
        window_label="1blk",
        chains=["ethereum", "polygon", "bnb"],
    )
    save_set_report(report, output_dir="outputs/tables", window_label="1blk")
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# EVM addresses are 20 bytes → 2^160 possible values
_ADDRESS_SPACE: int = 2**160


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_address_set(
    csv_path: str | Path,
    address_col: str = "address",
) -> FrozenSet[str]:
    """
    Load a CSV of addresses and return a frozenset of lowercase hex strings.

    Falls back to the first column when *address_col* is absent.
    Returns an empty frozenset if the file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Address CSV not found: %s — returning empty set", path)
        return frozenset()

    df = pd.read_csv(path, dtype=str)
    if address_col not in df.columns:
        address_col = df.columns[0]
        logger.debug("Column 'address' not found in %s; using '%s'", path, address_col)

    return frozenset(
        addr.strip().lower()
        for addr in df[address_col].dropna()
        if addr.strip()
    )


def save_set_report(
    report: Dict,
    output_dir: str | Path,
    window_label: str,
) -> None:
    """
    Persist set-intersection results to CSV files under *output_dir*.

    Files written
    -------------
    - set_intersections_active_{window}.csv
    - set_intersections_passive_{window}.csv
    - set_intersections_mixed_{window}.csv
    - set_intersections_summary_{window}.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(df: pd.DataFrame, name: str) -> None:
        # Drop the raw address lists before saving (large, not needed in CSV)
        cols_to_drop = [c for c in ("intersection_addresses",) if c in df.columns]
        df.drop(columns=cols_to_drop).to_csv(out / name, index=False)

    if "active_pairwise" in report and not report["active_pairwise"].empty:
        _save(
            report["active_pairwise"],
            f"set_intersections_active_{window_label}.csv",
        )
    if "passive_pairwise" in report and not report["passive_pairwise"].empty:
        _save(
            report["passive_pairwise"],
            f"set_intersections_passive_{window_label}.csv",
        )
    if "mixed" in report and not report["mixed"].empty:
        _save(report["mixed"], f"set_intersections_mixed_{window_label}.csv")
    if "summary" in report and not report["summary"].empty:
        report["summary"].to_csv(
            out / f"set_intersections_summary_{window_label}.csv", index=False
        )

    logger.info("Set intersection results saved to %s (window=%s)", out, window_label)


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────


def jaccard(set_a: FrozenSet[str], set_b: FrozenSet[str]) -> float:
    """Jaccard similarity  |A ∩ B| / |A ∪ B|."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def expected_random_jaccard(
    size_a: int,
    size_b: int,
    address_space: int = _ADDRESS_SPACE,
) -> float:
    """
    Expected Jaccard similarity if two sets are drawn uniformly at random
    from *address_space* without replacement.

    Since |A|, |B| ≪ 2^160 the expected intersection is essentially zero:

        E[|A ∩ B|] ≈ |A| · |B| / N
        E[J]       ≈ E[|A ∩ B|] / (|A| + |B| − E[|A ∩ B|])
    """
    if address_space <= 0 or size_a <= 0 or size_b <= 0:
        return 0.0
    exp_inter = size_a * size_b / address_space
    exp_union = size_a + size_b - exp_inter
    return exp_inter / exp_union if exp_union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Set operations
# ─────────────────────────────────────────────────────────────────────────────


def compute_pairwise_intersections(
    address_sets: Dict[str, FrozenSet[str]],
    label: str = "active",
) -> pd.DataFrame:
    """
    Compute all pairwise set intersections and Jaccard scores.

    Parameters
    ----------
    address_sets : mapping  chain_name → frozenset of addresses
    label        : "active" or "passive" (stored in the *set_type* column)

    Returns
    -------
    DataFrame with columns:
        chain_a, chain_b, set_type,
        size_a, size_b, intersection_size, union_size,
        jaccard, expected_random_jaccard, jaccard_enrichment,
        intersection_addresses
    """
    chains = sorted(address_sets)
    rows = []

    for chain_a, chain_b in itertools.combinations(chains, 2):
        set_a = address_sets[chain_a]
        set_b = address_sets[chain_b]
        inter = set_a & set_b
        union = set_a | set_b

        j = len(inter) / len(union) if union else 0.0
        exp_j = expected_random_jaccard(len(set_a), len(set_b))
        enrichment = (j / exp_j) if exp_j > 0 else float("inf")

        rows.append(
            {
                "chain_a": chain_a,
                "chain_b": chain_b,
                "set_type": label,
                "size_a": len(set_a),
                "size_b": len(set_b),
                "intersection_size": len(inter),
                "union_size": len(union),
                "jaccard": round(j, 8),
                "expected_random_jaccard": exp_j,
                "jaccard_enrichment": enrichment,
                "intersection_addresses": sorted(inter),
            }
        )

    return pd.DataFrame(rows)


def compute_triple_intersection(
    address_sets: Dict[str, FrozenSet[str]],
    label: str = "active",
) -> Dict:
    """
    Compute the intersection of all three chains simultaneously.

    Returns an empty dict when fewer than three chains are provided.
    """
    chains = sorted(address_sets)
    if len(chains) < 3:
        logger.warning("Triple intersection requires ≥3 chains; got %d", len(chains))
        return {}

    sets = [address_sets[c] for c in chains]
    triple_inter: FrozenSet[str] = sets[0] & sets[1] & sets[2]
    triple_union: FrozenSet[str] = sets[0] | sets[1] | sets[2]

    j = len(triple_inter) / len(triple_union) if triple_union else 0.0

    return {
        "chains": chains,
        "set_type": label,
        "sizes": {c: len(address_sets[c]) for c in chains},
        "intersection_size": len(triple_inter),
        "union_size": len(triple_union),
        "jaccard": round(j, 8),
        "intersection_addresses": sorted(triple_inter),
    }


def compute_mixed_intersections(
    active_sets: Dict[str, FrozenSet[str]],
    passive_sets: Dict[str, FrozenSet[str]],
) -> pd.DataFrame:
    """
    Mixed set analysis: active on Chain A  ∩  passive on Chain B.

    Covers all ordered pairs (A, B) where A ≠ B, i.e. every
    "active-on-one-chain / passive-on-another" combination.

    Returns
    -------
    DataFrame with columns:
        active_chain, passive_chain, set_type,
        size_active, size_passive,
        intersection_size, union_size,
        jaccard, expected_random_jaccard, jaccard_enrichment,
        intersection_addresses
    """
    chains = sorted(set(active_sets) | set(passive_sets))
    rows = []

    for chain_a, chain_b in itertools.permutations(chains, 2):
        active_a = active_sets.get(chain_a, frozenset())
        passive_b = passive_sets.get(chain_b, frozenset())
        inter = active_a & passive_b
        union = active_a | passive_b

        j = len(inter) / len(union) if union else 0.0
        exp_j = expected_random_jaccard(len(active_a), len(passive_b))

        rows.append(
            {
                "active_chain": chain_a,
                "passive_chain": chain_b,
                "set_type": "active_A_passive_B",
                "size_active": len(active_a),
                "size_passive": len(passive_b),
                "intersection_size": len(inter),
                "union_size": len(union),
                "jaccard": round(j, 8),
                "expected_random_jaccard": exp_j,
                "jaccard_enrichment": (j / exp_j) if exp_j > 0 else float("inf"),
                "intersection_addresses": sorted(inter),
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def build_full_set_intersection_report(
    interim_dir: str | Path,
    window_label: str,
    chains: Optional[List[str]] = None,
) -> Dict:
    """
    Load active/passive CSVs for all chains for a given window, then compute
    all pairwise, triple, and mixed intersections.

    Directory layout expected (produced by Person 1):
        {interim_dir}/{window_label}/{chain}/active_eoa_addresses.csv
        {interim_dir}/{window_label}/{chain}/passive_eoa_addresses.csv

    Parameters
    ----------
    interim_dir  : path to ``data/interim/``
    window_label : e.g. ``"1blk"``, ``"10blk"``, ``"1000blk"``, ``"10000blk"``
    chains       : chain names to include; defaults to
                   ``["ethereum", "polygon", "bnb"]``

    Returns
    -------
    dict with keys:
        "active_pairwise"  – DataFrame of pairwise active-set intersections
        "active_triple"    – dict with triple-intersection stats
        "passive_pairwise" – DataFrame of pairwise passive-set intersections
        "passive_triple"   – dict with triple-intersection stats
        "mixed"            – DataFrame of mixed (active A ∩ passive B) stats
        "summary"          – flat summary DataFrame (one row per comparison)
    """
    interim_dir = Path(interim_dir)
    chains = chains or ["ethereum", "polygon", "bnb"]

    active_sets: Dict[str, FrozenSet[str]] = {}
    passive_sets: Dict[str, FrozenSet[str]] = {}

    for chain in chains:
        chain_dir = interim_dir / window_label / chain
        active_sets[chain] = load_address_set(chain_dir / "active_eoa_addresses.csv")
        passive_sets[chain] = load_address_set(chain_dir / "passive_eoa_addresses.csv")
        logger.info(
            "window=%s  chain=%-10s  active=%d  passive=%d",
            window_label,
            chain,
            len(active_sets[chain]),
            len(passive_sets[chain]),
        )

    active_pairwise = compute_pairwise_intersections(active_sets, label="active")
    passive_pairwise = compute_pairwise_intersections(passive_sets, label="passive")
    active_triple = compute_triple_intersection(active_sets, label="active")
    passive_triple = compute_triple_intersection(passive_sets, label="passive")
    mixed = compute_mixed_intersections(active_sets, passive_sets)

    # ── flat summary table ────────────────────────────────────────────────
    summary_rows: list[Dict] = []

    def _add_pairwise(df: pd.DataFrame, kind: str) -> None:
        for _, row in df.iterrows():
            summary_rows.append(
                {
                    "window": window_label,
                    "comparison": f"{kind}[{row['chain_a']}] ∩ {kind}[{row['chain_b']}]",
                    "set_type": kind,
                    "intersection_size": row["intersection_size"],
                    "jaccard": row["jaccard"],
                    "expected_random_jaccard": row["expected_random_jaccard"],
                    "jaccard_enrichment": row["jaccard_enrichment"],
                }
            )

    _add_pairwise(active_pairwise, "active")
    _add_pairwise(passive_pairwise, "passive")

    for triple, kind in [(active_triple, "active"), (passive_triple, "passive")]:
        if triple:
            ch = triple["chains"]
            summary_rows.append(
                {
                    "window": window_label,
                    "comparison": (
                        f"{kind}[{ch[0]}] ∩ {kind}[{ch[1]}] ∩ {kind}[{ch[2]}]"
                    ),
                    "set_type": kind,
                    "intersection_size": triple["intersection_size"],
                    "jaccard": triple["jaccard"],
                    "expected_random_jaccard": None,
                    "jaccard_enrichment": None,
                }
            )

    for _, row in mixed.iterrows():
        summary_rows.append(
            {
                "window": window_label,
                "comparison": (
                    f"active[{row['active_chain']}] ∩ passive[{row['passive_chain']}]"
                ),
                "set_type": "mixed",
                "intersection_size": row["intersection_size"],
                "jaccard": row["jaccard"],
                "expected_random_jaccard": row["expected_random_jaccard"],
                "jaccard_enrichment": row["jaccard_enrichment"],
            }
        )

    return {
        "active_pairwise": active_pairwise,
        "active_triple": active_triple,
        "passive_pairwise": passive_pairwise,
        "passive_triple": passive_triple,
        "mixed": mixed,
        "summary": pd.DataFrame(summary_rows),
    }


def build_cross_window_set_summary(
    reports: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Combine summary DataFrames from multiple windows into one cross-window table.

    Parameters
    ----------
    reports : dict mapping window_label → report dict
              (each report is the return value of build_full_set_intersection_report)

    Returns
    -------
    DataFrame sorted by window then comparison type, with Jaccard and enrichment.
    """
    frames = []
    for window_label, report in reports.items():
        summary = report.get("summary")
        if summary is not None and not summary.empty:
            frames.append(summary)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["window", "comparison"])
