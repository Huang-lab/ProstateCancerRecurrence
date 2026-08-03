"""
Collect ELLA results and produce summary tables and plots.

Reads per-gene estimation JSON files, applies BH FDR correction,
and saves:
  - significant_genes.csv  (FDR ≤ 0.05)
  - all_genes_results.csv  (all genes with p-values)
  - subcellular_pattern_summary.png (barplot of pattern types)

Usage:
    python3 04_postprocess.py --sample B408
    python3 04_postprocess.py --sample B573
    python3 04_postprocess.py --sample both
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# ─────────────────────────────────────────────────────────────────────────────
OUT_ROOT = "/Users/wangz21/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Huang_lab/manuscripts/WTC_PRAD_VisiumHD_Recurrence/analysis/SubcellularExpressionPatern"


def get_paths(sample: str):
    out = os.path.join(OUT_ROOT, sample)
    return {
        "results_dir": os.path.join(out, "ella_results", "lightning_logs", "run1"),
        "prepared_data": os.path.join(out, "prepared_data"),
        "out_dir": out,
    }


def load_gene_names(paths: dict):
    """Read gene names in order from training_data.jsonl."""
    genes = []
    td = os.path.join(paths["prepared_data"], "training_data.jsonl")
    if not os.path.exists(td):
        return None
    with open(td) as f:
        for line in f:
            obj = json.loads(line)
            genes.append(obj["gene_id"])
    return genes


def collect_results(sample: str):
    paths = get_paths(sample)
    results_dir = paths["results_dir"]

    if not os.path.isdir(results_dir):
        print(f"  [skip] No results directory: {results_dir}")
        return None

    gene_names = load_gene_names(paths)

    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_estimation_result.json"):
            continue
        gene_idx = int(fname.split("_")[1])
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath) as f:
                res = json.load(f)
        except Exception:
            continue

        gene_name = gene_names[gene_idx] if gene_names and gene_idx < len(gene_names) else f"gene_{gene_idx}"
        rows.append({
            "gene_idx":     gene_idx,
            "gene":         gene_name,
            "p_cauchy":     res.get("p_cauchy", np.nan),
            "pattern_type": res.get("pattern_type", "unknown"),
            "pattern_score": res.get("pattern_score", np.nan),
        })

    if not rows:
        print(f"  No estimation result files found in {results_dir}")
        return None

    df = pd.DataFrame(rows).sort_values("gene_idx").reset_index(drop=True)
    print(f"  Collected {len(df)} gene results")

    # FDR correction
    valid = df["p_cauchy"].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(
            df.loc[valid, "p_cauchy"], alpha=0.05, method="fdr_bh"
        )
        df.loc[valid, "p_fdr"] = p_fdr
        df.loc[valid, "significant"] = reject
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False

    df["significant"] = df["significant"].fillna(False)
    n_sig = df["significant"].sum()
    print(f"  Significant genes (FDR ≤ 0.05): {n_sig}")

    return df


def save_results(sample: str, df: pd.DataFrame):
    paths = get_paths(sample)
    out_dir = paths["out_dir"]

    # All genes
    all_path = os.path.join(out_dir, "all_genes_results.csv")
    df.to_csv(all_path, index=False)
    print(f"  All genes → {all_path}")

    # Significant genes
    sig = df[df["significant"]].sort_values("p_fdr")
    sig_path = os.path.join(out_dir, "significant_genes.csv")
    sig.to_csv(sig_path, index=False)
    print(f"  Significant genes → {sig_path}")

    # ── Pattern summary plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # P-value distribution
    ax = axes[0]
    valid_p = df["p_cauchy"].dropna()
    ax.hist(valid_p, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("p-value (Cauchy combination)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{sample}: P-value distribution (n={len(valid_p)} genes)")
    ax.axvline(0.05, color="red", linestyle="--", label="p=0.05")
    ax.legend()

    # Pattern type breakdown for significant genes
    ax = axes[1]
    if sig.empty or sig["pattern_type"].isna().all():
        ax.text(0.5, 0.5, "No significant genes", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        pattern_counts = sig["pattern_type"].value_counts()
        pattern_counts.plot(kind="bar", ax=ax, color="salmon", edgecolor="black")
        ax.set_xlabel("Pattern type")
        ax.set_ylabel("Number of significant genes")
        ax.set_title(f"{sample}: Subcellular patterns (FDR ≤ 0.05, n={n_sig})")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "subcellular_pattern_summary.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary plot → {plot_path}")

    # ── Top 20 significant genes ─────────────────────────────────────────────
    if not sig.empty:
        print(f"\n  Top 20 significant genes ({sample}):")
        print(sig[["gene", "p_fdr", "pattern_type", "pattern_score"]].head(20).to_string(index=False))

    n_sig = int(df["significant"].sum())
    return {"sample": sample, "n_total": len(df), "n_significant": n_sig}


def postprocess_sample(sample: str):
    print(f"\n=== Postprocessing {sample} ===")
    df = collect_results(sample)
    if df is None:
        return
    return save_results(sample, df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["B408", "B573", "both"], default="both")
    args = parser.parse_args()
    samples = ["B408", "B573"] if args.sample == "both" else [args.sample]
    all_stats = []
    for s in samples:
        stats = postprocess_sample(s)
        if stats:
            all_stats.append(stats)

    if len(all_stats) == 2:
        # Combined summary
        combined_path = os.path.join(OUT_ROOT, "combined_summary.json")
        with open(combined_path, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nCombined summary → {combined_path}")


if __name__ == "__main__":
    main()
