"""
Prepare ELLA input from Visium HD 2µm binned expression data + watershed masks.

Memory-efficient implementation:
  - Loads sparse matrix and only densifies the bins assigned to cells
  - Uses vectorized operations where possible

For each segmented cell:
  - Cell boundary pixels    → cell_seg DataFrame
  - 2µm bin centers in cell → points in expr DataFrame

The ELLA input pkl contains:
  expr      DataFrame  (cell, type, gene, x, y, umi, centerX, centerY, sc_total)
  cell_seg  DataFrame  (cell, x, y)

Usage:
    python3 02_prepare_ella_input.py --sample B408
    python3 02_prepare_ella_input.py --sample B573 --min-bins 20 --min-umi 50
"""

import argparse
import gc
import json
import os
import pickle
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from skimage import measure

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/wangz21/wangy33.u.hpc.mssm.edu/10X_Single_Cell_RNA/TD006859_KHuang_T598"
OUT_ROOT  = "/Users/wangz21/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Huang_lab/manuscripts/WTC_PRAD_VisiumHD_Recurrence/analysis/SubcellularExpressionPatern"

SAMPLE_IDS = {
    "B408": "TD006859-B408",
    "B573": "TD006859-B573",
}

DEFAULT_MIN_BINS   = 10
DEFAULT_MIN_UMI    = 30
DEFAULT_MAX_CELLS  = 3000
DEFAULT_TOP_GENES  = 1000


def get_paths(sample: str):
    folder = SAMPLE_IDS[sample]
    base   = os.path.join(DATA_ROOT, folder, "outs")
    s2um   = os.path.join(base, "binned_outputs", "square_002um")
    out    = os.path.join(OUT_ROOT, sample)
    Path(out).mkdir(parents=True, exist_ok=True)
    return {
        "tissue_positions": os.path.join(s2um, "spatial", "tissue_positions.parquet"),
        "expr_h5":          os.path.join(s2um, "filtered_feature_bc_matrix.h5"),
        "masks_npy":        os.path.join(out, "cellpose_masks.npy"),
        "out_pkl":          os.path.join(out, "ella_input.pkl"),
        "out_stats":        os.path.join(out, "preparation_stats.json"),
        "out_dir":          out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Assign tissue bins to cell masks
# ─────────────────────────────────────────────────────────────────────────────

def assign_bins_to_cells(paths: dict):
    print("  Loading tissue positions …")
    tp = pd.read_parquet(paths["tissue_positions"])
    tp = tp[tp["in_tissue"] == 1].copy()

    masks = np.load(paths["masks_npy"])
    H, W  = masks.shape

    row_idx = np.round(tp["pxl_row_in_fullres"].values).astype(int).clip(0, H - 1)
    col_idx = np.round(tp["pxl_col_in_fullres"].values).astype(int).clip(0, W - 1)
    cell_labels = masks[row_idx, col_idx]
    del masks; gc.collect()

    tp = tp.copy()
    tp["row_px"] = row_idx
    tp["col_px"] = col_idx
    tp["cell_id"] = cell_labels

    tp_in_cell = tp[cell_labels > 0].copy()
    print(f"    In-tissue bins: {len(tp)}, in segmented cells: {len(tp_in_cell)}")
    print(f"    Unique cells with bins: {tp_in_cell['cell_id'].nunique()}")
    return tp_in_cell


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Load expression only for bins in cells (memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def load_expression_for_cells(paths: dict, tp_in_cell: pd.DataFrame, top_genes: int):
    print("  Loading expression matrix …")
    with h5py.File(paths["expr_h5"], "r") as f:
        barcodes   = [b.decode() for b in f["matrix/barcodes"][:]]
        gene_names = [g.decode() for g in f["matrix/features/name"][:]]
        data       = f["matrix/data"][:]
        indices    = f["matrix/indices"][:]
        indptr     = f["matrix/indptr"][:]
        shape      = tuple(f["matrix/shape"][:])   # (n_genes, n_barcodes)

    # (n_barcodes, n_genes) sparse matrix
    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    print(f"    Matrix: {X.shape[0]} barcodes × {X.shape[1]} genes")

    # Select top genes by total count
    gene_sums = np.asarray(X.sum(axis=0)).ravel()
    top_idx   = np.argsort(gene_sums)[::-1][:top_genes]
    X         = X[:, top_idx].tocsr()
    genes     = [gene_names[i] for i in top_idx]
    print(f"    Keeping top {len(genes)} genes")

    # Map barcodes to row index
    bc_to_idx = {bc: i for i, bc in enumerate(barcodes)}

    # Get only the rows needed (bins in cells)
    tp_in_cell = tp_in_cell.copy()
    tp_in_cell["expr_idx"] = tp_in_cell["barcode"].map(bc_to_idx)
    valid      = tp_in_cell["expr_idx"].notna()
    tp_in_cell = tp_in_cell[valid].copy()
    tp_in_cell["expr_idx"] = tp_in_cell["expr_idx"].astype(int)

    # Dense submatrix: only the relevant rows (bins in cells)
    needed_rows = tp_in_cell["expr_idx"].values
    X_sub = np.asarray(X[needed_rows, :].todense(), dtype=np.uint32)  # (n_bins_in_cells, top_genes)
    print(f"    Dense submatrix: {X_sub.shape}")

    del X; gc.collect()
    return tp_in_cell, genes, X_sub


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Filter and select cells
# ─────────────────────────────────────────────────────────────────────────────

def select_cells(tp_in_cell: pd.DataFrame, X_sub: np.ndarray,
                 min_bins: int, min_umi: int, max_cells: int):
    # per-row UMI sum (X_sub rows correspond to tp_in_cell rows)
    bin_umi = X_sub.sum(axis=1)  # shape (n_bins,)

    tp_in_cell = tp_in_cell.copy()
    tp_in_cell["bin_umi"] = bin_umi

    # per-cell bin count and total UMI
    cell_stats = tp_in_cell.groupby("cell_id").agg(
        n_bins=("barcode", "size"),
        total_umi=("bin_umi", "sum"),
    ).reset_index()

    valid_mask = (cell_stats["n_bins"] >= min_bins) & (cell_stats["total_umi"] >= min_umi)
    valid_ids  = set(cell_stats.loc[valid_mask, "cell_id"].values)
    print(f"    Cells ≥{min_bins} bins & ≥{min_umi} UMI: {len(valid_ids)}")

    if len(valid_ids) > max_cells:
        # Keep cells with most bins
        keep = (cell_stats[cell_stats["cell_id"].isin(valid_ids)]
                .nlargest(max_cells, "n_bins")["cell_id"].values)
        valid_ids = set(keep)
        print(f"    Capping to {max_cells} cells")

    return sorted(valid_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Extract cell boundaries from mask
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundaries(valid_cell_ids: list, paths: dict):
    print(f"  Extracting boundaries for {len(valid_cell_ids)} cells …")
    masks = np.load(paths["masks_npy"])

    boundaries = {}
    centroids  = {}
    # Process only the needed cell ids at once
    valid_set  = set(valid_cell_ids)
    props = measure.regionprops(masks)

    for p in props:
        if p.label not in valid_set:
            continue
        cell_id = p.label
        cy, cx = p.centroid   # row=y, col=x

        cell_mask = (masks == p.label)
        from skimage import measure as skm
        contours = skm.find_contours(cell_mask.astype(float), level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        step    = max(1, len(contour) // 150)
        contour = contour[::step]
        # (x=col, y=row)
        bnd = [(float(c[1]), float(c[0])) for c in contour]
        boundaries[cell_id] = bnd
        centroids[cell_id]  = (float(cx), float(cy))

    del masks; gc.collect()
    print(f"    Boundaries extracted: {len(boundaries)}")
    return boundaries, centroids


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Build ELLA DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframes(tp_in_cell: pd.DataFrame, X_sub: np.ndarray, genes: list,
                     valid_cells: list, boundaries: dict, centroids: dict):
    print("  Building ELLA DataFrames …")
    valid_set = set(valid_cells)

    subset = tp_in_cell[tp_in_cell["cell_id"].isin(valid_set)].copy()
    subset = subset.reset_index(drop=True)

    # Reindex X_sub rows to subset indices
    # tp_in_cell.index maps to X_sub rows
    sub_row_indices = subset.index.values   # positions in tp_in_cell
    X_cell = X_sub[sub_row_indices]         # (n_cell_bins, n_genes)

    genes_arr = np.array(genes)

    # Build cell_seg rows (boundaries)
    seg_rows = [
        {"cell": str(cid), "x": bx, "y": by}
        for cid in valid_cells
        for (bx, by) in boundaries[cid]
    ]
    df_cell_seg = pd.DataFrame(seg_rows)
    print(f"    cell_seg rows: {len(df_cell_seg)}")

    # Compute per-cell UMI totals for sc_total column
    cell_umi_total = {}
    for cid, grp in subset.groupby("cell_id"):
        rows_in_grp = grp.index.values
        cell_umi_total[cid] = int(X_cell[rows_in_grp].sum())

    # Build expr rows using vectorised approach
    expr_rows = []
    for cid, grp in subset.groupby("cell_id"):
        if cid not in boundaries:
            continue
        cx, cy   = centroids[cid]
        sc_total = cell_umi_total[cid]
        rows_in  = grp.index.values

        for local_i, global_i in enumerate(rows_in):
            bx = float(grp.at[global_i, "col_px"])   # x = col
            by = float(grp.at[global_i, "row_px"])   # y = row
            counts = X_cell[global_i]                  # shape (n_genes,)
            nz = np.where(counts > 0)[0]
            if len(nz) == 0:
                continue
            for gi in nz:
                expr_rows.append((
                    "epithelial",
                    str(cid),
                    genes_arr[gi],
                    bx, by,
                    int(counts[gi]),
                    int(round(cx)),
                    int(round(cy)),
                    sc_total,
                ))

    df_expr = pd.DataFrame(
        expr_rows,
        columns=["type", "cell", "gene", "x", "y", "umi", "centerX", "centerY", "sc_total"]
    )
    df_expr["cell"] = pd.Categorical(df_expr["cell"])
    df_expr["gene"] = pd.Categorical(df_expr["gene"])
    print(f"    expr rows: {len(df_expr)}, unique genes: {df_expr['gene'].nunique()}")
    return df_expr, df_cell_seg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def prepare_sample(sample: str, min_bins: int, min_umi: int,
                   max_cells: int, top_genes: int):
    paths = get_paths(sample)
    print(f"\n=== Preparing ELLA input for {sample} ===")

    if not os.path.exists(paths["masks_npy"]):
        raise FileNotFoundError(f"Masks not found: {paths['masks_npy']}")

    # 1. Assign bins to cells
    tp_in_cell = assign_bins_to_cells(paths)

    # 2. Load expression for those bins only
    tp_in_cell, genes, X_sub = load_expression_for_cells(paths, tp_in_cell, top_genes)

    # 3. Select cells passing filters
    valid_cells = select_cells(tp_in_cell, X_sub, min_bins, min_umi, max_cells)
    if not valid_cells:
        raise ValueError("No cells passed filters.")
    print(f"  Final cells: {len(valid_cells)}")

    # 4. Extract cell boundaries
    boundaries, centroids = extract_boundaries(valid_cells, paths)

    # Keep only cells that have boundaries
    valid_cells = [c for c in valid_cells if c in boundaries]
    print(f"  Cells with boundaries: {len(valid_cells)}")

    # 5. Build DataFrames
    df_expr, df_cell_seg = build_dataframes(
        tp_in_cell, X_sub, genes, valid_cells, boundaries, centroids
    )

    # 6. Assemble ELLA pkl
    cell_type = "epithelial"
    ella_data = {
        "types":       [cell_type],
        "cells":       {cell_type: [str(c) for c in valid_cells]},
        "cells_all":   [str(c) for c in valid_cells],
        "genes":       {cell_type: sorted(df_expr["gene"].cat.categories.tolist())},
        "cell_poly":   {str(cid): np.array(pts, dtype=float) for cid, pts in boundaries.items()},
        "cell_seg":    df_cell_seg,
        "nucleus_seg": pd.DataFrame(columns=["cell", "x", "y"]),
        "expr":        df_expr,
    }
    with open(paths["out_pkl"], "wb") as f:
        pickle.dump(ella_data, f, protocol=4)
    print(f"  Saved → {paths['out_pkl']}")

    stats = {
        "sample": sample,
        "n_cells": len(valid_cells),
        "n_genes": df_expr["gene"].nunique(),
        "n_expr_rows": len(df_expr),
    }
    with open(paths["out_stats"], "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["B408", "B573", "both"], default="both")
    parser.add_argument("--min-bins", type=int, default=DEFAULT_MIN_BINS)
    parser.add_argument("--min-umi",  type=int, default=DEFAULT_MIN_UMI)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    parser.add_argument("--top-genes", type=int, default=DEFAULT_TOP_GENES)
    args = parser.parse_args()

    samples = ["B408", "B573"] if args.sample == "both" else [args.sample]
    for s in samples:
        prepare_sample(s, args.min_bins, args.min_umi, args.max_cells, args.top_genes)


if __name__ == "__main__":
    main()
