"""
Cell segmentation on Visium HD CytAssist H&E images.

The CytAssist image is a brightfield H&E scan at ~4.65 µm/pixel.  Nuclei
appear as darker features in the green channel (both hematoxylin and eosin
absorb green light, giving high local contrast).  We use CLAHE contrast
enhancement + Otsu threshold + distance-transform watershed.

Pipeline
--------
1. Load CytAssist RGB image (3000 × 3200 px).
2. Extract the green channel (most variable in H&E tissue).
3. Apply CLAHE to enhance local contrast.
4. Invert → bright nuclei; threshold (Otsu) → binary mask.
5. Morphological cleaning + distance-transform watershed.
6. Save labeled mask array (uint32 NPY, values 0 = background / ≥1 = cell id).

Typical result: ~10,000–15,000 cells per tissue section at ~4.65 µm/pixel.

Usage
-----
    python3 01_cell_segmentation.py --sample B408
    python3 01_cell_segmentation.py --sample B573
    python3 01_cell_segmentation.py --sample both
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/wangz21/wangy33.u.hpc.mssm.edu/10X_Single_Cell_RNA/TD006859_KHuang_T598"
OUT_ROOT  = "/Users/wangz21/Library/CloudStorage/OneDrive-TheMountSinaiHospital/Huang_lab/manuscripts/WTC_PRAD_VisiumHD_Recurrence/analysis/SubcellularExpressionPatern"

SAMPLE_IDS = {
    "B408": "TD006859-B408",
    "B573": "TD006859-B573",
}


def get_paths(sample: str):
    folder = SAMPLE_IDS[sample]
    spatial_dir = os.path.join(DATA_ROOT, folder, "outs", "binned_outputs",
                               "square_002um", "spatial")
    out_dir = os.path.join(OUT_ROOT, sample)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return {
        "cytassist":    os.path.join(spatial_dir, "cytassist_image.tiff"),
        "scalefactors": os.path.join(spatial_dir, "scalefactors_json.json"),
        "masks_path":   os.path.join(out_dir, "cellpose_masks.npy"),
        "info_path":    os.path.join(out_dir, "segmentation_info.json"),
        "out_dir":      out_dir,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_sample(
    sample: str,
    clahe_clip_limit: float = 4.0,
    clahe_tile: int = 20,
    min_cell_area: int = 3,       # pixels²
    max_cell_area: int = 250,     # pixels²
    erode_radius: int = 1,
):
    """
    Segment cells from the H&E CytAssist image using CLAHE + watershed.

    Parameters
    ----------
    clahe_clip_limit : float
        CLAHE clip limit (higher → more aggressive local contrast stretch).
    clahe_tile : int
        CLAHE tile size (pixels).  Smaller = finer local contrast.
    min_cell_area : int
        Minimum accepted cell area in pixels² (removes noise blobs).
    max_cell_area : int
        Maximum accepted cell area in pixels² (removes tissue folds/artifacts).
    erode_radius : int
        Morphological erosion radius to separate touching cells before seeding.
    """
    paths = get_paths(sample)
    print(f"\n=== Segmenting {sample} ===")

    with open(paths["scalefactors"]) as f:
        sf = json.load(f)
    um_per_px = sf["microns_per_pixel"]
    print(f"  Scale: {um_per_px:.3f} µm/pixel")
    print(f"  Min cell area: {min_cell_area} px² ({min_cell_area*um_per_px**2:.0f} µm²)")
    print(f"  Max cell area: {max_cell_area} px² ({max_cell_area*um_per_px**2:.0f} µm²)")

    # ── 1. Load image ────────────────────────────────────────────────────────
    print("  Loading CytAssist image …")
    img = tifffile.imread(paths["cytassist"])   # (H, W, 3)  uint8
    H, W = img.shape[:2]
    print(f"  Shape: {img.shape}")

    # ── 2. Green channel + CLAHE ──────────────────────────────────────────────
    # In H&E, both hematoxylin and eosin absorb green light → high local
    # contrast in G channel even when global contrast is low.
    g = img[:, :, 1]                            # uint8
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit,
                             tileGridSize=(clahe_tile, clahe_tile))
    g_enh = clahe.apply(g)                      # uint8, locally stretched

    # ── 3. Invert + threshold ────────────────────────────────────────────────
    # Nuclei are darker → appear bright after inversion
    g_inv   = (255 - g_enh).astype(np.uint8)
    thresh  = filters.threshold_otsu(g_inv)
    binary  = g_inv > thresh
    print(f"  Otsu threshold: {thresh},  nucleus pixels: {binary.sum()} ({binary.mean()*100:.1f}%)")

    # ── 4. Clean binary mask ──────────────────────────────────────────────────
    binary = morphology.remove_small_objects(binary, min_size=min_cell_area + 1)
    binary = ndi.binary_fill_holes(binary)

    # ── 5. Watershed seeding ─────────────────────────────────────────────────
    if erode_radius > 0:
        eroded = morphology.erosion(binary, morphology.disk(erode_radius))
    else:
        eroded = binary

    distance = ndi.distance_transform_edt(eroded)
    peaks    = morphology.local_maxima(distance)
    seeds, n_seeds = ndi.label(peaks)
    print(f"  Watershed seeds: {n_seeds}")

    # ── 6. Watershed ─────────────────────────────────────────────────────────
    labels = segmentation.watershed(-distance, seeds, mask=binary, compactness=0.001)

    # ── 7. Filter by area ─────────────────────────────────────────────────────
    props    = measure.regionprops(labels)
    keep_ids = {p.label for p in props
                if min_cell_area <= p.area <= max_cell_area}
    print(f"  Cells before size filter: {len(props)},  after: {len(keep_ids)}")

    # Re-label
    final = np.zeros((H, W), dtype=np.uint32)
    for new_id, old_id in enumerate(sorted(keep_ids), start=1):
        final[labels == old_id] = new_id

    n_cells = int(final.max())
    areas   = [p.area for p in props if p.label in keep_ids]
    med_area_um2 = np.median(areas) * um_per_px**2
    print(f"  Final cell count: {n_cells}")
    print(f"  Median cell area: {np.median(areas):.1f} px² ({med_area_um2:.0f} µm²)")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    np.save(paths["masks_path"], final)
    print(f"  Saved masks → {paths['masks_path']}")

    info = {
        "sample":           sample,
        "n_cells":          n_cells,
        "microns_per_pixel": um_per_px,
        "image_shape":      list(img.shape),
        "otsu_threshold":   int(thresh),
        "nucleus_pct":      float(binary.mean() * 100),
        "median_area_px2":  float(np.median(areas)),
        "median_area_um2":  float(med_area_um2),
        "clahe_clip":       clahe_clip_limit,
        "clahe_tile":       clahe_tile,
        "min_cell_area":    min_cell_area,
        "max_cell_area":    max_cell_area,
        "erode_radius":     erode_radius,
    }
    with open(paths["info_path"], "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved info → {paths['info_path']}")
    return final, info


def main():
    parser = argparse.ArgumentParser(
        description="H&E watershed cell segmentation for Visium HD"
    )
    parser.add_argument("--sample", choices=["B408", "B573", "both"], default="both")
    parser.add_argument("--clahe-clip", type=float, default=4.0,
                        help="CLAHE clip limit (default: 4.0)")
    parser.add_argument("--clahe-tile", type=int, default=20,
                        help="CLAHE tile size in pixels (default: 20)")
    parser.add_argument("--min-area", type=int, default=3,
                        help="Min cell area in pixels² (default: 3)")
    parser.add_argument("--max-area", type=int, default=250,
                        help="Max cell area in pixels² (default: 250)")
    parser.add_argument("--erode-radius", type=int, default=1,
                        help="Erosion radius for seed separation (default: 1)")
    args = parser.parse_args()

    samples = ["B408", "B573"] if args.sample == "both" else [args.sample]
    for s in samples:
        segment_sample(
            s,
            clahe_clip_limit=args.clahe_clip,
            clahe_tile=args.clahe_tile,
            min_cell_area=args.min_area,
            max_cell_area=args.max_area,
            erode_radius=args.erode_radius,
        )


if __name__ == "__main__":
    main()
