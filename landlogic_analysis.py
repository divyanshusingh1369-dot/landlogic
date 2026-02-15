#!/usr/bin/env python3
"""Analyze DEM terrain slope and classify land-risk zones."""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.warp import calculate_default_transform, reproject, Resampling


# ---------------------------
# ARGUMENT PARSER
# ---------------------------
def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEM slope analysis and land risk classification"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(args)


# ---------------------------
# REPROJECT TO UTM (CRITICAL FIX)
# ---------------------------
def reproject_to_utm(src_path, dst_path):
    with rasterio.open(src_path) as src:
        if src.crs.is_projected:
            print("DEM already projected. Skipping reprojection.")
            return src_path

        print("Reprojecting DEM to UTM...")

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:32643", src.width, src.height, *src.bounds
        )

        profile = src.profile.copy()
        profile.update({
            "crs": "EPSG:32643",
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:32643",
                    resampling=Resampling.bilinear
                )

        return dst_path


# ---------------------------
# SLOPE CALCULATION
# ---------------------------
def compute_slope_degrees(dem, xres, yres):
    grad_y, grad_x = np.gradient(dem, yres, xres)
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_degrees = np.degrees(slope_radians)
    return grad_x, grad_y, slope_degrees


# ---------------------------
# RISK CLASSIFICATION
# ---------------------------
def classify_risk(slope_degrees):
    risk = np.zeros(slope_degrees.shape, dtype=np.uint8)
    risk[slope_degrees <= 5] = 1
    risk[(slope_degrees > 5) & (slope_degrees <= 15)] = 2
    risk[slope_degrees > 15] = 3
    return risk


# ---------------------------
# SAVE FUNCTIONS
# ---------------------------
def save_geotiff(path, array, profile, dtype, nodata):
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)


def save_risk_png(path, risk_array):
    cmap = ListedColormap(["#00000000", "#2ECC71", "#F1C40F", "#E74C3C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # Build a figure with map on top and a small data panel below.
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.2])

    ax_map = fig.add_subplot(gs[0])
    ax_map.imshow(risk_array, cmap=cmap, norm=norm)
    ax_map.set_title("Land Risk Zones")

    # Add grid lines to the PNG visualization.
    rows, cols = risk_array.shape
    step = max(1, min(rows, cols) // 20)
    ax_map.set_xticks(np.arange(0, cols, step))
    ax_map.set_yticks(np.arange(0, rows, step))
    ax_map.grid(color="white", linestyle="--", linewidth=0.4, alpha=0.6)
    ax_map.tick_params(labelbottom=False, labelleft=False, length=0)

    # Compute summary data to display below the map.
    valid = risk_array > 0
    total = int(np.count_nonzero(valid))
    green = int(np.count_nonzero(risk_array == 1))
    yellow = int(np.count_nonzero(risk_array == 2))
    red = int(np.count_nonzero(risk_array == 3))

    def pct(v):
        return (100.0 * v / total) if total else 0.0

    ax_data = fig.add_subplot(gs[1])
    ax_data.axis("off")
    summary_text = (
        f"Cells analyzed: {total:,}    "
        f"Green: {green:,} ({pct(green):.1f}%)    "
        f"Yellow: {yellow:,} ({pct(yellow):.1f}%)    "
        f"Red: {red:,} ({pct(red):.1f}%)"
    )
    ax_data.text(0.01, 0.55, summary_text, fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()



# ---------------------------
# MAIN ENGINE
# ---------------------------
def main():
    args = parse_args(sys.argv[1:])
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto reprojection fix
    reprojected_path = output_dir / "reprojected_dem.tif"
    input_path = Path(reproject_to_utm(input_path, reprojected_path))

    with rasterio.open(input_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        nodata_mask = dem == nodata
    else:
        nodata_mask = np.isnan(dem)

    dem[nodata_mask] = np.nan

    xres = abs(transform.a)
    yres = abs(transform.e)

    grad_x, grad_y, slope_degrees = compute_slope_degrees(dem, xres, yres)

    risk = classify_risk(slope_degrees)

    slope_degrees[nodata_mask] = np.nan
    risk[nodata_mask] = 0

    slope_tif = output_dir / "slope_degrees.tif"
    risk_tif = output_dir / "risk_zones.tif"
    risk_png = output_dir / "risk_zones.png"

    save_geotiff(slope_tif, slope_degrees, profile, "float32", np.nan)
    save_geotiff(risk_tif, risk, profile, "uint8", 0)
    save_risk_png(risk_png, risk)

    print("Outputs saved.")
    print("Min slope:", np.nanmin(slope_degrees))
    print("Max slope:", np.nanmax(slope_degrees))
    print("Mean slope:", np.nanmean(slope_degrees))

    return slope_degrees, risk


# ---------------------------
# COLAB EXECUTION BLOCK
# ---------------------------
if __name__ == "__main__":
    sys.argv = [
        "landlogic_analysis.py",
        "--input", "/content/YOUR_DEM_FILENAME.tif",
        "--output", "/content/results"
    ]
    slope_degrees, risk = main()
