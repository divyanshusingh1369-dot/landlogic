#!/usr/bin/env python3
"""Analyze DEM terrain slope and classify land-risk zones."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load a DEM GeoTIFF, compute slope/gradient, classify Green-Yellow-Red "
            "risk zones, and write GeoTIFF/PNG outputs."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input DEM GeoTIFF file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory where slope/risk rasters and map image are saved.",
    )
    return parser.parse_args()


def compute_slope_degrees(dem: np.ndarray, xres: float, yres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute x/y gradients and slope in degrees from DEM values."""
    # np.gradient returns rate of change in elevation per map unit in Y then X direction.
    grad_y, grad_x = np.gradient(dem, yres, xres)

    # Slope magnitude = sqrt((dz/dx)^2 + (dz/dy)^2), then convert to slope angle in degrees.
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_degrees = np.degrees(slope_radians)
    return grad_x, grad_y, slope_degrees


def classify_risk(slope_degrees: np.ndarray) -> np.ndarray:
    """Classify slope into risk zones: 1=Green, 2=Yellow, 3=Red."""
    risk = np.zeros(slope_degrees.shape, dtype=np.uint8)
    risk[slope_degrees <= 5] = 1
    risk[(slope_degrees > 5) & (slope_degrees <= 15)] = 2
    risk[slope_degrees > 15] = 3
    return risk


def save_geotiff(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata: float | int | None) -> None:
    """Save a single-band array as GeoTIFF using source spatial metadata."""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)


def save_risk_png(path: Path, risk_array: np.ndarray) -> None:
    """Save a color-coded PNG map for risk classes."""
    # Class colors: 0=transparent nodata, 1=Green, 2=Yellow, 3=Red.
    cmap = ListedColormap(["#00000000", "#2ECC71", "#F1C40F", "#E74C3C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.figure(figsize=(10, 8))
    plt.imshow(risk_array, cmap=cmap, norm=norm)
    plt.title("Land Risk Zones (Green / Yellow / Red)")
    plt.axis("off")

    # Build a simple legend describing each zone.
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="Green (<= 5°)", markerfacecolor="#2ECC71", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Yellow (> 5° and <= 15°)", markerfacecolor="#F1C40F", markersize=10),
        plt.Line2D([0], [0], marker="s", color="w", label="Red (> 15°)", markerfacecolor="#E74C3C", markersize=10),
    ]
    plt.legend(handles=legend_handles, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    """Run DEM slope analysis workflow and write outputs."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    slope_tif = output_dir / "slope_degrees.tif"
    risk_tif = output_dir / "risk_zones.tif"
    risk_png = output_dir / "risk_zones.png"

    # 1) Read DEM and geospatial metadata from GeoTIFF.
    with rasterio.open(input_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    # 2) Build nodata mask to avoid computing slope on invalid pixels.
    if nodata is not None:
        nodata_mask = dem == nodata
    else:
        nodata_mask = np.isnan(dem)

    dem_for_calc = dem.copy()
    dem_for_calc[nodata_mask] = np.nan

    # 3) Compute gradient and slope in degrees (using pixel resolution from affine transform).
    xres = abs(transform.a)
    yres = abs(transform.e)
    grad_x, grad_y, slope_degrees = compute_slope_degrees(dem_for_calc, xres, yres)

    # 4) Classify each cell into Green/Yellow/Red risk zones based on slope thresholds.
    risk = classify_risk(slope_degrees)

    # Keep nodata cells as nodata in outputs.
    slope_degrees[nodata_mask] = np.nan
    risk[nodata_mask] = 0

    # 5) Save slope raster and risk raster as GeoTIFF outputs.
    save_geotiff(slope_tif, slope_degrees, profile, dtype="float32", nodata=np.nan)
    save_geotiff(risk_tif, risk, profile, dtype="uint8", nodata=0)

    # 6) Save a color-coded PNG map to visualize risk zones.
    save_risk_png(risk_png, risk)

    # 7) Print quick summary paths for the user.
    print(f"Slope GeoTIFF: {slope_tif}")
    print(f"Risk GeoTIFF:  {risk_tif}")
    print(f"Risk PNG map:  {risk_png}")
    print(f"Mean |dz/dx| gradient: {np.nanmean(np.abs(grad_x)):.4f}")
    print(f"Mean |dz/dy| gradient: {np.nanmean(np.abs(grad_y)):.4f}")


if __name__ == "__main__":
    main()
