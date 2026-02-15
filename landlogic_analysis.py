#!/usr/bin/env python3
"""Analyze DEM terrain slope and classify land-risk zones."""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

LOGGER = logging.getLogger("landlogic")


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DEM slope analysis, risk classification, and map export"
    )
    parser.add_argument("--input", required=True, help="Path to input DEM GeoTIFF.")
    parser.add_argument("--output", required=True, help="Output directory for result files.")
    parser.add_argument(
        "--green-max",
        type=float,
        default=5.0,
        help="Maximum slope (degrees) for Green risk class (default: 5).",
    )
    parser.add_argument(
        "--yellow-max",
        type=float,
        default=15.0,
        help="Maximum slope (degrees) for Yellow risk class (default: 15).",
    )
    parser.add_argument(
        "--target-crs",
        default="auto",
        help="Target CRS for reprojection. Use 'auto' for dynamic UTM (default) or pass EPSG code like EPSG:32643.",
    )
    parser.add_argument(
        "--skip-reproject",
        action="store_true",
        help="Skip reprojection and use DEM native CRS/resolution.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parsed = parser.parse_args(args)

    if parsed.green_max < 0:
        parser.error("--green-max must be non-negative.")
    if parsed.yellow_max <= parsed.green_max:
        parser.error("--yellow-max must be greater than --green-max.")

    return parsed


def _utm_crs_for_bounds(bounds, src_crs: CRS) -> CRS:
    """Compute a UTM CRS from raster bounds/centroid in geographic coordinates."""
    if src_crs is None:
        raise ValueError("Input raster CRS is missing; cannot infer UTM zone.")

    wgs84 = CRS.from_epsg(4326)
    if src_crs == wgs84:
        lon = (bounds.left + bounds.right) / 2.0
        lat = (bounds.bottom + bounds.top) / 2.0
    else:
        from rasterio.warp import transform_bounds

        b = transform_bounds(src_crs, wgs84, bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
        lon = (b[0] + b[2]) / 2.0
        lat = (b[1] + b[3]) / 2.0

    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def reproject_dem(src_path: Path, dst_path: Path, target_crs: str) -> Path:
    """Reproject DEM if needed. Returns path to DEM to use for analysis."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Input DEM has no CRS; cannot reproject.")

        if target_crs.lower() == "auto":
            if src.crs.is_projected:
                LOGGER.info("DEM already projected (%s). Skipping reprojection.", src.crs)
                return src_path
            dst_crs = _utm_crs_for_bounds(src.bounds, src.crs)
        else:
            dst_crs = CRS.from_string(target_crs)

        if src.crs == dst_crs:
            LOGGER.info("Source CRS already matches target CRS (%s). Skipping reprojection.", dst_crs)
            return src_path

        LOGGER.info("Reprojecting DEM from %s to %s", src.crs, dst_crs)
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "compress": "deflate",
            }
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return dst_path


def compute_slope_degrees(dem: np.ndarray, xres: float, yres: float):
    """Compute x/y gradients and slope in degrees from DEM values."""
    grad_y, grad_x = np.gradient(dem, yres, xres)
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    slope_degrees = np.degrees(slope_radians)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return grad_x, grad_y, grad_mag, slope_degrees


def classify_risk(slope_degrees: np.ndarray, green_max: float, yellow_max: float) -> np.ndarray:
    """Classify slope into risk zones: 1=Green, 2=Yellow, 3=Red."""
    risk = np.zeros(slope_degrees.shape, dtype=np.uint8)
    risk[slope_degrees <= green_max] = 1
    risk[(slope_degrees > green_max) & (slope_degrees <= yellow_max)] = 2
    risk[slope_degrees > yellow_max] = 3
    return risk


def save_geotiff(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata):
    """Save one raster band as a GeoTIFF with updated metadata."""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)


def save_risk_png(path: Path, risk_array: np.ndarray, green_max: float, yellow_max: float):
    """Save color-coded risk map PNG with grid and summary panel."""
    import matplotlib.pyplot as plt

    cmap = ListedColormap(["#00000000", "#2ECC71", "#F1C40F", "#E74C3C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.2])

    ax_map = fig.add_subplot(gs[0])
    ax_map.imshow(risk_array, cmap=cmap, norm=norm)
    ax_map.set_title("Land Risk Zones")

    rows, cols = risk_array.shape
    step = max(1, min(rows, cols) // 20)
    ax_map.set_xticks(np.arange(0, cols, step))
    ax_map.set_yticks(np.arange(0, rows, step))
    ax_map.grid(color="white", linestyle="--", linewidth=0.4, alpha=0.6)
    ax_map.tick_params(labelbottom=False, labelleft=False, length=0)

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
        f"Green (<= {green_max:.1f}°): {green:,} ({pct(green):.1f}%)    "
        f"Yellow ({green_max:.1f}°–{yellow_max:.1f}°): {yellow:,} ({pct(yellow):.1f}%)    "
        f"Red (> {yellow_max:.1f}°): {red:,} ({pct(red):.1f}%)"
    )
    ax_data.text(0.01, 0.55, summary_text, fontsize=11, family="monospace")

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main(args=None):
    """Main processing pipeline."""
    parsed = parse_args(args)
    logging.basicConfig(level=getattr(logging, parsed.log_level), format="%(levelname)s: %(message)s")

    input_path = Path(parsed.input)
    output_dir = Path(parsed.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input DEM not found: {input_path}")

    if parsed.skip_reproject:
        analysis_dem_path = input_path
        LOGGER.info("Skipping reprojection by request.")
    else:
        reproj_path = output_dir / "reprojected_dem.tif"
        analysis_dem_path = reproject_dem(input_path, reproj_path, parsed.target_crs)

    with rasterio.open(analysis_dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    if dem.size == 0:
        raise ValueError("Input DEM contains no cells.")

    if nodata is not None:
        if np.issubdtype(dem.dtype, np.floating):
            nodata_mask = np.isclose(dem, nodata, equal_nan=True)
        else:
            nodata_mask = dem == nodata
    else:
        nodata_mask = np.isnan(dem)

    if np.all(nodata_mask):
        raise ValueError("All DEM cells are nodata; cannot compute slope.")

    dem_calc = dem.copy()
    dem_calc[nodata_mask] = np.nan

    xres = abs(transform.a)
    yres = abs(transform.e)
    if xres == 0 or yres == 0:
        raise ValueError(f"Invalid raster resolution xres={xres}, yres={yres}")

    grad_x, grad_y, grad_mag, slope_degrees = compute_slope_degrees(dem_calc, xres, yres)
    risk = classify_risk(slope_degrees, parsed.green_max, parsed.yellow_max)

    slope_degrees[nodata_mask] = np.nan
    grad_x[nodata_mask] = np.nan
    grad_y[nodata_mask] = np.nan
    grad_mag[nodata_mask] = np.nan
    risk[nodata_mask] = 0

    slope_tif = output_dir / "slope_degrees.tif"
    risk_tif = output_dir / "risk_zones.tif"
    grad_x_tif = output_dir / "gradient_x.tif"
    grad_y_tif = output_dir / "gradient_y.tif"
    grad_mag_tif = output_dir / "gradient_magnitude.tif"
    risk_png = output_dir / "risk_zones.png"

    save_geotiff(slope_tif, slope_degrees, profile, "float32", np.nan)
    save_geotiff(grad_x_tif, grad_x, profile, "float32", np.nan)
    save_geotiff(grad_y_tif, grad_y, profile, "float32", np.nan)
    save_geotiff(grad_mag_tif, grad_mag, profile, "float32", np.nan)
    save_geotiff(risk_tif, risk, profile, "uint8", 0)

    with rasterio.open(risk_tif, "r+") as risk_dst:
        risk_dst.update_tags(
            1,
            class_1=f"Green <= {parsed.green_max} deg",
            class_2=f"Yellow > {parsed.green_max} and <= {parsed.yellow_max} deg",
            class_3=f"Red > {parsed.yellow_max} deg",
        )

    save_risk_png(risk_png, risk, parsed.green_max, parsed.yellow_max)

    LOGGER.info("Outputs saved to %s", output_dir)
    LOGGER.info("Slope min=%.3f max=%.3f mean=%.3f", np.nanmin(slope_degrees), np.nanmax(slope_degrees), np.nanmean(slope_degrees))
    return slope_degrees, risk


if __name__ == "__main__":
    main()
