import sys
import re
import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


DEFAULT_CENTERS_CSV = Path(
    "/Users/ketankotla/Desktop/BarnhartLab/White_Noise_Analysis/all_Mi1/centers_and_contiguous_sizes.csv"
)
SMOOTH_WINDOW = 5
TIME_START_S = -1.5
TIME_END_S = 0.5


def _find_history_tifs(measurements_dir):
    return sorted(measurements_dir.glob("*-dff-weighted-history.tif"))


def _parse_output_and_roi(history_tif):
    match = re.match(r"^(?P<output>.+)-roi_(?P<roi>\d+)-dff-weighted-history$", history_tif.stem)
    if not match:
        return None, None
    return match.group("output"), int(match.group("roi"))


def _extract_center_trf(history_3d, center_rc):
    # Upcast before reductions so low-precision TIFF stacks do not overflow in sum/mean/std.
    history64 = np.asarray(history_3d, dtype=np.float64)
    mean = float(np.mean(history64, dtype=np.float64))
    std = float(np.std(history64, dtype=np.float64))
    z = (history64 - mean) / std if std != 0 else (history64 - mean) / 1e-8

    r, c = int(center_rc[0]), int(center_rc[1])
    if r < 0 or c < 0 or r >= z.shape[1] or c >= z.shape[2]:
        raise ValueError(f"Center {center_rc} is out of bounds for history shape {z.shape}")

    center_rc = (r, c)

    return z[:, center_rc[0], center_rc[1]].astype(np.float32)


def _load_centers_map(centers_csv_path):
    centers_map = {}
    with centers_csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            file_path = row.get("file", "")
            center_row = row.get("center_row")
            center_col = row.get("center_col")
            if not file_path or center_row is None or center_col is None:
                continue

            try:
                r = int(float(center_row))
                c = int(float(center_col))
            except ValueError:
                continue

            stem = Path(file_path).stem
            centers_map[stem] = (r, c)

    return centers_map


def load_ignore_list(csv_path):
    """
    Load ignore list from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: directory, roi_number
        
    Returns:
        Set of tuples (directory, roi_number) to skip
    """
    ignore_set = set()
    if csv_path is None:
        return ignore_set
    
    try:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"Warning: Ignore CSV file does not exist: {csv_path}")
            return ignore_set
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            print(f"Loading ignore list from {csv_path}")
            for row in reader:
                if len(row) >= 2:
                    directory = row[0].strip()
                    roi_number = row[1].strip()
                    ignore_set.add((directory, roi_number))
                    print(f"  Added: directory='{directory}', roi_number='{roi_number}'")
        print(f"Loaded {len(ignore_set)} entries to ignore")
    except Exception as e:
        print(f"Warning: Could not load ignore list from {csv_path}: {e}")
    
    return ignore_set


def should_skip_roi(directory_name, roi_number, ignore_set):
    """
    Check if a ROI should be skipped based on directory and ROI number.
    
    Args:
        directory_name: Name of the sample directory
        roi_number: ROI number (as int)
        ignore_set: Set of tuples (directory, roi_number_str) to skip
        
    Returns:
        True if this ROI should be skipped, False otherwise
    """
    roi_str = str(roi_number)
    return (directory_name, roi_str) in ignore_set


def _save_roi_trf(trf_dir, output_name, roi, trf):
    trf_dir.mkdir(parents=True, exist_ok=True)

    csv_path = trf_dir / f"{output_name}-roi_{int(roi)}-trf.csv"
    png_path = trf_dir / f"{output_name}-roi_{int(roi)}-trf.png"

    time_s = np.linspace(TIME_START_S, TIME_END_S, num=len(trf), endpoint=False, dtype=np.float64)
    out = np.column_stack((time_s, trf))
    np.savetxt(str(csv_path), out, delimiter=",", header="time_s,trf", comments="")

    plt.figure()
    plt.plot(time_s, trf)
    plt.xlim(TIME_START_S, TIME_END_S)
    plt.xlabel("time (s)")
    plt.ylabel("trf")
    plt.title(f"ROI {int(roi)}")
    plt.tight_layout()
    plt.savefig(str(png_path))
    plt.close()

    return csv_path, png_path


def _save_average_trf(trf_dir, output_name, trf, count):
    trf_dir.mkdir(parents=True, exist_ok=True)

    csv_path = trf_dir / f"{output_name}-average-trf.csv"
    png_path = trf_dir / f"{output_name}-average-trf.png"

    time_s = np.linspace(TIME_START_S, TIME_END_S, num=len(trf), endpoint=False, dtype=np.float64)
    out = np.column_stack((time_s, trf))
    np.savetxt(str(csv_path), out, delimiter=",", header="time_s,trf", comments="")

    plt.figure()
    plt.plot(time_s, trf)
    plt.xlim(TIME_START_S, TIME_END_S)
    plt.xlabel("time (s)")
    plt.ylabel("trf")
    plt.title(f"Average TRF (n={count})")
    plt.tight_layout()
    plt.savefig(str(png_path))
    plt.close()

    return csv_path, png_path


def _smooth_trf(trf, window=SMOOTH_WINDOW):
    if window <= 1 or len(trf) < 2:
        return np.asarray(trf, dtype=np.float64)

    w = min(int(window), len(trf))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.asarray(trf, dtype=np.float64), kernel, mode="same")


def process_sample(measurements_dir, centers_map, ignore_set):
    history_tifs = _find_history_tifs(measurements_dir)
    if not history_tifs:
        stats = {
            "total": 0,
            "used": 0,
            "parse_fail": 0,
            "shape_fail": 0,
            "center_missing": 0,
            "roi_ignored": 0,
            "length_mismatch": 0,
        }
        return None, 0, stats

    sample_sum = None
    sample_count = 0
    stats = {
        "total": len(history_tifs),
        "used": 0,
        "parse_fail": 0,
        "shape_fail": 0,
        "center_missing": 0,
        "roi_ignored": 0,
        "length_mismatch": 0,
    }
    
    # Get the directory name for ignore checking
    directory_name = measurements_dir.parent.name

    for history_tif in history_tifs:
        output_name, roi = _parse_output_and_roi(history_tif)
        if output_name is None or roi is None:
            print(f"Skipping {history_tif}: could not parse output/ROI from filename")
            stats["parse_fail"] += 1
            continue

        # Check if this ROI should be ignored
        if should_skip_roi(directory_name, roi, ignore_set):
            print(f"Skipping {history_tif}: ROI in ignore list")
            stats["roi_ignored"] += 1
            continue

        history = tifffile.imread(str(history_tif))
        if history.ndim != 3:
            print(f"Skipping {history_tif}: expected 3D history stack, got shape {history.shape}")
            stats["shape_fail"] += 1
            continue

        center_rc = centers_map.get(history_tif.stem)
        if center_rc is None:
            print(f"Skipping {history_tif}: no center found in centers CSV")
            stats["center_missing"] += 1
            continue

        trf = _extract_center_trf(history, center_rc)

        if sample_sum is None:
            sample_sum = trf.astype(np.float64, copy=True)
            sample_count = 1
        else:
            if len(trf) == len(sample_sum):
                sample_sum += trf
                sample_count += 1
            else:
                print(
                    f"Skipping {history_tif} in global combine: TRF length {len(trf)} does not match {len(sample_sum)}"
                )
                stats["length_mismatch"] += 1

        stats["used"] += 1

    return sample_sum, sample_count, stats


def main(root_dir, centers_csv_path=DEFAULT_CENTERS_CSV, ignore_csv_path=None):
    root = Path(root_dir)
    measurements_dirs = sorted(root.rglob("measurements"))
    if not measurements_dirs:
        print(f"No measurements folders found under: {root}")
        return

    centers_csv = Path(centers_csv_path)
    if not centers_csv.exists():
        print(f"Centers CSV not found: {centers_csv}")
        return

    centers_map = _load_centers_map(centers_csv)
    if not centers_map:
        print(f"No valid center rows loaded from: {centers_csv}")
        return

    ignore_set = load_ignore_list(ignore_csv_path)

    global_sum = None
    global_count = 0
    total_stats = {
        "total": 0,
        "used": 0,
        "parse_fail": 0,
        "shape_fail": 0,
        "center_missing": 0,
        "roi_ignored": 0,
        "length_mismatch": 0,
    }

    for measurements_dir in measurements_dirs:
        sample_sum, sample_count, sample_stats = process_sample(measurements_dir, centers_map, ignore_set)
        for key in total_stats:
            total_stats[key] += sample_stats.get(key, 0)

        if sample_sum is None or sample_count == 0:
            continue

        if global_sum is None:
            global_sum = sample_sum.copy()
            global_count = sample_count
        else:
            if len(sample_sum) != len(global_sum):
                print(
                    f"Skipping global combine for {measurements_dir}: TRF length {len(sample_sum)} does not match global length {len(global_sum)}"
                )
                continue
            global_sum += sample_sum
            global_count += sample_count

    if global_sum is None or global_count == 0:
        print("No valid TRFs found for global average.")
        print(
            "Center-map summary: "
            f"total={total_stats['total']}, used={total_stats['used']}, "
            f"center_missing={total_stats['center_missing']}, roi_ignored={total_stats['roi_ignored']}, "
            f"parse_fail={total_stats['parse_fail']}, "
            f"shape_fail={total_stats['shape_fail']}, length_mismatch={total_stats['length_mismatch']}"
        )
        return

    global_avg = global_sum / global_count
    global_avg_smoothed = _smooth_trf(global_avg, window=SMOOTH_WINDOW)
    global_trf_dir = root / "trfs"
    csv_path, png_path = _save_average_trf(global_trf_dir, "all-samples", global_avg, global_count)
    csv_path, png_path = _save_average_trf(global_trf_dir, "all-samples_smoothed", global_avg_smoothed, global_count)
    print(f"Saved global average TRF CSV: {csv_path}")
    print(f"Saved global average TRF plot: {png_path}")
    print(
        "Center-map summary: "
        f"total={total_stats['total']}, used={total_stats['used']}, "
        f"center_missing={total_stats['center_missing']}, roi_ignored={total_stats['roi_ignored']}, "
        f"parse_fail={total_stats['parse_fail']}, "
        f"shape_fail={total_stats['shape_fail']}, length_mismatch={total_stats['length_mismatch']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract and average temporal receptive fields (TRFs) from history TIFs.'
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='.',
        help='Root directory to search for measurements subdirectories.',
    )
    parser.add_argument(
        '--centers-csv',
        type=str,
        default=str(DEFAULT_CENTERS_CSV),
        help='Path to centers CSV file from find_center_surround.py.',
    )
    parser.add_argument(
        '--ignore',
        type=str,
        default=None,
        help='Path to CSV file with columns: directory, roi_number. ROIs listed will be skipped.',
    )
    args = parser.parse_args()
    
    main(args.root_dir, centers_csv_path=args.centers_csv, ignore_csv_path=args.ignore) 