import numpy as np
import tifffile
import sys
import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt

GRID_SIZE = 31
GRID_CENTER = (15, 15)
TIME_START_S = -1.5
TIME_END_S = 0.5


def process_dff_weighted_history_xyt(dff_weighted_history):
    """
    Process dff_weighted_history keeping the time dimension (xyt format).
    Z-score normalize each frame independently.
    
    Args:
        dff_weighted_history: Array of shape (time, 16, 16)
    
    Returns:
        normalized_frames: Array of shape (time, 16, 16) with each frame z-score normalized
    """
    # Upcast before processing to avoid overflow when inputs are low-precision TIFF data.
    dff64 = np.asarray(dff_weighted_history, dtype=np.float64)
    time_steps = dff64.shape[0]
    normalized_frames = np.zeros_like(dff64, dtype=np.float32)
    
    # Z-score normalize each frame independently
    for t in range(time_steps):
        frame = dff64[t, :, :]
        mean = np.mean(frame, dtype=np.float64)
        std = np.std(frame, dtype=np.float64)
        normalized_frames[t, :, :] = ((frame - mean) / (std + 1e-8)).astype(np.float32)
    
    return normalized_frames


def load_centers_from_csv(csv_path):
    """
    Load centers from centers_and_contiguous_sizes.csv.
    
    Args:
        csv_path: Path to centers_and_contiguous_sizes.csv
        
    Returns:
        Dict mapping file path (as string) to (center_row, center_col)
    """
    centers_dict = {}
    if csv_path is None or not Path(csv_path).exists():
        print(f"Warning: Centers CSV file does not exist: {csv_path}")
        return centers_dict
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row['file'].strip()
                center_row = int(row['center_row'].strip())
                center_col = int(row['center_col'].strip())
                centers_dict[file_path] = (center_row, center_col)
        print(f"Loaded {len(centers_dict)} centers from {csv_path}")
    except Exception as e:
        print(f"Warning: Could not load centers from {csv_path}: {e}")
    
    return centers_dict


def add_aligned_frames_to_grid_xyt(grid_sum, frames_xyt, center_rc, target_center_rc=GRID_CENTER):
    """
    Shift-add 3D frames (xyt) into grid_sum so center_rc maps to target_center_rc.
    
    Args:
        grid_sum: 3D array of shape (time, GRID_SIZE, GRID_SIZE)
        frames_xyt: 3D array of shape (time, 16, 16)
        center_rc: (center_row, center_col) in the 16x16 frame
        target_center_rc: (target_row, target_col) in the GRID_SIZE x GRID_SIZE grid
    """
    time_steps = frames_xyt.shape[0]
    h, w = frames_xyt.shape[1], frames_xyt.shape[2]
    target_r, target_c = target_center_rc

    for t in range(time_steps):
        frame = frames_xyt[t, :, :]
        for r in range(h):
            for c in range(w):
                rr = target_r + (r - center_rc[0])
                cc = target_c + (c - center_rc[1])
                if 0 <= rr < grid_sum.shape[1] and 0 <= cc < grid_sum.shape[2]:
                    grid_sum[t, rr, cc] += frame[r, c]


def update_grid_count_xyt(grid_count, frames_shape, center_rc, target_center_rc=GRID_CENTER):
    """
    Track how many files contributed to each aligned grid location (3D version).
    
    Args:
        grid_count: 3D array of shape (time, GRID_SIZE, GRID_SIZE)
        frames_shape: Shape of frames (time, 16, 16)
        center_rc: (center_row, center_col) in the 16x16 frame
        target_center_rc: (target_row, target_col) in the GRID_SIZE x GRID_SIZE grid
    """
    time_steps, h, w = frames_shape
    target_r, target_c = target_center_rc

    for t in range(time_steps):
        for r in range(h):
            for c in range(w):
                rr = target_r + (r - center_rc[0])
                cc = target_c + (c - center_rc[1])
                if 0 <= rr < grid_count.shape[1] and 0 <= cc < grid_count.shape[2]:
                    grid_count[t, rr, cc] += 1


def find_input_tif_files(root_dir):
    """Recursively find input dff tifs under root_dir."""
    root = Path(root_dir)

    # Prefer files with dff in the name, similar to recursive matching in the notebook.
    dff_like = [
        p for p in root.rglob('*.tif')
        if p.is_file() and 'dff' in p.name.lower()
    ]

    if dff_like:
        return sorted(dff_like)

    # Fallback: if no dff-named files are found, process all tif inputs.
    all_inputs = [
        p for p in root.rglob('*.tif')
        if p.is_file()
    ]
    return sorted(all_inputs)


def write_aligned_outputs_xyt(output_dir, prefix, grid_sum, grid_count):
    """Write aligned grid mean output in xyt format (3D)."""
    grid_mean = np.divide(
        grid_sum,
        #np.maximum(grid_count, 1),
        grid_count.max(),
        out=np.zeros_like(grid_sum, dtype=np.float64),
    )

    mean_out = output_dir / f'{prefix}_mean_31x31_xyt.tif'
    tifffile.imwrite(str(mean_out), grid_mean.astype(np.float32))
    
    print(f"Wrote aligned output: {mean_out.name}")
    
    return grid_mean


def save_center_pixel_trace(output_dir, prefix, grid_mean, center_rc=GRID_CENTER):
    """
    Extract and save the trace of the center pixel over time.
    
    Args:
        output_dir: Directory to save outputs
        prefix: Prefix for output filenames
        grid_mean: 3D array of shape (time, GRID_SIZE, GRID_SIZE)
        center_rc: (row, col) of center pixel
    """
    # Extract center pixel trace: grid_mean[:, center_rc[0], center_rc[1]]
    center_trace = grid_mean[:, center_rc[0], center_rc[1]]
    
    # Convert time indices to seconds
    time_s = np.linspace(TIME_START_S, TIME_END_S, num=len(center_trace), endpoint=False, dtype=np.float64)
    
    # Save as CSV
    csv_path = output_dir / f'{prefix}_center_pixel_trace.csv'
    out = np.column_stack((time_s, center_trace))
    np.savetxt(str(csv_path), out, delimiter=',', header='time_s,zscore', comments='')
    print(f"Wrote center pixel trace CSV: {csv_path.name}")
    
    # Save as PNG plot
    png_path = output_dir / f'{prefix}_center_pixel_trace.png'
    plt.figure(figsize=(10, 6))
    plt.plot(time_s, center_trace, linewidth=1.5)
    plt.xlim(TIME_START_S, TIME_END_S)
    plt.xlabel('Time (s)')
    plt.ylabel('Z-score')
    plt.title('Center Pixel Trace')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()
    print(f"Wrote center pixel trace plot: {png_path.name}")
    
    return csv_path, png_path


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Align dff_weighted_historys using centers from CSV and write aligned summaries.'
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='.',
        help='Root directory to recursively search for input tif files.',
    )
    parser.add_argument(
        '--centers-csv',
        type=str,
        default=None,
        help='Path to centers_and_contiguous_sizes.csv. If not provided, looks for it in root_dir.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    root_dir = Path(args.root_dir)
    tif_files = find_input_tif_files(root_dir)
    
    # Determine CSV path
    if args.centers_csv:
        csv_path = Path(args.centers_csv)
    else:
        csv_path = root_dir / 'centers_and_contiguous_sizes.csv'
    
    # Load centers from CSV
    csv_centers = load_centers_from_csv(str(csv_path))
    
    if not csv_centers:
        print("Error: No centers loaded from CSV. Exiting.")
        return
    
    # First pass: determine number of time steps
    num_time_steps = None
    for tif_path in tif_files:
        if str(tif_path) in csv_centers:
            dff = tifffile.imread(str(tif_path))
            if dff.ndim >= 3:
                num_time_steps = dff.shape[0]
                print(f"Detected {num_time_steps} time steps from {tif_path.name}")
                break
    
    if num_time_steps is None:
        print("Error: Could not determine number of time steps. Exiting.")
        return
    
    # Initialize 3D grids for txy format (time, x, y)
    grid_sum = np.zeros((num_time_steps, GRID_SIZE, GRID_SIZE), dtype=np.float64)
    grid_count = np.zeros((num_time_steps, GRID_SIZE, GRID_SIZE), dtype=np.int32)
    
    aligned_count = 0
    skipped_count = 0
    
    for tif_path in tif_files:
        dff = tifffile.imread(str(tif_path))
        if dff.ndim < 3:
            continue
        
        # Check if this file has a center in the CSV
        if str(tif_path) not in csv_centers:
            print(f"Skipping {tif_path.name} (no center in CSV)")
            skipped_count += 1
            continue
        
        center_rc = csv_centers[str(tif_path)]
        
        try:
            norm_xyt = process_dff_weighted_history_xyt(dff)
            add_aligned_frames_to_grid_xyt(grid_sum, norm_xyt, center_rc, GRID_CENTER)
            update_grid_count_xyt(grid_count, norm_xyt.shape, center_rc, GRID_CENTER)
            aligned_count += 1
        except Exception as e:
            print(f"Error processing {tif_path.name}: {e}")
            skipped_count += 1
    
    # Write outputs
    grid_mean = write_aligned_outputs_xyt(root_dir, 'aligned_dff_zscore', grid_sum, grid_count)
    save_center_pixel_trace(root_dir, 'aligned_dff_zscore', grid_mean, GRID_CENTER)
    
    print(f"\nSummary:")
    print(f"  Files aligned: {aligned_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Output shape: (time={num_time_steps}, x={GRID_SIZE}, y={GRID_SIZE})")
    print(f"  Output directory: {root_dir}")


if __name__ == '__main__':
    main()
