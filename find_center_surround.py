import numpy as np
import tifffile
import sys
import argparse
from collections import defaultdict
from pathlib import Path

GRID_SIZE = 31
GRID_CENTER = (15, 15)
ANALYSIS_START_INDEX = 10
ANALYSIS_END_INDEX = 15

def process_dff_weighted_history(dff_weighted_history):
    """
    Average dff_weighted_history across time to get a single 16x16 frame,
    then z-score normalize all pixels.
    
    Args:
        dff_weighted_history: Array of shape (time, 16, 16)
    
    Returns:
        normalized_frame: Z-score normalized 16x16 frame
    """
    if dff_weighted_history.shape[0] <= ANALYSIS_END_INDEX:
        raise ValueError(
            f"Need at least {ANALYSIS_END_INDEX + 1} frames for analysis window "
            f"{ANALYSIS_START_INDEX}:{ANALYSIS_END_INDEX}."
        )

    # Upcast before reduction to avoid overflow when inputs are low-precision TIFF data.
    dff64 = np.asarray(dff_weighted_history, dtype=np.float64)
    analysis_window = dff64[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX + 1]
    averaged_frame = np.mean(analysis_window, axis=0, dtype=np.float64)
    
    # Z-score normalize all pixels
    mean = np.mean(averaged_frame, dtype=np.float64)
    std = np.std(averaged_frame, dtype=np.float64)
    normalized_frame = ((averaged_frame - mean) / (std + 1e-8)).astype(np.float32)
    
    return normalized_frame

def average_dff_weighted_history(dff_weighted_history):
    """Average dff_weighted_history across time to get a single 2D frame."""
    if dff_weighted_history.shape[0] <= ANALYSIS_END_INDEX:
        raise ValueError(
            f"Need at least {ANALYSIS_END_INDEX + 1} frames for analysis window "
            f"{ANALYSIS_START_INDEX}:{ANALYSIS_END_INDEX}."
        )

    dff64 = np.asarray(dff_weighted_history, dtype=np.float64)
    analysis_window = dff64[ANALYSIS_START_INDEX:ANALYSIS_END_INDEX + 1]
    return np.mean(analysis_window, axis=0, dtype=np.float64).astype(np.float32)

def find_center_and_contiguous_positive_region(normalized_frame):
    """
    Find the maximum pixel (center) and all contiguous pixels with z-score > 0.5
    connected to that center (4-connected neighborhood: up/down/left/right).
    """
    if normalized_frame.ndim != 2:
        raise ValueError("normalized_frame must be a 2D array")

    h, w = normalized_frame.shape

    center_idx = np.argmax(normalized_frame)
    center_rc = np.unravel_index(center_idx, normalized_frame.shape)
    center_value = normalized_frame[center_rc]

    positive_mask = normalized_frame > 0.5
    contiguous_mask = np.zeros_like(positive_mask, dtype=bool)
    if not positive_mask[center_rc]:
        return center_rc, center_value, contiguous_mask, np.argwhere(contiguous_mask)

    stack = [center_rc]
    contiguous_mask[center_rc] = True

    # 4-connectivity only
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if positive_mask[nr, nc] and not contiguous_mask[nr, nc]:
                    contiguous_mask[nr, nc] = True
                    stack.append((nr, nc))

    contiguous_coords = np.argwhere(contiguous_mask)
    return center_rc, center_value, contiguous_mask, contiguous_coords


def find_contiguous_positive_region_from_seed(normalized_frame, center_rc, threshold=0.5):
    """
    Find contiguous positive region starting from a provided center coordinate.
    """
    if normalized_frame.ndim != 2:
        raise ValueError("normalized_frame must be a 2D array")

    h, w = normalized_frame.shape
    r0, c0 = center_rc
    if not (0 <= r0 < h and 0 <= c0 < w):
        raise ValueError("center_rc is outside normalized_frame bounds")

    center_value = normalized_frame[r0, c0]
    positive_mask = normalized_frame > threshold
    contiguous_mask = np.zeros_like(positive_mask, dtype=bool)
    if not positive_mask[r0, c0]:
        return center_rc, center_value, contiguous_mask, np.argwhere(contiguous_mask)

    stack = [(r0, c0)]
    contiguous_mask[r0, c0] = True
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if positive_mask[nr, nc] and not contiguous_mask[nr, nc]:
                    contiguous_mask[nr, nc] = True
                    stack.append((nr, nc))

    contiguous_coords = np.argwhere(contiguous_mask)
    return center_rc, center_value, contiguous_mask, contiguous_coords

def find_all_negative_pixels(normalized_frame, threshold=-0.5):
    """
    Find all pixels with z-score < threshold (default: -0.5).
    Returns a mask of all negative pixels and their coordinates.
    """
    if normalized_frame.ndim != 2:
        raise ValueError("normalized_frame must be a 2D array")
    
    negative_mask = normalized_frame < threshold
    negative_coords = np.argwhere(negative_mask)
    return negative_mask, negative_coords


def find_largest_contiguous_negative_region(normalized_frame):
    """
    Find the largest contiguous region with z-score < -0.5
    using 4-connected neighborhood (up/down/left/right).
    """
    if normalized_frame.ndim != 2:
        raise ValueError("normalized_frame must be a 2D array")

    h, w = normalized_frame.shape
    negative_mask = normalized_frame < -0.5
    visited = np.zeros_like(negative_mask, dtype=bool)
    largest_region_mask = np.zeros_like(negative_mask, dtype=bool)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if visited[r, c] or not negative_mask[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            current_coords = []

            while stack:
                cr, cc = stack.pop()
                current_coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if negative_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            if len(current_coords) > np.count_nonzero(largest_region_mask):
                largest_region_mask[:] = False
                for rr, cc in current_coords:
                    largest_region_mask[rr, cc] = True

    largest_region_coords = np.argwhere(largest_region_mask)
    return largest_region_mask, largest_region_coords


def find_masks_for_center(normalized_frame, center_rc=None):
    """Find center, center mask, surround mask, and combined mask for one frame."""
    if center_rc is None:
        center_rc, center_value, contiguous_mask, contiguous_coords = find_center_and_contiguous_positive_region(normalized_frame)
    else:
        center_rc, center_value, contiguous_mask, contiguous_coords = find_contiguous_positive_region_from_seed(
            normalized_frame,
            center_rc,
        )

    # Use all negative pixels with z-score < -0.5
    #surround_mask, _ = find_all_negative_pixels(normalized_frame)
    
    # Old code: find only the largest contiguous negative region
    surround_mask, _ = find_largest_contiguous_negative_region(normalized_frame)
    
    combined_mask = contiguous_mask | surround_mask

    return {
        'center': center_rc,
        'center_value': center_value,
        'contiguous_mask': contiguous_mask,
        'contiguous_coords': contiguous_coords,
        'surround_mask': surround_mask,
        'combined_mask': combined_mask,
    }


def find_masks_for_each_center(records):
    """
    Group frames by center location and find masks for each center separately.

    Args:
        records: List of dicts containing:
            - file_path: str
            - normalized_frame: 2D array
            - center: (row, col)

    Returns:
        Dict keyed by center (row, col), value is list of per-file mask dictionaries.
    """
    grouped_masks = defaultdict(list)

    for record in records:
        center_key = tuple(record['center'])
        masks = find_masks_for_center(record['normalized_frame'], center_key)
        grouped_masks[center_key].append({
            'file_path': record['file_path'],
            **masks,
        })

    return grouped_masks

def add_masked_frame_to_grid(grid_sum, frame, mask, center_rc, target_center_rc=GRID_CENTER):
    """Shift-add a masked frame into grid_sum so center_rc maps to target_center_rc."""
    h, w = frame.shape
    target_r, target_c = target_center_rc

    masked = frame * mask

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            # if (r, c) == center_rc:
            #     continue
            rr = target_r + (r - center_rc[0])
            cc = target_c + (c - center_rc[1])
            if 0 <= rr < grid_sum.shape[0] and 0 <= cc < grid_sum.shape[1]:
                grid_sum[rr, cc] += masked[r, c]


def update_grid_count(grid_count, frame_shape, mask, center_rc, target_center_rc=GRID_CENTER):
    """Track how many files contributed to each aligned grid location."""
    contribution_mask = np.zeros_like(grid_count, dtype=bool)
    h, w = frame_shape
    target_r, target_c = target_center_rc

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            # if (r, c) == center_rc:
            #     continue
            rr = target_r + (r - center_rc[0])
            cc = target_c + (c - center_rc[1])
            if 0 <= rr < grid_count.shape[0] and 0 <= cc < grid_count.shape[1]:
                contribution_mask[rr, cc] = True

    grid_count[contribution_mask] += 1


def write_grid_outputs(output_dir, prefix, grid_sum, grid_count):
    grid_mean = np.divide(
        grid_sum,
        #np.maximum(grid_count, 1),
        grid_count.max(),
        out=np.zeros_like(grid_sum, dtype=np.float64),
    )

    sum_out = output_dir / f'{prefix}_sum_31x31.tif'
    mean_out = output_dir / f'{prefix}_mean_31x31.tif'
    count_out = output_dir / f'{prefix}_count_31x31.tif'
    tifffile.imwrite(str(sum_out), grid_sum.astype(np.float32))
    tifffile.imwrite(str(mean_out), grid_mean.astype(np.float32))
    tifffile.imwrite(str(count_out), grid_count.astype(np.int32))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Find center/surround masks from dff tifs and write aligned summaries.'
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='.',
        help='Root directory to recursively search for input tif files.',
    )
    parser.add_argument(
        '--group-by-center-location',
        action='store_true',
        help='Group files by detected center location and write separate outputs per center.',
    )
    parser.add_argument(
        '--ignore',
        type=str,
        default=None,
        help='Path to CSV file with columns: directory, roi_number. ROIs listed will be skipped.',
    )
    return parser.parse_args(argv)


def is_generated_output_tif(path):
    name = path.name
    return (
        name.endswith('_center_mask.tif')
        or name.endswith('_contiguous_mask.tif')
        or name.endswith('_surround_mask.tif')
        or name.endswith('_combined_mask.tif')
        or name.endswith('_masked_centered_31x31.tif')
        or name.endswith('_31x31.tif')
    )


def find_input_tif_files(root_dir):
    """Recursively find input dff tifs under root_dir."""
    root = Path(root_dir)

    # Prefer files with dff in the name, similar to recursive matching in the notebook.
    dff_like = [
        p for p in root.rglob('*.tif')
        if p.is_file() and 'dff' in p.name.lower() and not is_generated_output_tif(p)
    ]

    if dff_like:
        return sorted(dff_like)

    # Fallback: if no dff-named files are found, process all tif inputs except generated outputs.
    all_inputs = [
        p for p in root.rglob('*.tif')
        if p.is_file() and not is_generated_output_tif(p)
    ]
    return sorted(all_inputs)


def load_ignore_list(csv_path):
    """
    Load ignore list from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: directory, roi_number
        
    Returns:
        Set of tuples (directory_path, roi_number) to skip
    """
    ignore_set = set()
    if csv_path is None:
        print("No ignore CSV file provided (--ignore not specified)")
        return ignore_set
    
    try:
        import csv
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"Warning: Ignore CSV file does not exist: {csv_path}")
            return ignore_set
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            print(f"Loading ignore list from {csv_path}")
            for row in reader:
                if len(row) >= 2:
                    #print(f"  Read row: {row}")
                    directory = row[0].strip()
                    roi_number = row[1].strip()
                    ignore_set.add((directory, roi_number))
                    #print(f"  Added: directory='{directory}', roi_number='{roi_number}'")
        print(f"Loaded {len(ignore_set)} entries to ignore")
    except Exception as e:
        print(f"Warning: Could not load ignore list from {csv_path}: {e}")
    
    return ignore_set


def should_skip_roi(tif_path, roi_number, ignore_set):
    """
    Check if a ROI should be skipped based on directory and ROI number.
    
    Args:
        tif_path: Path object for the TIF file
        roi_number: ROI number (as string)
        ignore_set: Set of tuples (directory_path, roi_number) to skip
        
    Returns:
        True if this ROI should be skipped, False otherwise
    """
    # Check against each entry in the ignore set
    for ignore_dir, ignore_roi in ignore_set:
        # Check if the file's directory path contains or matches the ignore directory
        if ignore_dir in str(tif_path.parent) or str(tif_path.parent).endswith(ignore_dir):
            if roi_number == ignore_roi:
                return True
    return False



def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    root_dir = Path(args.root_dir)
    tif_files = find_input_tif_files(root_dir)
    ignore_set = load_ignore_list(args.ignore)
    #print(ignore_set)
    grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    surround_grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    surround_grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    combined_grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    combined_grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    centers = []
    group_records = []

    for tif_path in tif_files:
        dff = tifffile.imread(str(tif_path))
        if dff.ndim < 3:
            continue

        # Extract ROI number from filename (assumes format like filename_roi123.tif or roi_0_data.tif)
        # Adjust this logic based on your actual filename format
        roi_number = None
        filename = tif_path.stem.lower()
        import re
        roi_match = re.search(r'roi[_\s]*(\d+)', filename)
        if roi_match:
            roi_number = roi_match.group(1)
        
        # Check if this ROI should be skipped
        if roi_number and should_skip_roi(tif_path, roi_number, ignore_set):
            print(f"Skipping {tif_path} (ROI {roi_number})")
            continue

        norm = process_dff_weighted_history(dff)
        masks = find_masks_for_center(norm)
        center = masks['center']
        center_value = masks['center_value']
        contiguous_mask = masks['contiguous_mask']
        contiguous_coords = masks['contiguous_coords']
        surround_mask = masks['surround_mask']
        combined_mask = masks['combined_mask']

        center_masks_dir = tif_path.parent / 'center_masks'
        surround_masks_dir = tif_path.parent / 'surround_masks'
        combined_masks_dir = tif_path.parent / 'combined_masks'
        center_masks_dir.mkdir(parents=True, exist_ok=True)
        surround_masks_dir.mkdir(parents=True, exist_ok=True)
        combined_masks_dir.mkdir(parents=True, exist_ok=True)

        center_out = center_masks_dir / f'{tif_path.stem}_center_mask.tif'
        tifffile.imwrite(str(center_out), contiguous_mask)

        surround_out = surround_masks_dir / f'{tif_path.stem}_surround_mask.tif'
        tifffile.imwrite(str(surround_out), surround_mask)

        combined_out = combined_masks_dir / f'{tif_path.stem}_combined_mask.tif'
        tifffile.imwrite(str(combined_out), combined_mask)

        add_masked_frame_to_grid(grid_sum, norm, contiguous_mask, center, GRID_CENTER)
        add_masked_frame_to_grid(surround_grid_sum, norm, surround_mask, center, GRID_CENTER)
        add_masked_frame_to_grid(combined_grid_sum, norm, combined_mask, center, GRID_CENTER)

        update_grid_count(grid_count, norm.shape, contiguous_mask, center, GRID_CENTER)
        update_grid_count(surround_grid_count, norm.shape, surround_mask, center, GRID_CENTER)
        update_grid_count(combined_grid_count, norm.shape, combined_mask, center, GRID_CENTER)

        centers.append((str(tif_path), int(center[0]), int(center[1]), float(center_value), int(contiguous_coords.shape[0])))

        if args.group_by_center_location:
            group_records.append({
                'file_path': str(tif_path),
                'normalized_frame': norm,
                'center': (int(center[0]), int(center[1])),
            })

    write_grid_outputs(root_dir, 'contiguous_masked_zscore', grid_sum, grid_count)
    write_grid_outputs(root_dir, 'surround_masked_zscore', surround_grid_sum, surround_grid_count)
    write_grid_outputs(root_dir, 'combined_masked_zscore', combined_grid_sum, combined_grid_count)

    centers_path = root_dir / 'centers_and_contiguous_sizes.csv'
    with centers_path.open('w') as fp:
        fp.write('file,center_row,center_col,center_zscore,contiguous_size\n')
        for row in centers:
            fp.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n')

    if args.group_by_center_location and group_records:
        grouped_masks = find_masks_for_each_center(group_records)
        grouped_root = root_dir / 'grouped_by_center_location'
        grouped_root.mkdir(parents=True, exist_ok=True)

        for center_key, center_entries in grouped_masks.items():
            center_dir = grouped_root / f'center_r{center_key[0]}_c{center_key[1]}'
            center_dir.mkdir(parents=True, exist_ok=True)

            c_grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
            c_grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
            s_grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
            s_grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
            b_grid_sum = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
            b_grid_count = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

            group_centers = []
            for entry in center_entries:
                file_path = Path(entry['file_path'])
                norm = next(r['normalized_frame'] for r in group_records if r['file_path'] == entry['file_path'])
                center = entry['center']
                contiguous_mask = entry['contiguous_mask']
                contiguous_coords = entry['contiguous_coords']
                surround_mask = entry['surround_mask']
                combined_mask = entry['combined_mask']

                center_masks_dir = center_dir / 'center_masks'
                surround_masks_dir = center_dir / 'surround_masks'
                combined_masks_dir = center_dir / 'combined_masks'
                center_masks_dir.mkdir(parents=True, exist_ok=True)
                surround_masks_dir.mkdir(parents=True, exist_ok=True)
                combined_masks_dir.mkdir(parents=True, exist_ok=True)

                tifffile.imwrite(str(center_masks_dir / f'{file_path.stem}_center_mask.tif'), contiguous_mask)
                tifffile.imwrite(str(surround_masks_dir / f'{file_path.stem}_surround_mask.tif'), surround_mask)
                tifffile.imwrite(str(combined_masks_dir / f'{file_path.stem}_combined_mask.tif'), combined_mask)

                add_masked_frame_to_grid(c_grid_sum, norm, contiguous_mask, center, GRID_CENTER)
                add_masked_frame_to_grid(s_grid_sum, norm, surround_mask, center, GRID_CENTER)
                add_masked_frame_to_grid(b_grid_sum, norm, combined_mask, center, GRID_CENTER)

                update_grid_count(c_grid_count, norm.shape, contiguous_mask, center, GRID_CENTER)
                update_grid_count(s_grid_count, norm.shape, surround_mask, center, GRID_CENTER)
                update_grid_count(b_grid_count, norm.shape, combined_mask, center, GRID_CENTER)

                group_centers.append((str(file_path), int(center[0]), int(center[1]), float(entry['center_value']), int(contiguous_coords.shape[0])))

            write_grid_outputs(center_dir, 'contiguous_masked_zscore', c_grid_sum, c_grid_count)
            write_grid_outputs(center_dir, 'surround_masked_zscore', s_grid_sum, s_grid_count)
            write_grid_outputs(center_dir, 'combined_masked_zscore', b_grid_sum, b_grid_count)

            grouped_centers_path = center_dir / 'centers_and_contiguous_sizes.csv'
            with grouped_centers_path.open('w') as fp:
                fp.write('file,center_row,center_col,center_zscore,contiguous_size\n')
                for row in group_centers:
                    fp.write(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n')


if __name__ == '__main__':
    main()