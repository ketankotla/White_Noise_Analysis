#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map existing ROI response curves to stimulus frames using count_frames.

For each row in the input CSV, this script:
1) loads a response CSV produced by measure_responses.py
2) loads stimulus CSV and runs ResponseTools.count_frames
3) joins stimulus frame metadata onto each response point by frame index
4) filters rows that have sufficient raw-stim context (before/after)
5) reconstructs time-binned history stimuli from raw stim columns 3:-1 into 16x16 arrays in memory-efficient chunks
6) writes a merged table (no ROI re-measurement)
"""

import argparse
import multiprocessing
import os
import csv

import numpy
import pandas
import tifffile

import ResponseTools_v3 as rt
import utility

history_before = 1.5
history_after = 0.5
time_bin_size = 0.1
compute_dtype = numpy.float32
count_dtype = numpy.int32
tiff_output_dtype = numpy.float16

def _to_int(value, default):
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return default


def _is_true(value):
	return str(value).strip().upper() == "TRUE"


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
		from pathlib import Path
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


def should_skip_roi(sample_name, roi_number, ignore_set):
	"""
	Check if a ROI should be skipped based on sample directory and ROI number.
	
	Args:
		sample_name: Name of the sample directory
		roi_number: ROI number (as int)
		ignore_set: Set of tuples (directory, roi_number_str) to skip
		
	Returns:
		True if this ROI should be skipped, False otherwise
	"""
	roi_str = str(int(roi_number))
	return (sample_name, roi_str) in ignore_set


def _pick_stim_file(stim_dir, stimulus_name=None):
	if stimulus_name:
		labeled = list(sorted(utility.get_file_names(stim_dir, file_type="csv", label=stimulus_name)))
		if labeled:
			return str(labeled[1])

	all_csv = list(sorted(utility.get_file_names(stim_dir, file_type="csv")))
	for file in all_csv:
		if 'ternary' in file:
			return str(file)
	raise FileNotFoundError(f"No stimulus CSV found in {stim_dir}")


def _infer_gt_index(stim_file):
	_, header = utility.read_csv_file(stim_file)
	for key in ("global_time", "global time", "time"):
		if key in header:
			return header.index(key)
	# Legacy default in this codebase.
	return 0


def _build_response_file(sample_dir, input_dict):
	out_dir = os.path.join(sample_dir, "measurements")
	out_name = input_dict.get("output_name", input_dict.get("ch1_name", "output"))
	return os.path.join(out_dir, f"{out_name}-raw.csv")


def _normalize_response_frame_column(response_df):
	# Always derive frame IDs from row order so mapping does not depend on any
	# pre-existing frame values in the response CSV.
	response_df = response_df.reset_index(drop=True)
	if "ROI" in response_df.columns:
		response_df["frame"] = response_df.groupby("ROI").cumcount()
	else:
		response_df["frame"] = response_df.index
	return response_df


def _build_stim_map_df(stim_data, gt_index):
	return pandas.DataFrame(
		{
			"frame": list(range(len(stim_data))),
			"stim_frame": stim_data[:, -1].astype(int),
			"stim_global_time": stim_data[:, gt_index],
		}
	)


def _read_raw_stim_times(stim_file, gt_index):
	rows, _ = utility.read_csv_file(stim_file)
	raw_times = []
	for row in rows:
		if gt_index < len(row):
			try:
				raw_times.append(float(row[gt_index]))
			except ValueError:
				raw_times.append(numpy.nan)
		else:
			raw_times.append(numpy.nan)

	raw_times = numpy.asarray(raw_times, dtype=float)
	valid_mask = ~numpy.isnan(raw_times)
	valid_idx = numpy.nonzero(valid_mask)[0]
	valid_times = raw_times[valid_mask]
	return valid_idx, valid_times


def _map_counted_frames_to_raw_indices(stim_map_df, stim_file, gt_index):
	raw_valid_idx, raw_valid_times = _read_raw_stim_times(stim_file, gt_index)
	if len(raw_valid_times) == 0:
		raise ValueError("No valid raw stimulus timestamps found for mapping.")

	timestamps = stim_map_df["stim_global_time"].to_numpy(dtype=float)
	positions = numpy.searchsorted(raw_valid_times, timestamps, side="right") - 1
	positions = numpy.clip(positions, 0, len(raw_valid_times) - 1)

	stim_map_df = stim_map_df.copy()
	stim_map_df["raw_stim_index"] = raw_valid_idx[positions].astype(numpy.int32)
	return stim_map_df


def _filter_rows_with_sufficient_history(
	mapped_df,
	raw_valid_idx,
	raw_valid_times,
	history_before=history_before,
	history_after=history_after,
):
	"""Keep only rows with enough time context before and after raw_stim_index."""
	if len(raw_valid_times) == 0:
		return mapped_df.iloc[0:0].copy()

	raw_idx = mapped_df["raw_stim_index"].to_numpy(dtype=numpy.int64)
	positions = numpy.searchsorted(raw_valid_idx, raw_idx)
	matched = (positions < len(raw_valid_idx)) & (raw_valid_idx[positions] == raw_idx)

	event_times = numpy.full(len(raw_idx), numpy.nan, dtype=float)
	event_times[matched] = raw_valid_times[positions[matched]]

	first_time = raw_valid_times[0]
	last_time = raw_valid_times[-1]
	valid_window = (event_times - history_before >= first_time) & (event_times + history_after <= last_time)
	valid = matched & valid_window

	filtered = mapped_df.loc[valid].copy()
	filtered["stim_event_time"] = event_times[valid]
	return filtered


def _load_raw_stimulus_patterns(stim_file):
	rows, _ = utility.read_csv_file(stim_file)
	patterns = []
	for row in rows:
		vec = row[3:-1]
		patterns.append([float(v) for v in vec])

	patterns = numpy.asarray(patterns, dtype=compute_dtype)
	if patterns.shape[1] ** 0.5 % 1 != 0:
		raise ValueError(
			f"Expected square number of stimulus values from columns 3:-1, got {patterns.shape[1]} in {stim_file}"
		)
	return patterns


def _ensure_dff(mapped_df):
	if "dff" in mapped_df.columns:
		mapped_df["dff"] = pandas.to_numeric(mapped_df["dff"], errors="coerce")
		mapped_df["dff"] = mapped_df["dff"].fillna(0.0)
		return mapped_df

	if "F" not in mapped_df.columns:
		raise ValueError("Mapped dataframe requires either 'dff' or 'F' column.")

	mapped_df = mapped_df.copy()
	mapped_df["F"] = pandas.to_numeric(mapped_df["F"], errors="coerce").fillna(0.0)

	if "ROI" in mapped_df.columns:
		baseline = mapped_df.groupby("ROI")["F"].transform("median")
	else:
		baseline = pandas.Series([mapped_df["F"].median()] * len(mapped_df), index=mapped_df.index)

	baseline = baseline.replace(0, 1e-9)
	mapped_df["dff"] = (mapped_df["F"] - baseline) / baseline
	return mapped_df


def _save_float_tiff_stack(array3d, out_file, output_dtype=tiff_output_dtype):
	"""Save array to TIFF, handling overflow by clipping to dtype range"""
	# Clip values to the valid range for the output dtype to avoid overflow
	if output_dtype == numpy.float16:
		# float16 range: ±65504
		array_clipped = numpy.clip(array3d, -65500, 65500)
	elif output_dtype == numpy.float32:
		# float32 has much larger range, no clipping needed
		array_clipped = array3d
	else:
		# For integer types, clip to min/max of that dtype
		info = numpy.iinfo(output_dtype) if numpy.issubdtype(output_dtype, numpy.integer) else None
		if info:
			array_clipped = numpy.clip(array3d, info.min, info.max)
		else:
			array_clipped = array3d
	
	tifffile.imwrite(out_file, array_clipped.astype(output_dtype, copy=False))


def _save_weighted_history_tiffs(
	output_dir,
	output_name,
	mapped_df,
	raw_patterns,
	raw_valid_idx,
	raw_valid_times,
	sample_name=None,
	ignore_set=None,
	history_before=history_before,
	history_after=history_after,
	time_bin_size=time_bin_size,
	chunk_size=64,
):
	if ignore_set is None:
		ignore_set = set()
	if "ROI" not in mapped_df.columns:
		raise ValueError("Per-ROI weighted output requires an 'ROI' column.")
	if history_before <= 0:
		raise ValueError("history_before must be positive.")
	if history_after < 0:
		raise ValueError("history_after must be >= 0.")
	if time_bin_size <= 0:
		raise ValueError("time_bin_size must be > 0.")
	if len(raw_valid_times) == 0:
		raise ValueError("No valid raw stimulus timestamps available for time binning.")

	total_window = history_before + history_after
	history_total = int(round(total_window / time_bin_size))
	if history_total <= 0:
		raise ValueError("Computed history_total must be positive.")
	if not numpy.isclose(history_total * time_bin_size, total_window, atol=1e-9):
		raise ValueError(
			f"history_before + history_after must be divisible by time_bin_size; got {total_window} and {time_bin_size}."
		)

	dim = int(raw_patterns.shape[1] ** 0.5)

	out_files = []
	for roi, group in mapped_df.groupby("ROI", sort=True):
		# Check if this ROI should be skipped
		if sample_name and should_skip_roi(sample_name, roi, ignore_set):
			print(f"  Skipping ROI {int(roi)} (in ignore list)")
			continue
		
		raw_idx = group["raw_stim_index"].to_numpy(dtype=numpy.int64)
		weights = group["dff"].to_numpy(dtype=compute_dtype)
		roi_sum = numpy.zeros((history_total, dim, dim), dtype=compute_dtype)
		event_sum = numpy.zeros((history_total, dim * dim), dtype=compute_dtype)
		event_count = numpy.zeros(history_total, dtype=count_dtype)

		for start in range(0, len(raw_idx), chunk_size):
			end = min(start + chunk_size, len(raw_idx))
			chunk_idx = raw_idx[start:end]
			chunk_w = weights[start:end]

			for event_idx, event_weight in zip(chunk_idx, chunk_w):
				pos = numpy.searchsorted(raw_valid_idx, event_idx)
				if pos >= len(raw_valid_idx) or raw_valid_idx[pos] != event_idx:
					continue

				event_time = raw_valid_times[pos]
				start_time = event_time - history_before
				end_time = event_time + history_after

				left = numpy.searchsorted(raw_valid_times, start_time, side="left")
				right = numpy.searchsorted(raw_valid_times, end_time, side="left")
				if right <= left:
					continue

				window_times = raw_valid_times[left:right]
				window_indices = raw_valid_idx[left:right]
				rel_time = window_times - event_time
				bin_ids = numpy.floor((rel_time + history_before) / time_bin_size).astype(numpy.int64)
				bin_ids = numpy.clip(bin_ids, 0, history_total - 1)

				window_patterns = raw_patterns[window_indices]
				event_sum.fill(0)
				event_count.fill(0)

				numpy.add.at(event_sum, bin_ids, window_patterns)
				numpy.add.at(event_count, bin_ids, 1)

				nonzero = event_count > 0
				if numpy.any(nonzero):
					event_sum[nonzero] /= event_count[nonzero, None].astype(compute_dtype)

				roi_sum += (event_weight * event_sum.reshape(history_total, dim, dim)).astype(compute_dtype, copy=False)
				roi_cropped = roi_sum[:, 8:21, :15]
				roi_cropped = numpy.rot90(roi_cropped, k=1, axes=(1, 2))
				

		roi_file = os.path.join(output_dir, f"{output_name}-roi_{int(roi)}-dff-weighted-history.tif")
		#_save_float_tiff_stack(roi_sum, roi_file)
		_save_float_tiff_stack(roi_cropped, roi_file)
		out_files.append(roi_file)

	return out_files


def _process_row(task):
	parent_dir, header, row, ignore_set = task
	input_dict = rt.get_input_dict(row, header)
	if "include" in input_dict and not _is_true(input_dict["include"]):
		return []

	sample_name = input_dict["sample_name"]
	sample_dir = os.path.join(parent_dir, sample_name)
	stim_dir = os.path.join(sample_dir, "stim_files")
	out_dir = os.path.join(sample_dir, "measurements")
	os.makedirs(out_dir, exist_ok=True)

	response_file = _build_response_file(sample_dir, input_dict)
	stim_file = _pick_stim_file(stim_dir, input_dict.get("stimulus_name"))
	if not os.path.exists(response_file):
		raise FileNotFoundError(f"Response file not found: {response_file}")

	count_input = dict(input_dict)
	if "gt_index" not in count_input or str(count_input["gt_index"]).strip() == "":
		count_input["gt_index"] = 0
	gt_index = int(count_input["gt_index"])


	print(f"mapping sample={input_dict.get('sample_name')} output={input_dict.get('output_name', input_dict.get('ch1_name'))}")

	response_df = pandas.read_csv(response_file)
	response_df = _normalize_response_frame_column(response_df)
	stim_data, stim_header = rt.count_frames(stim_file, count_input)

	min_frame = max(_to_int(input_dict.get("min_frame", 0), 0), 0)
	max_frame = _to_int(input_dict.get("max_frame", len(stim_data)), len(stim_data))
	max_frame = min(max_frame, len(stim_data))
	if max_frame <= min_frame:
		max_frame = len(stim_data)

	stim_data = stim_data[min_frame:max_frame]
	stim_map_df = _build_stim_map_df(stim_data, gt_index)
	stim_map_df = _map_counted_frames_to_raw_indices(stim_map_df, stim_file, gt_index)

	mapped_df = response_df.merge(stim_map_df, on="frame", how="inner")
	raw_patterns = _load_raw_stimulus_patterns(stim_file)
	raw_valid_idx, raw_valid_times = _read_raw_stim_times(stim_file, gt_index)
	mapped_df = _filter_rows_with_sufficient_history(
		mapped_df,
		raw_valid_idx=raw_valid_idx,
		raw_valid_times=raw_valid_times,
		history_before=history_before,
		history_after=history_after,
	)
	mapped_df = _ensure_dff(mapped_df)
	mapped_df = mapped_df.reset_index(drop=True)

	output_name = input_dict.get("output_name", input_dict.get("ch1_name", "output"))
	mapped_csv = os.path.join(out_dir, f"{output_name}-stim-mapped.csv")
	mapped_df.to_csv(mapped_csv, index=False)
	roi_tiffs = _save_weighted_history_tiffs(
		out_dir,
		output_name,
		mapped_df,
		raw_patterns,
		raw_valid_idx,
		raw_valid_times,
		sample_name=sample_name,
		ignore_set=ignore_set,
		history_before=history_before,
		history_after=history_after,
		time_bin_size=time_bin_size,
	)

	if _is_true(input_dict.get("verbose", "FALSE")):
		parsed_dir = os.path.join(stim_dir, "parsed")
		os.makedirs(parsed_dir, exist_ok=True)
		parsed_csv = os.path.join(parsed_dir, f"parsed-{os.path.basename(str(stim_file))}")
		utility.write_csv(stim_data, stim_header, parsed_csv)

	print(f"saved mapped responses to {mapped_csv}")

	return [f'done with {mapped_csv}']


def run(parent_dir, input_csv, workers=None, ignore_csv_path=None):
	rows, header = utility.read_csv_file(input_csv)
	ignore_set = load_ignore_list(ignore_csv_path)
	tasks = [(parent_dir, header, row, ignore_set) for row in rows]

	if workers is None:
		workers = max(1, (os.cpu_count() or 1) - 1)

	if workers <= 1:
		for task in tasks:
			for line in _process_row(task):
				print(line)
		return
	print(f'running with {workers} workers')
	with multiprocessing.get_context("spawn").Pool(processes=workers) as pool:
		for log_lines in pool.imap_unordered(_process_row, tasks):
			for line in log_lines:
				print(line)


def parse_args():
	parser = argparse.ArgumentParser(description="Assign stimulus frame IDs and map them to ROI responses.")
	parser.add_argument(
		"--parent-dir",
		default="..",
		help="Parent directory containing sample folders.",
	)
	parser.add_argument(
		"--input-csv",
		default="../new_inputs_binned.csv",
		help="CSV that lists sample rows to process.",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=None,
		help="Number of worker processes for per-row multiprocessing (default: CPU count - 1).",
	)
	parser.add_argument(
		"--ignore",
		type=str,
		default=None,
		help="Path to CSV file with columns: directory, roi_number. ROIs listed will be skipped.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(args.parent_dir, args.input_csv, workers=args.workers, ignore_csv_path=args.ignore)
