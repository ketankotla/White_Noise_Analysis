#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure ROI response curves from image stacks.

For each row in an input CSV, this script:
1) loads image and mask files
2) measures fluorescence traces for one/all/thresholded ROIs
3) saves raw per-ROI curves and plots (F vs frame)
"""

import argparse
import os

import numpy
import pandas
import seaborn
from matplotlib import pyplot as plt

import ResponseClassSimple_v3 as ResponseClassSimple
import ResponseTools_v3 as rt
import utility


def _to_int(value, default):
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return default


def _is_true(value):
	return str(value).strip().upper() == "TRUE"


def _build_image_path(sample_dir, input_dict):
	ch1_name = input_dict["ch1_name"]
	aligned = _is_true(input_dict.get("aligned", "TRUE"))
	if aligned:
		return os.path.join(sample_dir, "aligned_images", f"{ch1_name}_wn-aligned.tif")
		#return os.path.join(sample_dir, "aligned_images", f"{ch1_name}-aligned.tif")
	return os.path.join(sample_dir, "images", f"{ch1_name}.tif")


def _build_paths(parent_dir, input_dict):
	sample_dir = os.path.join(parent_dir, input_dict["sample_name"])
	mask_name = input_dict.get("mask_name")
	if mask_name:
		mask_file = os.path.join(sample_dir, "masks", f"{mask_name}.tif")
	else:
		mask_file = _pick_mask_file(os.path.join(sample_dir, "masks"))
	return {
		"sample_dir": sample_dir,
		"image_file": _build_image_path(sample_dir, input_dict),
		"mask_file": mask_file,
		"plot_dir": os.path.join(sample_dir, "plots"),
		"output_dir": os.path.join(sample_dir, "measurements"),
		"mask_dir": os.path.join(sample_dir, "masks"),
	}


def _pick_mask_file(mask_dir):
	candidates = sorted(utility.get_file_names(mask_dir, file_type="all", label=""))
	tifs = [f for f in candidates if f.lower().endswith(".tif")]
	preferred = [f for f in tifs if "-labels" not in os.path.basename(f) and 'numbered' not in os.path.basename(f).lower()]
	if preferred:
		return preferred[0]
	if tifs:
		return tifs[0]
	raise FileNotFoundError(f"No mask .tif found in: {mask_dir}")


def _create_response_objects(images, mask, input_dict):
	roi_mode = str(input_dict.get("ROI", "all"))
	if roi_mode == "one":
		responses, roi_numbers, labels = rt.measure_one_ROI(images, mask)
	elif roi_mode == "thresholded":
		threshold = float(input_dict.get("threshold", 0))
		responses, roi_numbers, labels = rt.measure_multiple_thresholded_ROIs(images, mask, threshold)
	else:
		responses, roi_numbers, labels = rt.measure_multiple_ROIs(images, mask)

	response_objects = []
	for fluorescence, roi_number in zip(responses, roi_numbers):
		ro = ResponseClassSimple.Response(
			sample_name=input_dict.get("sample_name"),
			reporter=input_dict.get("reporter_name"),
			genotype=input_dict.get("genotype"),
			compartment=input_dict.get("compartment"),
			stimulus_name=input_dict.get("stimulus_name"),
			ROI_number=roi_number,
			F=fluorescence,
		)
		response_objects.append(ro)

	return response_objects, labels


def _save_raw_responses_without_stim(response_objects, filename):
	out_frames = []
	for ro in response_objects:
		n_frames = len(ro.F)
		out = pandas.DataFrame(
			{
				"ROI": ro.ROI_number,
				"frame": list(range(n_frames)),
				"F": ro.F,
			}
		)
		out_frames.append(out)

	if out_frames:
		output = pandas.concat(out_frames, ignore_index=True)
	else:
		output = pandas.DataFrame(columns=["ROI", "frame", "F"])

	output.to_csv(filename, index=False)
	return output


def _save_dff_responses(response_objects, filename):
	out_frames = []
	for ro in response_objects:
		f = numpy.asarray(ro.F, dtype=float)
		n_frames = len(f)
		if n_frames == 0:
			continue

		# Use the median trace value as baseline for robust dF/F normalization.
		baseline = float(numpy.median(f))
		if baseline == 0:
			baseline = 1e-9
		dff = (f - baseline) / baseline

		out = pandas.DataFrame(
			{
				"ROI": ro.ROI_number,
				"frame": list(range(n_frames)),
				"dff": dff,
			}
		)
		out_frames.append(out)

	if out_frames:
		output = pandas.concat(out_frames, ignore_index=True)
	else:
		output = pandas.DataFrame(columns=["ROI", "frame", "dff"])

	output.to_csv(filename, index=False)
	return output


def _plot_roi_curves(df, filename, y_col):
	seaborn.lineplot(df, x="frame", y=y_col, hue="ROI", palette="tab10")
	plt.savefig(filename)
	plt.clf()


def _plot_dff_per_roi(dff_df, output_dir, output_name):
	if dff_df.empty:
		return

	for roi in sorted(dff_df["ROI"].unique()):
		roi_df = dff_df[dff_df["ROI"] == roi]
		roi_file = os.path.join(output_dir, f"{output_name}-dff-roi_{int(roi)}.png")
		seaborn.lineplot(roi_df, x="frame", y="dff", color="black")
		plt.title(f"dF/F ROI {int(roi)}")
		plt.savefig(roi_file)
		plt.clf()


def run(parent_dir, input_csv):
	rows, header = utility.read_csv_file(input_csv)

	for row in rows:
		input_dict = rt.get_input_dict(row, header)

		if "include" in input_dict and not _is_true(input_dict["include"]):
			continue

		paths = _build_paths(parent_dir, input_dict)
		os.makedirs(paths["plot_dir"], exist_ok=True)
		os.makedirs(paths["output_dir"], exist_ok=True)

		print(
			"analyzing sample "
			+ f"{input_dict.get('sample_name')} {input_dict.get('output_name')} "
			+ f"{os.path.basename(paths['mask_file'])}"
		)

		images = utility.read_tifs(paths["image_file"])
		mask = utility.read_tif(paths["mask_file"])
		n = len(images)

		min_frame = max(_to_int(input_dict.get("min_frame", 0), 0), 0)
		max_frame_raw = _to_int(input_dict.get("max_frame", n), n)
		max_frame = min(max_frame_raw, n)
		if max_frame <= min_frame:
			max_frame = n

		images = images[min_frame:max_frame]

		response_objects, labels = _create_response_objects(images, mask, input_dict)

		output_name = input_dict.get("output_name", input_dict.get("ch1_name", "output"))
		raw_csv = os.path.join(paths["output_dir"], f"{output_name}-raw.csv")
		dff_csv = os.path.join(paths["output_dir"], f"{output_name}-dff.csv")
		raw_plot = os.path.join(paths["plot_dir"], f"{output_name}-raw.png")
		dff_plot = os.path.join(paths["plot_dir"], f"{output_name}-dff.png")
		mask_base = os.path.splitext(os.path.basename(paths["mask_file"]))[0]
		labels_tif = os.path.join(paths["mask_dir"], f"{mask_base}-labels.tif")

		raw_df = _save_raw_responses_without_stim(response_objects, raw_csv)
		dff_df = _save_dff_responses(response_objects, dff_csv)
		_plot_roi_curves(raw_df, raw_plot, y_col="F")
		_plot_roi_curves(dff_df, dff_plot, y_col="dff")
		_plot_dff_per_roi(dff_df, paths["plot_dir"], output_name)
		utility.save_tif(labels.astype("uint16"), labels_tif)

		print(f"saved ROI responses to {raw_csv}")
		print(f"saved ROI dF/F to {dff_csv}")


def parse_args():
	parser = argparse.ArgumentParser(description="Measure frame-wise responses from each ROI.")
	parser.add_argument(
		"--parent-dir",
		default="..",
		help="Parent directory containing sample folders listed in input CSV.",
	)
	parser.add_argument(
		"--input-csv",
		default="../new_inputs_binned.csv",
		help="Input CSV with per-sample analysis parameters.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(args.parent_dir, args.input_csv)
