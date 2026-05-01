#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter ROIs by dF/F peak and Fano factor criteria and save matching ROIs to CSV.

Saves ROIs where:
- Maximum dF/F peak > 5 OR peak occurs before frame 200 OR
- Fano factor < 0.7 OR Fano factor > 7

Usage:
    python filter_rois.py <parent_dir> [-o <output_csv>]
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd


def calculate_fano_factor_for_roi(fluorescence_data):
	"""Calculate Fano factor for a single ROI.
	
	Fano factor = variance / mean
	
	Args:
		fluorescence_data: array-like of fluorescence values
		
	Returns:
		fano factor value
	"""
	fluorescence_data = np.asarray(fluorescence_data, dtype=float)
	
	mean_val = np.mean(fluorescence_data)
	var_val = np.var(fluorescence_data)
	
	# Avoid division by zero
	if mean_val == 0:
		return np.nan
	else:
		return var_val / mean_val


def get_roi_peak_stats(dff_df, roi):
	"""Get maximum dF/F and frame at which peak occurs for an ROI.
	
	Args:
		dff_df: DataFrame with dF/F data
		roi: ROI number
		
	Returns:
		dict with max_dff and peak_frame
	"""
	roi_data = dff_df[dff_df['ROI'] == roi]
	
	if len(roi_data) == 0:
		return {'max_dff': 0, 'peak_frame': 0}
	
	max_dff = roi_data['dff'].max()
	peak_frame = roi_data.loc[roi_data['dff'].idxmax(), 'frame']
	
	return {'max_dff': max_dff, 'peak_frame': peak_frame}


def process_dff_csv(csv_file):
	"""Process a dF/F CSV and get peak stats for all ROIs.
	
	Args:
		csv_file: path to dff.csv file
		
	Returns:
		DataFrame with max_dff and peak_frame for each ROI
	"""
	df = pd.read_csv(csv_file)
	
	results = []
	for roi in sorted(df['ROI'].unique()):
		stats = get_roi_peak_stats(df, roi)
		stats['ROI'] = int(roi)
		results.append(stats)
	
	return pd.DataFrame(results)


def process_raw_csv(raw_csv_file):
	"""Process a raw fluorescence CSV and calculate Fano factors for all ROIs.
	
	Args:
		raw_csv_file: path to raw.csv file
		
	Returns:
		DataFrame with Fano factor for each ROI
	"""
	df = pd.read_csv(raw_csv_file)
	
	results = []
	for roi in sorted(df['ROI'].unique()):
		roi_data = df[df['ROI'] == roi]['F'].values
		fano = calculate_fano_factor_for_roi(roi_data)
		results.append({'ROI': int(roi), 'fano_factor': fano})
	
	return pd.DataFrame(results)


def save_filtered_rois(output_df, output_dir, output_name="filtered_rois.csv"):
	"""Save ROIs meeting criteria to CSV.
	
	Criteria: 
	- (max_dff > 5 OR peak_frame < 200) OR
	- (fano_factor < 0.7 OR fano_factor > 7)
	
	Args:
		output_df: DataFrame with dF/F and Fano factor stats for all ROIs
		output_dir: directory to save output CSV
		output_name: name of output CSV file
	"""
	# Filter: (max dF/F > 5 OR peak before frame 200) OR (fano < 0.7 OR fano > 7)
	dff_criteria = (output_df['max_dff'] > 5) | (output_df['peak_frame'] < 100)
	fano_criteria = (output_df['fano_factor'] < 0.15) | (output_df['fano_factor'] > 7)
	filtered_df = output_df[dff_criteria | fano_criteria]
	
	if len(filtered_df) > 0:
		result_df = filtered_df[['sample_name', 'ROI']].copy()
		result_df.columns = ['sample', 'roi']
		
		output_file = os.path.join(output_dir, output_name)
		result_df.to_csv(output_file, index=False)
		print(f"\nFiltered ROIs saved to: {output_file}")
		print(f"  Criteria: (max_dff > 5 OR peak_frame < 100) OR (fano < 0.15 OR fano > 7)")
		print(f"  Total ROIs: {len(result_df)}")
		print(f"  - max_dff > 5: {len(filtered_df[filtered_df['max_dff'] > 5])}")
		print(f"  - peak_frame < 100: {len(filtered_df[filtered_df['peak_frame'] < 100])}")
		print(f"  - fano_factor < 0.15: {len(filtered_df[filtered_df['fano_factor'] < 0.15])}")
		print(f"  - fano_factor > 7: {len(filtered_df[filtered_df['fano_factor'] > 7])}")
	else:
		print(f"\nNo ROIs met filtering criteria")


def main(parent_dir, output_file=None):
	"""Calculate dF/F and Fano factor stats and filter ROIs.
	
	Args:
		parent_dir: parent directory containing sample folders with measurements
		output_file: optional output CSV file path for all results
	"""
	measurement_dirs = glob.glob(os.path.join(parent_dir, "**/measurements"), recursive=True)
	
	all_results = []
	
	for measurement_dir in sorted(measurement_dirs):
		# Extract sample directory and name
		sample_dir = os.path.dirname(measurement_dir)
		sample_name = os.path.basename(sample_dir)
		
		dff_csv_files = glob.glob(os.path.join(measurement_dir, "*-dff.csv"))
		raw_csv_files = glob.glob(os.path.join(measurement_dir, "*-raw.csv"))
		
		for dff_csv_file, raw_csv_file in zip(sorted(dff_csv_files), sorted(raw_csv_files)):
			basename = os.path.basename(dff_csv_file)
			
			# Skip files with 'MLB' in the name
			if 'MLB' in basename:
				print(f"Skipping: {basename} (contains 'MLB')")
				continue
			
			print(f"Processing: {basename} from sample {sample_name}")
			
			try:
				# Process dF/F stats
				dff_stats = process_dff_csv(dff_csv_file)
				
				# Process Fano factor from raw data
				fano_stats = process_raw_csv(raw_csv_file)
				
				# Merge on ROI
				roi_stats = dff_stats.merge(fano_stats, on='ROI')
				
				roi_stats['sample_name'] = sample_name
				roi_stats['sample_dir'] = sample_dir
				all_results.append(roi_stats)
			except Exception as e:
				print(f"  Error: {e}")
	
	if all_results:
		output_df = pd.concat(all_results, ignore_index=True)
		
		# Reorder columns
		cols = ['sample_name', 'sample_dir', 'ROI', 'max_dff', 'peak_frame', 'fano_factor']
		output_df = output_df[cols]
		
		# Sort by sample and ROI
		output_df = output_df.sort_values(['sample_name', 'ROI']).reset_index(drop=True)
		
		if output_file:
			output_df.to_csv(output_file, index=False)
			print(f"\nAll results saved to: {output_file}")
		
		print(f"\nTotal ROIs analyzed: {len(output_df)}")
		
		# Save filtered ROIs
		save_filtered_rois(output_df, parent_dir)
		
		return output_df
	else:
		print("No measurement files found to process.")
		return None


def parse_args():
	parser = argparse.ArgumentParser(description="Filter ROIs by dF/F peak criteria.")
	parser.add_argument("parent_dir", help="Parent directory containing sample folders with measurements.")
	parser.add_argument("-o", "--output", help="Output CSV file with all results")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	main(args.parent_dir, args.output)
