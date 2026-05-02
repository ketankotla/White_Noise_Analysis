# White Noise Analysis - In Vivo Imaging Pipeline

## Overview

This pipeline analyzes neural responses to white noise visual stimuli from in vivo two-photon imaging data. It measures fluorescence responses, aligns them to stimuli, extracts spatiotemporal receptive fields (SRF/TRF), and predicts neural responses.


### Utilities & Support
- **ResponseClassSimple_v3.py** → Data structure for response objects
- **ResponseTools_v3.py** → Core response measurement and analysis functions
- **utility.py** → General utility functions (file I/O, image processing)
- **alignment.py** → Core image registration functionality (imported by align_from_lif.py)
- **fit_trf_util.py** → Temporal filter utility functions
- **measure_responses_binned.py** → Alternative: Extract time-binned responses
- **avg_RF.py** → Average receptive field across samples
- **SEM_plot.py** → Statistical plotting utilities

## File Descriptions

### Workflow Scripts

#### [align_from_lif.py](align_from_lif.py)
Parses images from LIF (Leica Image Format) files, aligns them, and generates average projections for masking.
- **Usage**: `python align_from_lif.py` (edit parent_dir and csv_filename inside script)
- **Flags**: None (configuration via internal variables)
- **Inputs**: 
  - LIF files (parent_dir/lif_files/)
  - input_alignment.csv with columns: sample, lif_name, job_index, ch1_name, ch1_index, use_ch2, ch2_name, ch2_index, use_target, target_name, target_start, target_stop, save_avg
- **Outputs**: Aligned image stacks (.tif), average projection (.tif)

#### [measure_responses.py](measure_responses.py)
Extracts fluorescence traces from ROIs across image stacks.
- **Usage**: `python measure_responses.py --parent-dir <parent_dir> --input-csv <input.csv>`
- **Flags**:
  - `--parent-dir`: Parent directory containing sample folders (default: `../all_mi1`)
  - `--input-csv`: Input CSV file (default: `../new_inputs_binned.csv`)
- **Inputs**: 
  - Input CSV with columns: sample_name, ch1_name, mask_name, stimulus_name, ROI, threshold, reporter_name, genotype, compartment, aligned
  - Image stacks (.tif)
  - ROI masks (.tif)
- **Outputs**: Per-ROI fluorescence traces (CSV), plots

#### [align_stim.py](align_stim.py)
Maps ROI response curves to stimulus frames. Joins stimulus frame metadata to each response point and reconstructs time-binned history stimuli.
- **Usage**: `python align_stim.py --parent-dir <parent_dir> --input-csv <input.csv> --workers <num_workers>`
- **Flags**:
  - `--parent-dir`: Parent directory (default: `../all_mi1`)
  - `--input-csv`: Input CSV file (default: `../new_inputs_binned.csv`)
  - `--workers`: Number of worker processes (default: CPU count - 1)
  - `--ignore`: Path to CSV with ROIs to exclude
- **Input CSV Columns**:
  - `skip_first_frames`: (Optional) Number of aligned frames to skip when calculating weighted history TIFs (default: 0)
- **Key Parameters**:
  - `history_before`: 1.5 seconds
  - `history_after`: 0.5 seconds
  - `time_bin_size`: 0.1 seconds
- **Inputs**: 
  - ROI response CSV (from measure_responses.py)
  - Stimulus CSV with raw stim data
- **Outputs**: 
  - Merged response-stimulus table (CSV)
  - Time-binned stimulus history TIFF stacks

#### [find_center_surround.py](find_center_surround.py)
Identifies center and contiguous positive regions of receptive fields from dF/F weighted history. Generates center, surround, and combined SRF masks.
- **Usage**: `python find_center_surround.py <root_dir>`
- **Flags**:
  - `root_dir`: Root directory to search (default: `.`)
  - `--group-by-center-location`: Group files by detected center location
  - `--ignore`: Path to CSV with ROIs to exclude
- **Key Parameters**:
  - `ANALYSIS_START_INDEX`: 10 (frame)
  - `ANALYSIS_END_INDEX`: 15 (frame)
  - `threshold`: 0.5 (z-score)
- **Inputs**: dff-weighted-history TIFF stacks
- **Outputs**: 
  - centers_and_contiguous_sizes.csv with center location, value, and contiguous region
  - surround_masked_*.tif - Surround receptive field 
  - combined_masked_*.tif - Combined center + surround receptive field
  - contiguous_masked_*.tif - Center receptive field

#### [find_trf.py](find_trf.py)
Extracts temporal receptive fields from time-binned stimulus history TIFF stacks.
- **Usage**: `python find_trf.py <root_dir> --centers-csv <centers.csv>`
- **Flags**:
  - `root_dir`: Root directory to search (default: `.`)
  - `--centers-csv`: Path to centers CSV file (default: hardcoded DEFAULT_CENTERS_CSV)
  - `--ignore`: Path to CSV with ROIs to exclude
- **Inputs**: 
  - dff-weighted-history TIFF stacks
  - centers_and_contiguous_sizes.csv
- **Outputs**: TRF CSV (time points × filter values)

### Supporting Scripts

#### [fit_srf.py](fit_srf.py)
Fits 2D spatial receptive field models to individual ROI maps. Supports single Gaussian or Difference of Gaussians (DoG).
- **Usage**: `python fit_srf.py <rf_image.tif> --show-plot --save-params <output.json>`
- **Flags**:
  - `tif_path`: Path to input 2D TIFF file (positional, required)
  - `--gaussian`: Fit single Gaussian instead of DoG
  - `--show-plot`: Display original, fitted, and residual images
  - `--save-fit`: Path to save fitted image (.npy)
  - `--save-params`: Path to save fitted parameters (JSON)
- **Inputs**: 2D TIFF image (RF map)
- **Outputs**: SRF parameters (JSON), fitted image (optional .npy), plots

#### [bin_by_center.py](bin_by_center.py)
Spatially bins responses based on receptive field center location.
- **Usage**: `python bin_by_center.py <root_dir> --ignore <ignore_list.csv>`
- **Flags**:
  - `root_dir`: Root directory to search (default: `.`)
  - `--ignore`: Path to CSV with ROIs to exclude
- **Inputs**: dff-weighted-history TIFF stacks
- **Outputs**: Center-binned response data

#### [filter_rois.py](filter_rois.py)
Filters ROIs by quality metrics: dF/F peak amplitude and Fano factor (variance/mean ratio).
- **Usage**: `python filter_rois.py <parent_dir> -o <output.csv>`
- **Flags**:
  - `parent_dir`: Parent directory (positional, required)
  - `-o, --output`: Output CSV file
- **Key Parameters**:
  - dF/F threshold: 5
  - peak_frame threshold: 200 (frame)
  - Fano factor range: 0.7-7
- **Inputs**: dF/F CSV files from measurement
- **Outputs**: CSV with failing ROI IDs and quality metrics to use with --ignore flag
- **Criteria**: dF/F > 5 OR peak before frame 200 OR Fano factor 0.7-7

#### [predict_resp.py](predict_resp.py)
Predicts neural responses by convolving stimulus with fitted spatial and temporal receptive fields.
- **Usage**: `python predict_resp.py <stimulus.csv> <srf_params.json> <trf_params.csv> --roi <roi_number> --centers-csv <centers.csv> --output <pred.npy>`
- **Positional Arguments**:
  - `stimulus`: Stimulus file (TIFF or CSV)
  - `srf_params`: SRF parameters JSON file
  - `trf_params`: TRF parameters CSV file
- **Flags**:
  - `--roi`: ROI number to center the SRF
  - `--centers-csv`: Path to centers_and_contiguous_sizes.csv
  - `--gcamp-mapping`: Path to GCaMP-stim mapping CSV
  - `--stimulus-rate`: Stimulus frame rate (default: 30.0 Hz)
  - `-o, --output`: Output file path
- **Key Parameters**:
  - `stimulus_rate`: 20.0 (Hz)
- **Inputs**: Stimulus, SRF parameters, TRF parameters
- **Outputs**: Predicted response trace (numpy array or .npy file)

## Data Formats

### CSV Specifications


#### Output CSV (measure_responses.py)
Columns: frame, fluorescence_value, ROI_number, [metadata...]

#### SRF Parameters JSON (fit_srf.py)
```json
{
  "amp": 1.2,
  "x0": 15.5,
  "y0": 14.2,
  "sigma_x": 3.5,
  "sigma_y": 3.2,
  "offset": 0.1
}
```

#### TRF Parameters CSV (find_trf.py)
Columns: time_point, filter_value


## Dependencies

### Core Libraries
- numpy, scipy, scikit-learn
- pandas, seaborn, matplotlib
- tifffile, Pillow (PIL)
- pystackreg (image registration)
- readlif (LIF file parsing)

### Installation
```bash
pip install numpy scipy scikit-learn pandas seaborn matplotlib tifffile Pillow pystackreg readlif
```

## Typical Analysis Workflow

1. **Prepare LIF Images**: 
   ```bash
   python align_from_lif.py  # Edit parent_dir and csv_filename in script
   ```

2. **Measure Raw Responses**: 
   ```bash
   python measure_responses.py --parent-dir ../all_Mi1 --input-csv ../new_inputs_binned.csv
   ```

3. **Align Stimulus**: 
   ```bash
   python align_stim.py --parent-dir ../all_Mi1 --input-csv ../new_inputs_binned.csv --workers 3
   ```

4. **Find Center/Surround**: 
   ```bash
   python find_center_surround.py . --group-by-center-location
   ```

5. **Extract Temporal Filters**: 
   ```bash
   python find_trf.py . --centers-csv centers_and_contiguous_sizes.csv
   ```

6. **Fit Spatial Models** (optional): 
   ```bash
   python fit_srf.py path/to/rf_image.tif --show-plot --save-params srf.json
   ```

7. **Filter ROIs** (optional): 
   ```bash
   python filter_rois.py parent_dir -o filtered_rois.csv
   ```

8. **Predict Responses** (optional): 
   ```bash
   python predict_resp.py stimulus.csv srf_params.json trf_params.csv \
       --roi 25 --centers-csv centers_and_contiguous_sizes.csv --output pred.npy
   ```


## Output Directory Structure

```
sample_name/
├── images/                    # Raw extracted images
├── aligned_images/            # Motion-corrected images
├── masks/                     # ROI masks
├── measurements/              # Measurement outputs and RF-related TIFF stacks
│   ├── dff.csv               # dF/F responses
│   ├── *-dff-weighted-history.tif  # Stimulus history weighted by dF/F
│   └── roi_[N]-history.tif   # Aligned stimulus history for ROI N
├── stim_files/               # Stimulus CSV files
├── plots/                    # Visualization outputs
└── parsed/                   # Parsed stimulus files (if verbose mode)
```