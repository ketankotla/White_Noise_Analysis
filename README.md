# White Noise Analysis - In Vivo Imaging Pipeline

## Overview

This pipeline analyzes neural responses to white noise visual stimuli from in vivo two-photon imaging data. It measures fluorescence responses, aligns them to stimuli, extracts spatiotemporal receptive fields (SRF/TRF), and predicts neural responses.

## Workflow Order

1. **align_from_lif.py** → Extract and align images from LIF files
   ```bash
   python align_from_lif.py  # Reads input_alignment.csv from parent directory
   ```

2. **measure_responses.py** → Extract fluorescence traces from ROIs
   ```bash
   python measure_responses.py path/to/input.csv
   ```

3. **align_stim.py** → Align ROI responses to stimulus frames and reconstruct stimulus history
   ```bash
   python align_stim.py path/to/input.csv --output-dir path/to/output
   ```

4. **find_center_surround.py** → Identify center/surround region of receptive fields
   ```bash
   python find_center_surround.py path/to/measurements/dir --output centers_and_contiguous_sizes.csv
   ```

5. **find_trf.py** → Extract temporal receptive fields (TRF)
   ```bash
   python find_trf.py path/to/measurements/dir --centers-csv centers_and_contiguous_sizes.csv --output-dir path/to/output
   ```

### Supporting Scripts
- **alignment.py** → Core image registration functionality (imported by align_from_lif.py)
  ```python
  from alignment import registerImage, transformImage
  ```

- **measure_responses_binned.py** → Alternative: Extract time-binned responses
  ```bash
  python measure_responses_binned.py  # Reads inputs_binned.csv from parent directory
  ```

- **fit_srf.py** → Fit spatial receptive field (SRF) with Gaussian/DoG model
  ```bash
  python fit_srf.py path/to/rf_image.tif --show-plot --save-fit output.npy
  ```

- **bin_by_center.py** → Bin responses based on RF center location
  ```bash
  python bin_by_center.py path/to/measurements/dir --centers-csv centers_and_contiguous_sizes.csv
  ```

- **filter_rois.py** → Filter ROIs by quality metrics (dF/F peak, Fano factor)
  ```bash
  python filter_rois.py path/to/parent/dir -o filtered_rois.csv
  ```

- **predict_resp.py** → Predict responses using fitted SRF and TRF models
  ```bash
  python predict_resp.py stimulus.csv srf_params.json trf_params.csv \
      --roi 25 --centers-csv centers_and_contiguous_sizes.csv --output pred.npy
  ```

- **flash_analysis.py** → Analyze flash stimulus responses
  ```python
  from flash_analysis import extract_flash_data, plot_flash_and_pred
  ```

### Utilities & Support
- **ResponseClassSimple_v3.py** → Data structure for response objects
- **ResponseTools_v3.py** → Core response measurement and analysis functions
- **utility.py** → General utility functions (file I/O, image processing)
- **convolve.py** → Convolution and nonlinearity functions
- **returnmeans.py** → Statistical averaging and fitting functions
- **windowing.py** → Sliding window utilities
- **fit_trf_util.py** → Temporal filter utility functions
- **avg_RF.py** → Average receptive field across samples
- **SEM_plot.py** → Statistical plotting utilities

## File Descriptions

### Image Alignment & Processing

#### [align_from_lif.py](align_from_lif.py)
Parses images from LIF (Leica Image Format) files, aligns them, and generates average projections for masking.
- **Inputs**: LIF files, input CSV with channel specifications
- **Outputs**: Aligned image stacks (.tif), average projection for masking
- **Key Parameters**: Channel index, target frame range, alignment mode

#### [alignment.py](alignment.py)
Core image registration module providing functions for image-to-reference alignment using various transformation modes (translation, rigid, rotation, affine, bilinear).
- **Key Functions**:
  - `registerImage()` - Register image to reference
  - `transformImage()` - Apply transformation matrix to image
- **Modes**: translation, rigid, rotation, affine, bilinear

### Stimulus & Response Alignment

#### [align_stim.py](align_stim.py)
Maps ROI response curves to stimulus frames. Joins stimulus frame metadata to each response point, filters sufficient context, and reconstructs time-binned history stimuli as 16×16 arrays.
- **Inputs**: 
  - ROI response CSV (from measure_responses.py)
  - Stimulus CSV file with raw stim data
  - Optional: Ignore list CSV
- **Outputs**: 
  - Merged response-stimulus table (CSV)
  - Time-binned stimulus history as TIFF stacks (frames × 16 × 16)
  - Filtered rows with valid temporal context (1.5s history before, 0.5s after)
- **Key Parameters**: 
  - history_before=1.5s, history_after=0.5s
  - time_bin_size=0.1s
  - Output dtype: float16 (TIFF), float32 (compute)

#### [ResponseTools_v3.py](ResponseTools_v3.py)
Core library for response measurement and stimulus processing.
- **Key Functions**:
  - `count_frames()` - Align stimulus frames to imaging frames
  - `measure_one_ROI()` - Extract single ROI fluorescence
  - `measure_multiple_ROIs()` - Extract all ROI fluorescence
  - `measure_multiple_thresholded_ROIs()` - Extract ROIs above threshold
  - `bin_images()` - Time-bin images by stimulus epoch
- **Input Format**: 
  - Stimulus CSV with global time index and frame trigger column
  - Image stacks (3D arrays)
  - ROI masks
- **Output Format**: Fluorescence traces per ROI

### Response Measurement

#### [measure_responses.py](measure_responses.py)
Extracts fluorescence traces from ROIs across image stacks. Reads one row from input CSV per execution, measuring single/multiple/thresholded ROIs and saving raw response curves.
- **Inputs**:
  - Input CSV with sample name, image file, mask file specifications
  - Image stacks (.tif files)
  - ROI masks (.tif files)
- **Outputs**:
  - Per-ROI fluorescence traces (CSV)
  - Plots of fluorescence vs. frame
- **CSV Configuration**: 
  - Columns: sample_name, ch1_name, mask_name, stimulus_name, ROI (one/all/thresholded), threshold
  - Optional: aligned flag

#### [measure_responses_binned.py](measure_responses_binned.py)
Extracts time-binned fluorescence responses synchronized with stimulus epochs. Bins images by stimulus event timing.
- **Inputs**:
  - Input CSV (inputs_binned.csv) with binning configuration
  - Aligned image stacks
  - ROI masks
  - Stimulus files
- **Outputs**:
  - Time-binned fluorescence per ROI (CSV)
  - Binned plots
- **Binning**: Groups frames by stimulus epoch with configurable time bins

### Receptive Field Extraction & Fitting

#### [find_trf.py](find_trf.py)
Extracts temporal receptive fields from time-binned stimulus history TIFF stacks. Loads history files, extracts center pixel TRF, and Z-score normalizes.
- **Inputs**:
  - dff-weighted-history TIFF stacks (time × 16 × 16)
  - Centers and contiguous sizes CSV
- **Outputs**:
  - TRF CSV (time points × filter values)
  - Optional: Plots
- **Processing**:
  - Analysis window: frames 10-15
  - Z-score normalization
  - Weighted by dF/F

#### [fit_srf.py](fit_srf.py)
Fits 2D spatial receptive field models to individual ROI maps. Supports both single Gaussian and Difference of Gaussians (DoG, default) fitting.
- **Inputs**:
  - 2D TIFF image (RF map)
  - Optional: Single Gaussian flag
- **Outputs**:
  - SRF parameters JSON: {amp, x0, y0, sigma_x, sigma_y, offset}
  - Optional: Fitted image (.npy)
  - Optional: Plots (original, fitted, residuals)
- **Models**:
  - Single Gaussian: 6 parameters
  - DoG (center-surround): 10 parameters

#### [find_center_surround.py](find_center_surround.py)
Identifies center and contiguous positive regions of receptive fields from dF/F weighted history.
- **Inputs**: dff-weighted-history TIFF stacks
- **Outputs**: 
  - centers_and_contiguous_sizes.csv with center location, value, and contiguous region
- **Processing**:
  - Averages frames 10-15 (analysis window)
  - Z-score normalizes pixels
  - Finds max pixel (center)
  - 4-connected flood fill for positive regions (z-score > 0.5)

#### [bin_by_center.py](bin_by_center.py)
Spatially bins responses based on receptive field center location to extract local receptive field structure.
- **Inputs**: 
  - dff-weighted-history TIFF stacks
  - Center coordinates
- **Outputs**: Center-binned response data
- **Analysis Window**: Frames 10-15 of history, z-score normalized

#### [fit_trf_util.py](fit_trf_util.py)
Utility functions for temporal filter fitting with bandpass model.
- **Functions**:
  - `bp()` - Bandpass filter with two time constants
- **Parameters**: tau1, tau2, scale, offset

### Receptive Field Analysis & Visualization

#### [avg_RF.py](avg_RF.py)
Averages receptive field maps across multiple samples and generates statistics on peak pixel locations.
- **Inputs**: Directory structure with measurement TIFF files
- **Outputs**:
  - avg_RF.tif - Averaged receptive field
  - max_pixels.tif - Histogram of peak pixel locations
- **Processing**: 
  - Normalizes each RF to [0,1]
  - Averages across samples
  - Tracks peak pixel frequency

#### [filter_rois.py](filter_rois.py)
Filters ROIs by quality metrics: dF/F peak amplitude and Fano factor (variance/mean ratio).
- **Inputs**: dF/F CSV files from measurement
- **Outputs**: CSV with passing ROI IDs and quality metrics
- **Filtering Criteria**:
  - Maximum dF/F > 5, OR
  - Peak occurs before frame 200, OR
  - Fano factor < 0.7 OR > 7

### Response Prediction

#### [predict_resp.py](predict_resp.py)
Predicts neural responses by convolving stimulus with fitted spatial and temporal receptive fields.
- **Inputs**:
  - Stimulus (TIFF or CSV)
  - SRF parameters (JSON from fit_srf.py)
  - TRF parameters (CSV with time and filter values)
  - Optional: ROI number, center CSV, GCaMP-stim mapping
- **Outputs**: Predicted response trace (1D numpy array or .npy file)
- **Process**:
  - Extracts spatial RF as 2D Gaussian/DoG
  - Centers on ROI if specified
  - Convolves stimulus frames with spatial RF
  - Convolves stimulus convolution with temporal RF
  - Optional: Maps to recorded frame range using GCaMP timing

**Usage Examples**:
```bash
# Basic prediction
python predict_resp.py stimulus.csv srf_params.json trf_params.csv --output pred.npy

# With ROI centering and GCaMP mapping
python predict_resp.py stimulus.csv srf_params.json trf_params.csv \
    --roi 5 --centers-csv centers_and_contiguous_sizes.csv \
    --gcamp-mapping Gcamp-stim-mapped.csv --output pred.npy
```

### Analysis & Visualization

#### [flash_analysis.py](flash_analysis.py)
Analyzes neural responses to flash stimuli and predicts responses using white noise filters.
- **Key Functions**:
  - `extract_flash_data()` - Extract and average flash responses
  - `flash_wn_pred()` - Predict flash responses from white noise TRF
  - `plot_flash_data_saline_OA()` - Compare saline vs. octopamine
  - `plot_flash_and_pred()` - Plot measured and predicted responses
- **Parameters**: Window size, baseline subtraction mode, detrending

#### [SEM_plot.py](SEM_plot.py)
Creates publication-quality plots with error bars (SEM) from averaged neural data.
- **Inputs**: CSV with averaged response data (time, response, condition)
- **Outputs**: PNG plots with seaborn styling
- **Features**: Line plots with error bars, despined axes, customizable axes

#### [ResponseClassSimple_v3.py](ResponseClassSimple_v3.py)
Data class for storing individual ROI response information.
- **Attributes**: 
  - sample_name, reporter, genotype, compartment
  - stimulus_name, ROI_number
  - F (fluorescence trace)

### Utility Functions

#### [utility.py](utility.py)
General-purpose utilities for file I/O and image processing.
- **Key Functions**:
  - `get_file_names()` - Find files by type and label
  - `read_csv_file()` - Load CSV with optional header
  - `write_csv()` - Write data to CSV
  - `read_tifs()` - Load multi-page TIFF stacks
  - `alignMultiPageTiff()` - Align TIFF sequence
  - `loadLifFile()` - Parse LIF format files
  - `getLifImage()` - Extract specific image from LIF

#### [convolve.py](convolve.py)
Convolution and nonlinear transformation functions for filter-based prediction.
- **Key Functions**:
  - `sigmoid()` - Parameterized sigmoid nonlinearity
  - `softplus()` - Softplus nonlinearity
- **Uses**: Fast FFT-based convolution, Scikit-learn regression

#### [returnmeans.py](returnmeans.py)
Functions for averaging and fitting receptive field parameters.
- **Key Functions**:
  - `return_mean_temporal()` - Average temporal filters
  - `return_mean_spatial()` - Average spatial filters
  - `return_mean_nonlin()` - Average nonlinearity functions
  - `fit_nonlin()` - Fit nonlinearity to data
- **Uses**: Statistical binning, curve fitting

#### [windowing.py](windowing.py)
Sliding window utilities for signal processing.
- **Key Functions**:
  - `sliding_window()` - Generate sliding windows over data
- **Parameters**: Size, stepsize, axis, copy mode

## Data Formats

### CSV Specifications

#### Input CSV (measure_responses.py)
```
sample_name,ch1_name,mask_name,stimulus_name,ROI,threshold,reporter_name,genotype,compartment,aligned
sample_001,green,mask_1,ternary_noise,all,,GCaMP,wild_type,axon,TRUE
```

#### Input CSV (align_from_lif.py - input_alignment.csv)
```
sample,lif_name,job_index,ch1_name,ch1_index,use_ch2,ch2_name,ch2_index,use_target,target_name,target_start,target_stop,save_avg
sample_001,imaging_001,0,green_channel,0,FALSE,red_channel,1,FALSE,avg_reference,0,50,TRUE
```

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

### Image Formats
- **TIFF**: Grayscale 8/16/32-bit, multi-page stacks supported
- **LIF**: Leica Image Format (requires readlif library)
- **Stimulus Arrays**: 16×16 pixel spatial dimensions, variable time bins

## Key Parameters

| Component | Parameter | Default | Unit |
|-----------|-----------|---------|------|
| align_stim.py | history_before | 1.5 | seconds |
| align_stim.py | history_after | 0.5 | seconds |
| align_stim.py | time_bin_size | 0.1 | seconds |
| find_center_surround.py | ANALYSIS_START_INDEX | 10 | frame |
| find_center_surround.py | ANALYSIS_END_INDEX | 15 | frame |
| find_center_surround.py | threshold | 0.5 | z-score |
| filter_rois.py | dF/F threshold | 5 | - |
| filter_rois.py | peak_frame threshold | 200 | frame |
| filter_rois.py | Fano factor range | 0.7-7 | - |
| predict_resp.py | stimulus_rate | 30.0 | Hz |

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

1. **Prepare LIF Images**: Run `align_from_lif.py` with input_alignment.csv
2. **Measure Raw Responses**: Run `measure_responses.py` for each sample/ROI
3. **Align Stimulus**: Run `align_stim.py` to synchronize responses with stimulus frames and generate stimulus history
4. **Find Center/Surround**: Run `find_center_surround.py` to identify RF center and contiguous positive regions
5. **Extract Temporal Filters**: Run `find_trf.py` to extract temporal receptive fields
6. **Fit Spatial Models** (optional): Run `fit_srf.py` for spatial receptive fields
7. **Filter ROIs** (optional): Run `filter_rois.py` to select high-quality ROIs
8. **Predict Responses** (optional): Run `predict_resp.py` to test RF model predictions
9. **Analyze Results** (optional): Use `flash_analysis.py` and `SEM_plot.py` for visualization

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