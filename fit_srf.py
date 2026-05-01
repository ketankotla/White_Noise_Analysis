"""Fit a Gaussian or Difference of Gaussians (DoG) model to a 2D TIFF image.

Models receptive field structure. By default fits DoG (center-surround).

Usage:
	python fit_srf.py /path/to/image.tif              # Fit DoG model
	python fit_srf.py /path/to/image.tif --single-gaussian  # Fit single Gaussian

Optional flags:
	--show-plot             Display original image, fitted image, and residuals
	--save-fit <path>       Save fitted image as .npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
	import tifffile
except ImportError as exc:
	raise ImportError(
		"tifffile is required to read TIFF files. Install with: pip install tifffile"
	) from exc


def gaussian_2d(
	xy: tuple[np.ndarray, np.ndarray],
	amp: float,
	x0: float,
	y0: float,
	sigma_x: float,
	sigma_y: float,
	offset: float,
) -> np.ndarray:
	"""Simple 2D Gaussian model.
	
	Models receptive field as a single Gaussian blob.
	"""
	x, y = xy
	exponent = ((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)
	gaussian = amp * np.exp(-exponent)
	return (offset + gaussian).ravel()


def dog_2d(
	xy: tuple[np.ndarray, np.ndarray],
	amp_center: float,
	x0: float,
	y0: float,
	sigma_x_center: float,
	sigma_y_center: float,
	amp_surround: float,
	sigma_x_surround: float,
	sigma_y_surround: float,
	offset: float,
) -> np.ndarray:
	"""Difference of Gaussians model: center gaussian minus surround gaussian.
	
	Models center-surround receptive field structures common in visual neurons.
	amp_surround should be negative (inhibitory surround).
	"""
	x, y = xy
	# Center gaussian (positive)
	exponent_center = ((x - x0) ** 2) / (2 * sigma_x_center**2) + ((y - y0) ** 2) / (2 * sigma_y_center**2)
	center = amp_center * np.exp(-exponent_center)
	# Surround gaussian (typically negative/inhibitory)
	exponent_surround = ((x - x0) ** 2) / (2 * sigma_x_surround**2) + ((y - y0) ** 2) / (2 * sigma_y_surround**2)
	surround = amp_surround * np.exp(-exponent_surround)
	return (offset + center + surround).ravel()


def initial_guess_gaussian(image: np.ndarray) -> tuple:
	"""Estimate initial Gaussian parameters from image moments.
	
	Returns: (amp, x0, y0, sigma_x, sigma_y, offset)
	"""
	offset = float(np.percentile(image, 10))
	adjusted = image - offset
	adjusted_pos = adjusted.copy()
	adjusted_pos[adjusted_pos < 0] = 0

	total_pos = float(np.sum(adjusted_pos))
	ny, nx = image.shape

	if total_pos <= 0:
		# Fallback for problematic images
		x0 = (nx - 1) / 2
		y0 = (ny - 1) / 2
		amp = float(np.max(image) - offset)
		sigma = max(nx / 6, 1.0)
		return amp, x0, y0, sigma, sigma, offset

	# Compute center from positive pixels
	y_idx, x_idx = np.indices(image.shape)
	x0 = float(np.sum(x_idx * adjusted_pos) / total_pos)
	y0 = float(np.sum(y_idx * adjusted_pos) / total_pos)

	sigma_x = float(np.sqrt(np.sum(((x_idx - x0) ** 2) * adjusted_pos) / total_pos))
	sigma_y = float(np.sqrt(np.sum(((y_idx - y0) ** 2) * adjusted_pos) / total_pos))

	sigma_x = max(sigma_x, 1.0)
	sigma_y = max(sigma_y, 1.0)
	amp = float(np.max(image) - offset)

	return amp, x0, y0, sigma_x, sigma_y, offset


def initial_guess(image: np.ndarray) -> tuple:
	"""Estimate initial DoG parameters from image moments.
	
	Returns: (amp_center, x0, y0, sigma_x_center, sigma_y_center, 
	          amp_surround, sigma_x_surround, sigma_y_surround, offset)
	"""
	offset = float(np.percentile(image, 10))
	adjusted = image - offset
	adjusted_pos = adjusted.copy()
	adjusted_pos[adjusted_pos < 0] = 0

	total_pos = float(np.sum(adjusted_pos))
	ny, nx = image.shape

	if total_pos <= 0:
		# Fallback for problematic images
		x0 = (nx - 1) / 2
		y0 = (ny - 1) / 2
		amp_center = float(np.max(image) - offset)
		amp_surround = float(np.min(image) - offset)  # Usually negative
		amp_surround = min(amp_surround, -0.1)  # Ensure negative
		sigma_center = max(nx / 6, 1.0)
		sigma_surround = max(nx / 3, 2.0)
		return amp_center, x0, y0, sigma_center, sigma_center, amp_surround, sigma_surround, sigma_surround, offset

	# Compute center from positive pixels
	y_idx, x_idx = np.indices(image.shape)
	x0 = float(np.sum(x_idx * adjusted_pos) / total_pos)
	y0 = float(np.sum(y_idx * adjusted_pos) / total_pos)

	sigma_x_center = float(np.sqrt(np.sum(((x_idx - x0) ** 2) * adjusted_pos) / total_pos))
	sigma_y_center = float(np.sqrt(np.sum(((y_idx - y0) ** 2) * adjusted_pos) / total_pos))

	sigma_x_center = max(sigma_x_center, 1.0)
	sigma_y_center = max(sigma_y_center, 1.0)
	amp_center = float(np.max(image) - offset)

	# Estimate surround from negative pixels and overall width
	amp_surround = float(np.min(image) - offset)  # Usually negative
	amp_surround = min(amp_surround, -0.01 * amp_center)  # Scale with center amplitude
	# Surround typically has larger spatial extent
	sigma_x_surround = sigma_x_center * 2.0
	sigma_y_surround = sigma_y_center * 2.0

	return amp_center, x0, y0, sigma_x_center, sigma_y_center, amp_surround, sigma_x_surround, sigma_y_surround, offset


def fit_dog_to_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Fit a Difference of Gaussians model and return optimal parameters and covariance."""
	if image.ndim != 2:
		raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

	image = image.astype(float)
	ny, nx = image.shape

	y_idx, x_idx = np.indices((ny, nx))
	p0 = initial_guess(image)

	# Bounds for: amp_center, x0, y0, sigma_x_center, sigma_y_center,
	#             amp_surround, sigma_x_surround, sigma_y_surround, offset
	lower_bounds = [0.0, 0.0, 0.0, 1e-6, 1e-6, -np.inf, 1e-6, 1e-6, -np.inf]
	upper_bounds = [np.inf, nx - 1, ny - 1, np.inf, np.inf, 0.0, np.inf, np.inf, np.inf]

	popt, pcov = curve_fit(
		dog_2d,
		(x_idx, y_idx),
		image.ravel(),
		p0=p0,
		bounds=(lower_bounds, upper_bounds),
		maxfev=20000,
	)
	return popt, pcov


def fit_gaussian_to_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Fit a single Gaussian model and return optimal parameters and covariance."""
	if image.ndim != 2:
		raise ValueError(f"Expected a 2D image, got shape {image.shape}.")

	image = image.astype(float)
	ny, nx = image.shape

	y_idx, x_idx = np.indices((ny, nx))
	p0 = initial_guess_gaussian(image)

	# Bounds for: amp, x0, y0, sigma_x, sigma_y, offset
	lower_bounds = [0.0, 0.0, 0.0, 1e-6, 1e-6, -np.inf]
	upper_bounds = [np.inf, nx - 1, ny - 1, np.inf, np.inf, np.inf]

	popt, pcov = curve_fit(
		gaussian_2d,
		(x_idx, y_idx),
		image.ravel(),
		p0=p0,
		bounds=(lower_bounds, upper_bounds),
		maxfev=20000,
	)
	return popt, pcov


def save_srf_parameters(params: np.ndarray, param_names: list, output_path: Path, is_dog: bool = True) -> None:
	"""Save fitted SRF parameters to a JSON file.
	
	Args:
		params: Array of fitted parameters
		param_names: List of parameter names in order
		output_path: Path to save the JSON file
		is_dog: Whether the model is Difference of Gaussians (True) or single Gaussian (False)
	"""
	params_dict = {
		"model_type": "DoG" if is_dog else "Gaussian",
		"parameters": {name: float(value) for name, value in zip(param_names, params)}
	}
	
	with open(output_path, 'w') as f:
		json.dump(params_dict, f, indent=2)
	
	print(f"Saved SRF parameters to: {output_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fit a 2D model (Gaussian or Difference of Gaussians) to a TIFF image.")
	parser.add_argument("tif_path", type=Path, help="Path to input 2D TIFF file")
	parser.add_argument(
		"--gaussian",
		action="store_true",
		help="Fit a single Gaussian model instead of Difference of Gaussians",
	)
	parser.add_argument(
		"--show-plot",
		action="store_true",
		help="Display original image, fitted image, and residuals",
	)
	parser.add_argument(
		"--save-fit",
		type=Path,
		default=None,
		help="Optional path to save fitted image as .npy",
	)
	parser.add_argument(
		"--save-params",
		type=Path,
		default=None,
		help="Optional path to save fitted parameters as .json",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.tif_path.exists():
		raise FileNotFoundError(f"File not found: {args.tif_path}")

	image = tifffile.imread(args.tif_path)
	if image.ndim != 2:
		raise ValueError(
			f"Input TIFF must be 2D. Got shape {image.shape}. "
			"If this is a stack, select one frame before fitting."
		)

	if args.gaussian:
		# Fit single Gaussian
		popt, _ = fit_gaussian_to_image(image)
		amp, x0, y0, sigma_x, sigma_y, offset = popt

		print("Fitted Gaussian parameters:")
		print(f"  amplitude:  {amp:.6g}")
		print(f"  sigma_x:    {sigma_x:.6g}")
		print(f"  sigma_y:    {sigma_y:.6g}")
		print(f"  x0:         {x0:.6g}")
		print(f"  y0:         {y0:.6g}")
		print(f"  offset:     {offset:.6g}")

		if args.save_params is not None:
			param_names = ["amplitude", "x0", "y0", "sigma_x", "sigma_y", "offset"]
			save_srf_parameters(popt, param_names, args.save_params, is_dog=False)

		ny, nx = image.shape
		y_idx, x_idx = np.indices((ny, nx))
		fitted = gaussian_2d((x_idx, y_idx), *popt).reshape(image.shape)
	else:
		# Fit Difference of Gaussians (default)
		popt, _ = fit_dog_to_image(image)
		amp_center, x0, y0, sigma_x_center, sigma_y_center, amp_surround, sigma_x_surround, sigma_y_surround, offset = popt

		print("Fitted Difference of Gaussians parameters:")
		print(f"  Center component:")
		print(f"    amplitude:  {amp_center:.6g}")
		print(f"    sigma_x:    {sigma_x_center:.6g}")
		print(f"    sigma_y:    {sigma_y_center:.6g}")
		print(f"  Surround component (inhibitory):")
		print(f"    amplitude:  {amp_surround:.6g}")
		print(f"    sigma_x:    {sigma_x_surround:.6g}")
		print(f"    sigma_y:    {sigma_y_surround:.6g}")
		print(f"  Shared parameters:")
		print(f"    x0:         {x0:.6g}")
		print(f"    y0:         {y0:.6g}")
		print(f"    offset:     {offset:.6g}")

		if args.save_params is not None:
			param_names = ["amp_center", "x0", "y0", "sigma_x_center", "sigma_y_center", "amp_surround", "sigma_x_surround", "sigma_y_surround", "offset"]
			save_srf_parameters(popt, param_names, args.save_params, is_dog=True)

		ny, nx = image.shape
		y_idx, x_idx = np.indices((ny, nx))
		fitted = dog_2d((x_idx, y_idx), *popt).reshape(image.shape)

	if args.save_fit is not None:
		np.save(args.save_fit, fitted)
		print(f"Saved fitted image to: {args.save_fit}")

	if args.show_plot:
		import matplotlib.pyplot as plt
		from matplotlib.gridspec import GridSpec

		residual = image.astype(float) - fitted

		fig, axes = plt.subplots(1, 3, figsize=(14, 4))
		axes[0].imshow(image, cmap="viridis")
		axes[0].set_title("Original")
		axes[1].imshow(fitted, cmap="viridis")
		axes[1].set_title("Fitted Model")
		axes[2].imshow(residual, cmap="coolwarm")
		axes[2].set_title("Residual")
		for ax in axes:
			ax.set_axis_off()
		plt.tight_layout()
		plt.show()

		# Create a second figure with SRF and 1D Gaussian profiles
		# Using GridSpec for better layout control
		fig2 = plt.figure(figsize=(10, 10))
		gs = GridSpec(2, 2, figure=fig2, width_ratios=[1, 0.2], height_ratios=[0.2, 1], 
		              hspace=0.05, wspace=0.05)

		# Main 2D SRF plot
		ax_main = fig2.add_subplot(gs[1, 0])
		ax_main.imshow(fitted, cmap="viridis", origin="lower")
		ax_main.set_xlabel("X")
		ax_main.set_ylabel("Y")

		# X direction profile (top)
		ax_top = fig2.add_subplot(gs[0, 0], sharex=ax_main)
		ax_top.set_title("Predicted SRF", fontsize=14, pad=10)
		ax_top.set_ylabel("Response")

		# Y direction profile (right, rotated)
		ax_right = fig2.add_subplot(gs[1, 1], sharey=ax_main)
		ax_right.set_xlabel("Response")

		# Create 1D Gaussian profiles with fine interpolation for smooth curves
		x_profile = np.linspace(0, nx - 1, 500)
		y_profile = np.linspace(0, ny - 1, 500)

		if args.gaussian:
			amp, x0_val, y0_val, sigma_x_val, sigma_y_val, offset_val = popt
			x_gaussian_1d = amp * np.exp(-((x_profile - x0_val) ** 2) / (2 * sigma_x_val**2)) + offset_val
			y_gaussian_1d = amp * np.exp(-((y_profile - y0_val) ** 2) / (2 * sigma_y_val**2)) + offset_val
		else:
			amp_center, x0_val, y0_val, sigma_x_center_val, sigma_y_center_val, amp_surround, sigma_x_surround, sigma_y_surround, offset_val = popt
			# Center component
			x_gaussian_center = amp_center * np.exp(-((x_profile - x0_val) ** 2) / (2 * sigma_x_center_val**2))
			y_gaussian_center = amp_center * np.exp(-((y_profile - y0_val) ** 2) / (2 * sigma_y_center_val**2))
			# Surround component
			x_gaussian_surround = amp_surround * np.exp(-((x_profile - x0_val) ** 2) / (2 * sigma_x_surround**2))
			y_gaussian_surround = amp_surround * np.exp(-((y_profile - y0_val) ** 2) / (2 * sigma_y_surround**2))
			# Combined
			x_gaussian_1d = x_gaussian_center + x_gaussian_surround + offset_val
			y_gaussian_1d = y_gaussian_center + y_gaussian_surround + offset_val

		# Plot X direction Gaussian (above main plot)
		ax_top.plot(x_profile, x_gaussian_1d, linewidth=2, color="blue")
		ax_top.grid(True, alpha=0.3)
		ax_top.tick_params(labelbottom=False)

		# Plot Y direction Gaussian (to the right, rotated 90 degrees)
		ax_right.plot(y_gaussian_1d, y_profile, linewidth=2, color="red")
		ax_right.grid(True, alpha=0.3)
		ax_right.tick_params(labelleft=False)

		plt.show()


if __name__ == "__main__":
	main()
