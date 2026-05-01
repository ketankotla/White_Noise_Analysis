"""
Fit temporal receptive field (TRF) curves to biphasic functions

Utilizes the bp() function and fitting utilities from flash_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from flash_analysis import bp


# ============================================================================
# ALTERNATIVE MODELS - in case biphasic doesn't fit well
# ============================================================================

def exponential_decay(t, tau, scale, offset=0):
    """Simple exponential decay"""
    return scale * np.exp(-t / tau) + offset

def exponential_rise_decay(t, tau1, tau2, scale, offset=0):
    """Rise and decay (alpha function / convolution of two exponentials)"""
    # Handle edge case where tau1 == tau2
    if np.isscalar(tau1) and np.isscalar(tau2) and tau1 == tau2:
        # Limit case: t * exp(-t/tau) / tau^2
        return scale * (t / (tau1 ** 2)) * np.exp(-t / tau1) + offset
    else:
        # Standard alpha function: (exp(-t/tau2) - exp(-t/tau1)) / (tau2 - tau1)
        denom = tau2 - tau1
        if np.isscalar(denom) and denom == 0:
            return scale * (t / (tau1 ** 2)) * np.exp(-t / tau1) + offset
        return scale * (np.exp(-t / tau2) - np.exp(-t / tau1)) / denom + offset

def gaussian_impulse(t, t_peak, width, scale, offset=0):
    """Gaussian-shaped impulse response"""
    return scale * np.exp(-((t - t_peak) ** 2) / (2 * width ** 2)) + offset

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def read_trf_csv(csv_path, time_col='time_s', response_col='trf', delimiter=','):
    """
    Read a TRF curve from a CSV file using numpy (no pandas required)
    
    Args
    ----
    csv_path : str
        Path to CSV file containing TRF data
    time_col : str or int
        Column name or index for time values
    response_col : str or int
        Column name or index for response values
    delimiter : str
        CSV delimiter (default: ',')
    
    Returns
    -------
    tdata : np.ndarray
        Time values
    ydata : np.ndarray
        Response values
    """
    # Read header to find column indices
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(delimiter)
    
    # Find column indices
    if isinstance(time_col, str):
        time_idx = header.index(time_col)
    else:
        time_idx = time_col
    
    if isinstance(response_col, str):
        response_idx = header.index(response_col)
    else:
        response_idx = response_col
    
    # Load data, skipping header
    data = np.loadtxt(csv_path, delimiter=delimiter, skiprows=1)
    
    tdata = data[:, time_idx]
    ydata = data[:, response_idx]
    
    return tdata, ydata


def compare_models(csv_path, time_col='time_s', response_col='trf', delimiter=','):
    """
    Try multiple models and compare fit quality
    
    Args
    ----
    csv_path : str
        Path to CSV file
    time_col : str or int
        Column name or index for time
    response_col : str or int
        Column name or index for response
    delimiter : str
        CSV delimiter
    
    Returns
    -------
    results : dict
        Dictionary with R² values and parameters for each model
    """
    tdata, ydata = read_trf_csv(csv_path, time_col=time_col, 
                                 response_col=response_col, delimiter=delimiter)
    
    results = {}
    
    # Model 1: Biphasic (from flash_analysis)
    try:
        fit_results = fit_trf_biphasic(tdata, ydata, verbose=False, plot=False)
        results['Biphasic'] = {
            'r2': fit_results['r2'],
            'popt': fit_results['popt'],
            'yfit': fit_results['yfit']
        }
    except Exception as e:
        results['Biphasic'] = {'r2': -1, 'error': str(e)}
        print(f"  ⚠ Biphasic fitting error: {e}")
    
    # Model 2: Exponential decay
    try:
        p0 = [0.1, np.max(np.abs(ydata)), np.mean(ydata)]
        popt, _ = curve_fit(exponential_decay, tdata, ydata, p0=p0, maxfev=10000)
        yfit = exponential_decay(tdata, *popt)
        ss_res = np.sum((ydata - yfit) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        results['Exponential Decay'] = {
            'r2': r2,
            'popt': popt,
            'yfit': yfit
        }
    except Exception as e:
        results['Exponential Decay'] = {'r2': -1, 'error': str(e)}
        print(f"  ⚠ Exponential Decay fitting error: {e}")
    
    # Model 3: Rise-Decay (alpha function)
    try:
        # Better initial parameter estimation
        peak_idx = np.argmax(np.abs(ydata))
        peak_time = tdata[peak_idx]
        scale_est = np.max(np.abs(ydata))
        offset_est = np.mean(ydata)
        
        # For rise-decay: tau1 << tau2 typically
        # tau1 is rise time, tau2 is decay time
        tau1_est = max(tdata[1] - tdata[0], peak_time / 10)  # small rise time
        tau2_est = peak_time * 1.5  # longer decay time
        
        p0 = [tau1_est, tau2_est, scale_est, offset_est]
        popt, _ = curve_fit(exponential_rise_decay, tdata, ydata, p0=p0, maxfev=10000)
        yfit = exponential_rise_decay(tdata, *popt)
        ss_res = np.sum((ydata - yfit) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        results['Rise-Decay'] = {
            'r2': r2,
            'popt': popt,
            'yfit': yfit
        }
    except Exception as e:
        results['Rise-Decay'] = {'r2': -1, 'error': str(e)}
        print(f"  ⚠ Rise-Decay fitting error: {e}")
    
    # Model 4: Gaussian
    try:
        peak_idx = np.argmax(np.abs(ydata))
        t_peak = tdata[peak_idx]
        p0 = [t_peak, (tdata[-1] - tdata[0]) / 4, np.max(np.abs(ydata)), np.mean(ydata)]
        popt, _ = curve_fit(gaussian_impulse, tdata, ydata, p0=p0, maxfev=10000)
        yfit = gaussian_impulse(tdata, *popt)
        ss_res = np.sum((ydata - yfit) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        results['Gaussian'] = {
            'r2': r2,
            'popt': popt,
            'yfit': yfit
        }
    except Exception as e:
        results['Gaussian'] = {'r2': -1, 'error': str(e)}
        print(f"  ⚠ Gaussian fitting error: {e}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"Model Comparison Results")
    print(f"{'='*60}")
    
    # Separate successful and failed models
    successful = [(k, v) for k, v in results.items() if 'r2' in v and v['r2'] > -1]
    failed = [(k, v) for k, v in results.items() if 'r2' in v and v['r2'] <= -1]
    
    # Sort successful models by R²
    sorted_results = sorted(successful, key=lambda x: x[1]['r2'], reverse=True)
    
    # Print successful models
    print("\n✓ SUCCESSFUL FITS:")
    if sorted_results:
        for i, (model_name, model_result) in enumerate(sorted_results, 1):
            r2 = model_result['r2']
            status = "✓" if r2 > 0.7 else "~" if r2 > 0.5 else "✗"
            print(f"{i}. {model_name:20s} R² = {r2:.4f}  {status}")
    else:
        print("  (none)")
    
    # Print failed models with error messages
    if failed:
        print("\n✗ FAILED FITS:")
        for model_name, model_result in failed:
            error_msg = model_result.get('error', 'Unknown error')
            print(f"  • {model_name:20s} {error_msg}")
    
    # Plot comparison of best models
    if sorted_results:
        fig, axes = plt.subplots(1, min(3, len(sorted_results)), figsize=(15, 4))
        if len(sorted_results) == 1:
            axes = [axes]
        
        for idx, (model_name, model_result) in enumerate(sorted_results[:3]):
            ax = axes[idx] if len(sorted_results) > 1 else axes[0]
            ax.plot(tdata, ydata, 'o', label='Data', alpha=0.7, markersize=3, color='black')
            ax.plot(tdata, model_result['yfit'], '-', linewidth=2, label='Fit', color='red')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Response')
            ax.set_title(f"{model_name}\n(R² = {model_result['r2']:.4f})")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    return results


def diagnose_trf_data(csv_path, time_col='time_s', response_col='trf', delimiter=','):
    """
    Visualize TRF data to understand its shape and determine if biphasic is appropriate
    
    Args
    ----
    csv_path : str
        Path to CSV file
    time_col : str or int
        Column name or index for time
    response_col : str or int
        Column name or index for response
    delimiter : str
        CSV delimiter
    """
    tdata, ydata = read_trf_csv(csv_path, time_col=time_col, 
                                 response_col=response_col, delimiter=delimiter)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Plot 1: Raw data
    axes[0, 0].plot(tdata, ydata, 'o-', linewidth=2, markersize=3, color='black')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Response')
    axes[0, 0].set_title('Raw TRF Data')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    
    # Plot 2: Normalized
    ydata_norm = (ydata - np.min(ydata)) / (np.max(ydata) - np.min(ydata))
    axes[0, 1].plot(tdata, ydata_norm, 'o-', linewidth=2, markersize=3, color='blue')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Normalized Response')
    axes[0, 1].set_title('Normalized TRF Data')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    
    # Plot 3: Log scale (if positive)
    if np.all(ydata > 0):
        axes[1, 0].semilogy(tdata, np.abs(ydata), 'o-', linewidth=2, markersize=3, color='green')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Response (log scale)')
        axes[1, 0].set_title('Log-scale View')
        axes[1, 0].grid(True, alpha=0.3, which='both')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
    else:
        axes[1, 0].semilogy(tdata, np.abs(ydata), 'o-', linewidth=2, markersize=3, color='green')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('|Response| (log scale)')
        axes[1, 0].set_title('Log-scale View (absolute values)')
        axes[1, 0].grid(True, alpha=0.3, which='both')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
    
    # Plot 4: Derivative to understand shape
    dy = np.gradient(ydata, tdata)
    axes[1, 1].plot(tdata, dy, 'o-', linewidth=2, markersize=3, color='red')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('d(Response)/dt')
    axes[1, 1].set_title('Derivative (to see inflection points)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].spines['top'].set_visible(False)
    axes[1, 1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print diagnostics
    print(f"\n{'='*60}")
    print(f"TRF Data Diagnostics")
    print(f"{'='*60}")
    print(f"Number of points: {len(ydata)}")
    print(f"Time range: {tdata[0]:.6f} to {tdata[-1]:.6f} s")
    print(f"Response range: {np.min(ydata):.6f} to {np.max(ydata):.6f}")
    print(f"Mean response: {np.mean(ydata):.6f}")
    print(f"Std dev: {np.std(ydata):.6f}")
    
    # Find peak
    peak_idx = np.argmax(np.abs(ydata))
    print(f"\nPeak location: {tdata[peak_idx]:.6f} s")
    print(f"Peak value: {ydata[peak_idx]:.6f}")
    
    # Check if mostly positive or negative
    pos_frac = np.sum(ydata > 0) / len(ydata)
    print(f"Fraction positive: {pos_frac:.2%}")
    print(f"Fraction negative: {(1-pos_frac):.2%}")
    
    print(f"\n{'='*60}")
    print(f"Model Suitability Notes:")
    print(f"{'='*60}")
    if pos_frac < 0.2 or pos_frac > 0.8:
        print("✓ Looks like single polarity (good for biphasic)")
    else:
        print("✗ Mixed polarity - may be challenging for biphasic")
    
    # Check if biphasic makes sense
    if ydata[0] * ydata[-1] < 0:
        print("⚠ Data starts and ends with opposite signs")
    elif abs(dy[0]) > abs(dy[-1]):
        print("✓ Steeper at beginning than end (good for biphasic)")
    else:
        print("⚠ Steeper at end than beginning (unusual for biphasic)")


def estimate_initial_params(tdata, ydata):
    """
    Estimate reasonable initial parameters from data
    
    Args
    ----
    tdata : np.ndarray
        Time values
    ydata : np.ndarray
        Response values
    
    Returns
    -------
    p0 : list
        Initial parameter guesses [tau1, tau2, scale, c]
    bounds : tuple
        Lower and upper bounds for parameters
    """
    # Estimate scale from data range (always positive)
    scale = np.max(np.abs(ydata))
    
    # Estimate time constants from peak location and width
    # Find peak of absolute value
    abs_ydata = np.abs(ydata)
    peak_idx = np.argmax(abs_ydata)
    peak_time = tdata[peak_idx]
    
    # Avoid division by zero or very small times
    if peak_time < 0.0001:
        peak_time = tdata[-1] / 2
    
    # Estimate tau1 (rise time) as roughly 1/4 of peak time
    tau1_est = max(0.0001, peak_time / 4)
    
    # Estimate tau2 (decay time) - slower than tau1, roughly peak time
    tau2_est = max(tau1_est * 1.5, peak_time)
    
    # c parameter (relative weight) - usually ~1
    c_est = 1.0
    
    p0 = [tau1_est, tau2_est, scale, c_est]
    
    # Set bounds with strict inequalities
    # tau1: between very small and tau2
    tau1_lower = 0.00001
    tau1_upper = tau2_est * 0.9  # must be less than tau2
    
    # tau2: larger than tau1
    tau2_lower = tau1_est * 1.1
    tau2_upper = max(peak_time * 5, tau2_est * 3)
    
    # scale: positive range
    scale_lower = scale * 0.01
    scale_upper = scale * 100
    
    # c: can be positive or negative
    c_lower = -10
    c_upper = 10
    
    bounds = (
        [tau1_lower, tau2_lower, scale_lower, c_lower],
        [tau1_upper, tau2_upper, scale_upper, c_upper]
    )
    
    return p0, bounds


def fit_trf_biphasic(tdata, ydata, 
                      p0=None,
                      bounds=None,
                      verbose=False,
                      plot=False):
    """
    Fit a TRF curve to a biphasic function
    
    The biphasic function is: bp(t, tau1, tau2, scale, c)
    where:
        - tau1, tau2: time constants for the two phases
        - scale: scaling factor
        - c: relative weight of the two lobes
    
    Args
    ----
    tdata : np.ndarray
        Time values
    ydata : np.ndarray
        Response values (TRF curve)
    p0 : list, optional
        Initial parameter guesses [tau1, tau2, scale, c]
        If None, estimates automatically from data
    bounds : tuple, optional
        Lower and upper bounds for parameters
        If None, estimates automatically from data
    verbose : bool
        If True, print fitting results and diagnostics
    plot : bool
        If True, plot data and fit
    
    Returns
    -------
    fit_results : dict
        Dictionary containing:
        - 'popt': optimal parameters [tau1, tau2, scale, c]
        - 'pcov': covariance matrix
        - 'r2': R-squared value
        - 'residuals': residuals of fit
        - 'yfit': fitted values
    """
    
    # Auto-estimate parameters if not provided
    if p0 is None or bounds is None:
        p0_est, bounds_est = estimate_initial_params(tdata, ydata)
        if p0 is None:
            p0 = p0_est
        if bounds is None:
            bounds = bounds_est
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"TRF Fitting Diagnostics")
        print(f"{'='*60}")
        print(f"Data: {len(ydata)} points from {tdata[0]:.6f} to {tdata[-1]:.6f}")
        print(f"Data range: [{np.min(ydata):.6f}, {np.max(ydata):.6f}]")
        print(f"\nInitial parameter guess (p0): {p0}")
        print(f"Lower bounds: {bounds[0]}")
        print(f"Upper bounds: {bounds[1]}")
    
    # Fit biphasic function to data
    try:
        popt, pcov = curve_fit(bp, tdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    except RuntimeError as e:
        if verbose:
            print(f"\nWarning: Fitting did not fully converge: {e}")
        popt, _ = curve_fit(bp, tdata, ydata, p0=p0, bounds=bounds, maxfev=100000)
    
    # Calculate fitted values and R-squared
    yfit = bp(tdata, *popt)
    ss_res = np.sum((ydata - yfit) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    residuals = ydata - yfit
    
    fit_results = {
        'popt': popt,
        'pcov': pcov,
        'r2': r2,
        'residuals': residuals,
        'yfit': yfit
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Biphasic Function Fit Results:")
        print(f"{'='*60}")
        print(f"  tau1: {popt[0]:.6f}")
        print(f"  tau2: {popt[1]:.6f}")
        print(f"  scale: {popt[2]:.6f}")
        print(f"  c: {popt[3]:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.6f}")
        print(f"  Max absolute error: {np.max(np.abs(residuals)):.6f}")
        
        # Check if fit is poor
        if r2 < 0.5:
            print(f"\n⚠️  WARNING: Poor fit quality (R² = {r2:.3f})")
            print(f"   The biphasic model may not be appropriate for this data.")
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot fit
        axes[0].plot(tdata, ydata, 'o', label='Data', alpha=0.7, markersize=4)
        axes[0].plot(tdata, yfit, '-', linewidth=2, label='Biphasic Fit', color='red')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Response')
        axes[0].set_title(f'TRF Fit (R² = {r2:.4f})')
        axes[0].legend()
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Plot residuals
        axes[1].plot(tdata, residuals, 'o-', alpha=0.7, markersize=4, color='purple')
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Fit Residuals')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    return fit_results


def fit_trf_from_csv(csv_path, time_col='time_s', response_col='trf',
                     p0=None,
                     bounds=None,
                     verbose=True, plot=True, delimiter=','):
    """
    Read a TRF curve from CSV and fit to biphasic function
    
    Automatically estimates initial parameters and bounds from the data for robust fitting.
    
    Args
    ----
    csv_path : str
        Path to CSV file
    time_col : str or int
        Column name or index for time
    response_col : str or int
        Column name or index for response
    p0 : list, optional
        Initial parameter guesses. If None, estimates from data.
    bounds : tuple, optional
        Parameter bounds. If None, estimates from data.
    verbose : bool
        Print fitting results
    plot : bool
        Plot results
    delimiter : str
        CSV delimiter (default: ',')
    
    Returns
    -------
    fit_results : dict
        Fitting results (see fit_trf_biphasic)
    tdata : np.ndarray
        Time values
    ydata : np.ndarray
        Response values
    """
    
    # Read data from CSV
    tdata, ydata = read_trf_csv(csv_path, time_col=time_col, 
                                 response_col=response_col, delimiter=delimiter)
    
    # Fit biphasic function
    fit_results = fit_trf_biphasic(tdata, ydata, p0=p0, bounds=bounds,
                                    verbose=verbose, plot=plot)
    
    return fit_results, tdata, ydata


if __name__ == '__main__':
    # Example usage
    
    print("\n" + "="*60)
    print("TRF Fitting Tool - Getting Started")
    print("="*60)
    
    print("\n1. First, diagnose your data shape:")
    print("   from fit_trf import diagnose_trf_data")
    print("   diagnose_trf_data('your_data.csv')")
    
    print("\n2. Compare different models to find best fit:")
    print("   from fit_trf import compare_models")
    print("   results = compare_models('your_data.csv')")
    
    print("\n3. Fit the best model:")
    print("   from fit_trf import fit_trf_from_csv")
    print("   fit_results, tdata, ydata = fit_trf_from_csv('your_data.csv',")
    print("                                                  verbose=True,")
    print("                                                  plot=True)")
    
    print("\n" + "="*60)
    print("Available Models:")
    print("="*60)
    print("  • Biphasic    - bandpass/difference of exponentials")
    print("  • Exponential - simple decay (single time constant)")
    print("  • Rise-Decay  - alpha function (rise then decay)")
    print("  • Gaussian    - smooth impulse response")
    print("\n" + "="*60)
