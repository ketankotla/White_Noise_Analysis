import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt


def bp(t,tau1,tau2,scale,c=1):
    
    """Simple bandpass filter with 2 "lobes"
    the filter is scaled with c=1 such that the filter integrates to zero"""

    r = ((t/(tau1**2))*np.exp(-t/tau1) - c*(t/(tau2**2))*np.exp(-t/tau2))
    
    return scale*r


def fit_bp(t, y, p0=None, bounds=([-np.inf, 1e-12, 1e-12, 0], [np.inf, np.inf, np.inf, np.inf])):
    """Fit bp(t, tau1, tau2, scale, c=1) to data points."""

    t = np.asarray(t)
    y = np.asarray(y)

    if p0 is None:
        scale0 = np.max(np.abs(y)) if y.size else 1.0
        p0 = (1.0, 2.0, scale0, 1.0)

    popt, pcov = curve_fit(bp, t, y, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov


def read_csv_and_fit(csv_file, p0=None, bounds=([-np.inf, 1e-12, 1e-12, 0], [np.inf, np.inf, np.inf, np.inf])):
    """
    Read a CSV file and fit a biphasic function to the data.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    p0 : tuple, optional
        Initial parameter guess
    bounds : tuple, optional
        Bounds for parameters
    
    Returns:
    --------
    popt : ndarray
        Optimal parameters [tau1, tau2, scale, c]
    pcov : ndarray
        Covariance matrix
    t : ndarray
        Time values
    y : ndarray
        Response values
    """
    # Read CSV file (first column is time, second column is response)
    df = pd.read_csv(csv_file)
    t = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    # Fit the biphasic function
    popt, pcov = fit_bp(t, y, p0=p0, bounds=bounds)
    
    return popt, pcov, t, y


def plot_fit(t, y, popt, title="Biphasic Fit", figsize=(10, 6)):
    """
    Plot the data and fitted biphasic curve.
    
    Parameters:
    -----------
    t : ndarray
        Time values
    y : ndarray
        Response values
    popt : ndarray
        Fitted parameters [tau1, tau2, scale, c]
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    # Generate smooth curve for plotting
    t_smooth = np.linspace(t.min(), t.max(), 500)
    y_fitted = bp(t_smooth, *popt)
    
    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(t, y, 'o', label='Data', markersize=6, alpha=0.7)
    plt.plot(t_smooth, y_fitted, '-', label='Fitted Curve', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    popt, pcov, t, y = read_csv_and_fit("../all_Mi4/trfs/Mi4_all-samples-average-trf.csv")
    print(f"Fitted parameters: tau1={popt[0]:.4f}, tau2={popt[1]:.4f}, scale={popt[2]:.4f}, c={popt[3]:.4f}")
    plot_fit(t, y, popt)


