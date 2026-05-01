import numpy as np
from scipy.optimize import curve_fit


def bp(t,tau1,tau2,scale,c=1):
    
    """Simple bandpass filter with 2 "lobes"
    the filter is scaled with c=1 such that the filter integrates to zero"""

    r = ((t/(tau1**2))*np.exp(-t/tau1) - c*(t/(tau2**2))*np.exp(-t/tau2))
    
    return scale*r
