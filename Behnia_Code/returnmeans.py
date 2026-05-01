#import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import pprint
import math
import time #counting for filter extraction
from scipy.optimize import curve_fit
from itertools import cycle
from scipy.signal import lfilter, decimate
from scipy.stats import binned_statistic


# Contains custom schema for ephys data
#from axolotl.schema.recordings_simple_ephys import RecordingInfo,RecordingData,FFFAlignmentParams,FFFData, WNFilterParams, WNData, DGData

# Contains custom plotting functions for data lists queried from datajoint schema
#from axolotl.plotting.ephysplot import plot_fff, plot_xt, plot_spatial, plot_temporal, plot_dg, plot_dg_peaks

from convolve import sigmoid, softplus # static nonlinear functions used here


"""

The following functions are meant to average and fit parameters to formatted
data extracted from the DataJoint database. The functions are:

    * simple gaussian

    * return_mean_temporal()

    * return_mean_spatial()

    * return_mean_nonlin()

    * return_mean_fff()

    * return_mean_dg_peaks()


    fitting functions
    -----------------

    * fit_nonlin()



TODO:
-----

fit_spatial()
fit_temporal()
"""

def simple_gaussian(x,mu,sig,scale):
    s = np.exp(-0.5*((x-mu)/sig)**2)
    s /= np.max(np.abs(s))
    return scale*s





""" THIS CODE IS MODIFEID FROM AXOLOTL"""
def return_mean_temporal(wn_dict,baseline=None,plot=False):

    """
    Returns mean trace of temporal filters in white noise list of dictionaries

    Args
    ----
    wn_dict: technically an Ordered List of dictionaries
    
    baseline: older version of code wasn't doing anything

    """
    dt = []
    for wn in wn_dict:
        dt.append(wn['sampling_rate']*wn['downsampling_factor'])

    dt = np.unique(dt)
    assert len(dt) == 1,'must be same sampling rate and downsampling factor'
    dt = dt[0]

    if plot:
        fig = plt.figure()
    
    av=[]
    for wn in wn_dict:

        trace = wn['temporal_filter'] # this returns the dot product of linear filter and x_wieghts
        scale = np.max(np.abs(wn['complete_filter']))
        # 499 is bc some cells are shorter
        # this will present a problem with downsampling differently =/ <<<<<<<<<<<<<<<<<
        trace = scale*trace[:499]/np.max(np.abs(trace))
        

        #dt = np.round(wn['filter_time']/len(trace),4) #<<< this might be a problem, be careful!


        t = np.linspace(0,len(trace)*dt,len(trace))

        if baseline == 'start':
            b = trace[0]
        elif baseline == 'end':
            b = np.mean(trace[-100:])
        else:
            b=0
            
        av.append(trace-b)
            
        if plot:
            plt.plot(trace-b,label=wn['recording_id']+ '-'+wn['subrecording_number'])

    mn = np.squeeze(np.mean(av,0))
    std = np.squeeze(np.std(av,0))

    n = len(wn_dict)


    if plot:
        plt.plot(mn,color='k',label='mean') 
        
    stat = {}
    stat['mean'] = mn
    stat['std'] = std
    stat['t'] = t
    stat['dt'] = dt
    stat['n'] = n
    
    if plot:
        plt.legend()
        plt.show()

    return stat








def return_mean_spatial(wn_list,plot_figure=False,**kwargs):

    """
    Return Spatial mean from an ordered dictionary of individual WN filters

    This function
    1. Select 1-D spatial receptive field as a slice of the 2-D space-time field
    2. Finds the center of the field by fitting a gaussian using scipy curve_fit
    3. Center to zero, and pad with nan so that an average can be taken around
        all centered RFs


    Args
    ----
    wn_list: Ordered List of dict objects, queried and fetched from datajoint
            where wn_list = (RecordingInfo * RecordingData * WNData).fetch(as_dict=True)

    Returns
    -------
    stat: dict
        'mean': np.nanmean of spatial receptive fields
        'std': np.nanstd of spatial receptive fields
        'x': 1-D array of x-values
        'dx': spatial sampling
        'spatial_list': list of 1-D arrays of spatial receptive fields

    """

    # legend is optional for when there are too many things to plot...
    if 'p0' in kwargs.keys():
        p0 = kwargs['p0']
    else:
        p0 = [30,4,1e-3] # mu, sig1,scale


    # used for finding units around max peak, [ind-dt:ind+dt,:]
    if 'dt' in kwargs.keys():
        dt = kwargs['dt']
    else:
        dt = 1

    # wrapper function to fit gaussian
    def fit_curve(x,p0):

        # preset to max location
        p0[0] = np.argmax(sp)*dx
        #print(p0)

        bounds=([-130,0,-20], [130,30,20])

        popt, pcov = curve_fit(simple_gaussian, x, sp,p0=p0,bounds=bounds)
        return popt


    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle(prop_cycle.by_key()['color'])

    # Plot each cell extracted from database
    if plot_figure:
        fig,ax = plt.subplots(1,1,figsize=(6,6))

    spatial_av = []

    # Loop through all the receptive field instances in the list
    for i,w in enumerate(wn_list):

        # this comes from DataJoint RecordingInfo()
        dx = float(w['wn_bar_width'])

        # find index of temporal peak point
        ind = np.argmin(w['temporal_filter'])

        # unusual case, except when downsampling
        if ind <= 2:
            dt = 1

        # keep colors the same for fits
        c = next(colors)

        x2= np.linspace(-10,10,100)*dx

        sp = -1*np.mean(w['complete_filter'][ind-dt:ind+dt,:],0)
        std = -1*np.std(np.real(w['complete_filter'])[ind-dt:ind+dt,:],0)

        x = np.arange(len(sp))*dx # since they are all 2.5 or 5 degrees

        # Fit curve, as defined above
        popt=fit_curve(x,p0)

        x1 = np.arange(len(sp))*dx # -popt[0]
        closest_index = (np.abs(x-popt[0])).argmin()
        x1 = x1 - x1[closest_index]

        #print('x1',x1)
        #print('fit center',popt[0])
        #print('closest index:',x1[closest_index])
        #print('x1_shifted',x1)


        # PAD WITH NAN FOR SPATIAL MEAN
        left_pad = int((120-np.abs(np.min(x1)))/dx)
        right_pad = int((120-np.abs(np.max(x1)))/dx)
        sp_pad = np.pad(sp,(left_pad,right_pad),mode='constant',constant_values=(np.nan,np.nan))
        #print(sp_pad)

        spatial_av.append(sp_pad)


        # Plot Gaussian Fit
        if plot_figure:
            ax.set_title('Gaussian Fit')

            ax.plot(x1,sp,color=c,alpha=0.5,linewidth=3)
            ax.fill_between(x1,sp - std, sp + std,color=c,alpha=0.2)

            x2= np.linspace(-10,10,100)*dx
            ax.plot(x2,simple_gaussian(x=x2,mu=0,sigma=popt[1],scale=popt[2]),'--',color=c,label=r'$\sigma=$'+str(np.round(popt[1],2)))

            ax.legend()
            plt.tight_layout()


    if plot_figure:
        ax.set_xlabel('space (\u00b0)')
        plt.show()

    stat = {}
    mn = np.nanmean(spatial_av,0)
    stat['mean'] = mn
    stat['std'] = np.nanstd(spatial_av,0)
    stat['x'] = np.linspace(-len(mn)/2,len(mn)/2,len(mn))*dx
    stat['dx'] = dx
    stat['spatial_list'] = spatial_av # store for reference

    return stat




def return_mean_nonlin(wn_list,bin_size=200,verbose=False,**kwargs):

    """
    This extracts the binned statistic used in nonlinearity calculations

    ideally this would be saved for each analysis and not "recalculated" each time

    1. Cut white noise traces as specified in database
    2. Find upper and lower bounds to standardize bin range
    3. Bin all data with same bin size and range

    Args
    ----
    wn_list: Ordered List of dict objects, queried and fetched from datajoint
        e.g. wn_list = (RecordingInfo * RecordingData * WNData).fetch(as_dict=True)

    bin_size: single int, used for scipy.stats.binned_statistic. It is Important
        to use the same bins across samples!


    Returns
    ----
    binned_dict: dict object
        'y_list': list of binned y values (response voltage)
        'y_pred_list': list of binned y linear prediction values
        'y_mean': mean of binned y values
        'y_std': std of binned y values
        'y_pred_mean': mean of binned linear prediction values
        'bin_size': single value
        'range': tuple of min and max values of all samples combined

    """

    minmax=[]
    cut_traces = []
    for w in wn_list:
        if verbose:
            print('\n')
            print(w['cell_type']+'-'+w['recording_id'])
            print('-------')
            print('start time: ',int(w['wn_start_time']))
            print('end time: ',int(w['wn_end_time']))
            print('duration: ',str(int(w['wn_end_time'])-int(w['wn_start_time'])),' seconds')
            print('# bins: ',w['bins'])

        # 1. Cut white noise traces
        bin_size = int(w['bins'])

        timestamps = w['timestamps'][::int(w['downsampling_factor'])]
        traces = decimate(w['ephys_trace'], int(w['downsampling_factor']))
        tbool = (timestamps >= int(w['wn_start_time'])) & (timestamps < int(w['wn_end_time']))
        cut_trace = traces[tbool] -w['cut_trace_mean']

        #assert(len(cut_traces)==len(w['linear_prediction']),'correct size')
        if len(cut_trace)!=len(w['linear_prediction']):
            cut_trace = cut_trace[-len(w['linear_prediction']):]
            if verbose: print('len(cut_traces)!=len(w[linear_prediction])')

        # 2. Find upper and lower bounds
        minmax.append(np.min(cut_trace))
        minmax.append(np.max(cut_trace))
        cut_traces.append(cut_trace)


    y_list = []
    y_pred_list=[]
    for w,trace in zip(wn_list,cut_traces):

        # 3. Bin all data with same bin size and range

        y, y_pred, _ = binned_statistic(w['linear_prediction'], trace,bins=bin_size,range=(np.min(minmax),np.max(minmax)))

        # interp?
        # mask = np.isnan(y)
        # y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        y_pred = np.mean([y_pred[:-1], y_pred[1:]], axis=0)

        y_list.append(y)
        y_pred_list.append(y_pred)


    y_mean = np.nanmean(y_list,0)
    y_std = np.nanstd(y_list,0)
    y_pred_mean = np.nanmean(y_pred_list,0)

    binned_dict = {}
    binned_dict['y_list'] = y_list
    binned_dict['y_pred_list'] = y_pred_list
    binned_dict['y_mean'] = y_mean
    binned_dict['y_std'] = y_std
    binned_dict['y_pred_mean'] = y_pred_mean
    binned_dict['bin_size'] = bin_size
    binned_dict['range'] = (np.min(minmax),np.max(minmax))

    return binned_dict

# Outside of the loop, use this code to plot the binned_dict output
#     fig,ax = plt.subplots(1,1,figsize=(6,6))

#     for y, y_pred in zip(binned_dict['y_list'],binned_dict['y_pred_list']):
#         plt.plot(y_pred,y)

#     y_mean = np.mean(binned_dict['y_list'],0)

#     y_pred_mean = np.mean(binned_dict['y_pred_list'],0)

#     plt.plot(y_pred_mean,y_mean,'k',linewidth=3)
#     plt.plot(0.004*np.array([-1,1]),0.004*np.array([-1,1]),'gray')
#     plt.grid(which='both')




def return_mean_fff(fff_list,category,duration,baseline='end',av_within_cell=True,legend=True):

    """
    Function to extract mean full field flash data


    Some cells were presented with full field flashes multiple times, and therefore
    a single 'recording_id' might have multiple subrecording_numbers. This gives us
    the choice of averaging within a cell followed by averaging across cells, or
    averaging across all individual recordings.

    Args
    ----
    fff_list: list, list of ordered dicts queried with datajoint
    category: string, 'on_repeats' or 'off_repeats'
    duration: float, in seconds (e.g. 1)
    baseline: string, 'start' or 'end'
    av_within_cell=True: boolean
        Average within cell first, then average across cells in fff_list
    legend=True: boolean


    Returns
    -------
    fig: figure handle

    (I cleaned up this function on January 7, 2020
    modified from plot_fff Jan 28, 2020)
    """


    traces = [] # the mean of this is taken at the end and plotted in ax[2]

    # >>> AVERAGE WITHIN CELL <<<
    # check for non_unique recording_id values and average
    if av_within_cell:
        print('>> Average within cell')

        r_id = [r['recording_id'] for r in fff_list] # all recording ids

        for r in np.unique(r_id):

            # find index of unique/repeated recording ids
            idx = np.where(np.asarray(r_id)==r)[0]

            #print(idx.tolist())
            #print(np.asarray(r_id)[idx])

            av_within = []
            sub_rec_list = [] # keeps track of subrecording id, for book keeping
            for i in idx: # this works for list of 1 as well as larger
                f = fff_list[i]

                dt = f['sampling_rate'] # presumably these are the same, otherwise throw error

                trace = np.mean(f[category],1)
                trace_cut = trace[:int(int(duration)/dt)] # necessary for occasional mismatch lengths
                av_within.append(trace_cut)

                sub_rec_list.append(f['subrecording_number'].zfill(3))

            trace = np.mean(av_within,0) # this treats av_within as a single trace
            trace_std = np.std(av_within,0) # this should be zero if list idx is size 1

            t = np.linspace(0,len(trace)*dt,len(trace))

            if baseline == 'start':
                b = trace[0]
            else:
                b = np.mean(trace[-100:])

            traces.append(trace-b)

    # >>> DON"T AVERAGE WITHIN CELL <<<
    if av_within_cell==False:
        print('>> No average within cell')

        for f in fff_list:

            dt = f['sampling_rate']

            trace = np.mean(f[category],1) # this treats av_within as a single trace
            trace = trace[:int(int(duration)/dt)] # necessary for occasional mismatch lengths

            t = np.linspace(0,len(trace)*dt,len(trace))

            if baseline == 'start':
                b = trace[0]
            else:
                b = np.mean(trace[-100:])

#             ax[1].plot(t,trace-b,label=f['recording_id']+'-'+f['subrecording_number'].zfill(3))
#             ax[1].set_title('Zero Baseline')
#             ax[2].plot(t,trace-b,alpha=0.25,label=f['recording_id'])

            traces.append(trace-b)

    mean = np.mean(traces,0)
    std = np.std(traces,0)


    mean_fff = {}
    mean_fff['mean'] = mean
    mean_fff['std'] = std
    mean_fff['dt'] = dt
    mean_fff['t'] = t
    mean_fff['n'] = len(traces)
    mean_fff['category'] = category
    mean_fff['duration'] = duration
    mean_fff['av_within_cell'] = av_within_cell

    return mean_fff









""" THIS CODE IS MODIFEID FROM AXOLOTL"""
def return_mean_dg_peaks(dg_list,normalize=False):

    """

    Args
    ----
    dg_list: Ordered List of dict objects, queried and fetched from datajoint
            where dg_list = (RecordingInfo * RecordingData * DGData).fetch(as_dict=True)
            
    normalize: boolean, whether to normalize across cells


    Returns
    ----
    dg_dict:
        'dg_mean': mean for each peak
        'dg_std': std for each peak
        'dg_peak_list': list of peaks for each cell from dg_list
        'freq_array': 1-D array of frequency values
        'name': string, either 'Spatial' or 'Temporal'
    """

    av_peaks = []
    for dg in dg_list:

        spatialfrequency_array = dg['spatialfrequency_array']
        temporalfrequency_array = dg['temporalfrequency_array']
        sampling_rate = dg['sampling_rate']

        # make sure this is correct
        psths = dg['psths']
        psths_std = dg['psths_std']
        repeat_dur = dg['repeat_duration']

        # Check if changing spatial or temporal frequency
        if np.median(spatialfrequency_array) != spatialfrequency_array[0]:
            freq_array = spatialfrequency_array
            name = 'Spatial'

        if np.mean(temporalfrequency_array) != temporalfrequency_array[0]:
            freq_array = temporalfrequency_array
            name = 'Temporal'

        peaks = []


        for (p,std,freq,rep) in zip(psths,psths_std,freq_array,repeat_dur):

            #c = next(stim_colors)

            dif = np.max(p[:rep]) - np.min(p[:rep])
            dif_std = np.sqrt(std[np.argmax(p[:rep])]**2 + std[np.argmin(p[:rep])]**2) # is this correct?
            #plt.errorbar(1/freq,dif,yerr=dif_std,marker='d',color=c)

            peaks.append(dif)


        #lab = dg['cell_type']+' '+dg['recording_id']+'-'+dg['subrecording_number'].zfill(3)+'  '+dg['bath_solution'] #+' '+dg['stimulus_name']
        #plt.plot(1/freq_array,peaks,'--',alpha = 1,label=lab,color=next(av_colors))
        #plt.show() # plot after returning handle


        if normalize:
            av_peaks.append(peaks/np.max(peaks))
        else:
            av_peaks.append(peaks)

    # plot average
    y = np.squeeze(np.mean(av_peaks,0))
    y_std = np.std(av_peaks,0)

    dg_dict = {}
    dg_dict['dg_mean'] = y
    dg_dict['dg_std'] = y_std
    dg_dict['dg_peak_list'] = av_peaks
    dg_dict['freq_array'] = freq_array
    dg_dict['name'] = name

    return dg_dict





""" FITTING FUNCTIONS """









def fit_nonlin(wn_list,nonlin='softplus',sigma_weight='std',verbose=False,**kwargs):

    """
    Return mean of parameterized nonlinearities

    1. Calculate mean nonlinearity
    2. Calculate nanmean

    Args
    ----
    nonlin: string either 'softplus' or 'sigmoid'
    sigma_weight: array, None or 'sts'. A 1-d sigma should contain values of standard deviations of errors in ydata


    Returns
    ----
    fn_dict:
        'y_mean': mean binned data calculated from return_mean_nonlin
        'y_pred_mean': mean binned linear prediction from return_mean_nonlin
        'popt': parameters found by curvefit. This is the most important output
        'nonlin': string either 'softplus' or 'sigmoid'
        'sigma_weight': either None or 'std' - if std, it will take std of bins into account
        'y_fit_nonan': final bins used for fitting, with nonans
        'y_pred_fit_nonan': final bins used for fitting, with nonans

    # note: could also save function ...


    """

    """Cut the traces and reevaluate the bins (it would be better if it was saved...)"""

    if verbose: print('sigma nonlinear weighting',sigma_weight)

    # 1. Calculate mean nonlinearity
    binned_dict = return_mean_nonlin(wn_list)

    y_mean = binned_dict['y_mean']
    y_std = binned_dict['y_std']
    y_pred_mean = binned_dict['y_pred_mean']


    """Fit parameters to mean"""

    # # 2. Calculate nanmean (often there are nans in y_mean)
    # y_mean_mask = ~np.isnan(y_mean)
    # if verbose: print(y_mean_mask) # these should be continuous
    #
    # y_fit = y_mean[y_mean_mask]
    # y_std_fit = y_std[y_mean_mask]
    # y_pred_fit = y_pred_mean[y_mean_mask]
    #
    # # this is messy - need to figure out how to do this
    # # A 1-d sigma should contain values of standard deviations of errors in ydata
    # # In this case, the optimized function is chisq = sum((r / sigma) ** 2).
    #
    # if sigma_weight == 'std':
    #     sigma_weight = y_std_fit
    # else:
    #     sigma_weight = np.ones(len(y_fit))

    # 2. Calculate nanmean (often there are nans in y_mean)
    mask1 = ~np.isnan(y_mean) # some values are nan
    tmp_std = y_std[mask1]
    mask2 = np.isfinite(1/tmp_std) # some std values are 0, cannot be taken into account with std weighting

    # if there is only one sample, mask2 should all be true
    # this is an unusual case
    #if np.unique(tmp_std)[0] == 0.:
    if (np.unique(tmp_std) == 0.).all():
        mask2=len(tmp_std)*[True]

#     y_fit = y_mean[mask1]
#     y_std_fit = y_std[mask1]
#     y_pred_fit = y_pred_mean[mask1]

    y_fit = y_mean[mask1][mask2]
    y_std_fit = y_std[mask1][mask2]
    y_pred_fit = y_pred_mean[mask1][mask2]

    # A 1-d sigma should contain values of standard deviations of errors in ydata
    # In this case, the optimized function is chisq = sum((r / sigma) ** 2).

    if sigma_weight == 'std':
        sigma_weight = y_std_fit
    else:
        sigma_weight = np.ones(len(y_std_fit))

    if nonlin == 'softplus':
        if verbose: print('>> evaluate softplus')
        p0=[1.1,0.9,1.1,-0.001,1] # a,b,c,d,k ... k can only equal 1-1.2 for now
        if verbose:
            print('softplus function soft(x) = c*np.log(1+np.exp(a*x+b))**k + d')
            print('softplus parameters: a,b,c,d,k')
            print('>> p0 = ',p0)
        bounds = ([0,-100,-150,-1,1],[5000,50,150,1,1.2])
        if verbose: print('>> bounds = ',bounds)
        popt, pcov = curve_fit(softplus, y_pred_fit,y_fit, p0, sigma=sigma_weight, bounds=bounds,maxfev=10000)

#         fig,ax = plt.subplots(1,1)
#         plt.plot(x,softplus(x,*popt))
#         plt.plot(x,nonlin_mean)
#         plt.show()

    if nonlin == 'sigmoid':
        if verbose: print('>> evaluate sigmoid')
        p0=[1.5,10,1.1,-5] # k,L,x0,b
        popt, pcov = curve_fit(sigmoid, y_pred_fit,y_fit, p0,sigma=sigma_weight, maxfev=1000000)

    fn_dict = {}
    fn_dict['y_mean'] = y_mean
    fn_dict['y_pred_mean'] = y_pred_mean
    fn_dict['popt'] = popt
    fn_dict['nonlin'] = nonlin
    fn_dict['sigma_weight'] = sigma_weight
    fn_dict['y_fit_nonan'] = y_fit
    fn_dict['y_pred_fit_nonan'] = y_pred_fit

    if verbose: print('popt',popt)

    return fn_dict






    
    
    
    
    
