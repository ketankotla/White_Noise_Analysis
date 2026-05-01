""" Various Functions for taking average and plotting Flash Data 

created: October 1, 2020

updated: January 21, 2021

# UPDATE IN AXOLOTL!!! >>> FONT SIZES, LINEWIDTH ETC


The following functions were built specifically for short impulse flashes. Jessie began collecting data for 20, 40, 80 and 160 ms flashes in July 2020. She also collected some data at low contrast.

These functions were incorporated into axolotl.traces on January 21, 2021

For better or for worse, these functions are very specific to the stimulus construction and likely do not generalize well. We are not too worried about this, as we are mostly done with our data collection and don't plan on incorporating novel stimuli.

Functions
---------

Analysis:

* extract_flash_data() - extracts and averages flash responses

* flash_wn_pred() - predicts flash response from white noise filters


Plotting:

* plot_flash_data_saline_OA() - plots average flash responses of saline and OA simultaneously

* plot_flash_and_pred() - plots flash responses with white noise predicted responses

* plot_flash_contrast_data() - plots flash responses at both low and high contrast







"""


import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pprint
import time #counting for filter extraction
from itertools import cycle

from scipy.optimize import curve_fit
from scipy.signal import lfilter, decimate
from scipy.stats import binned_statistic
from scipy import signal

from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid

# for pdf generation
from matplotlib.backends.backend_pdf import PdfPages

# Contains custom plotting functions for data lists queried from datajoint schema

from convolve import sigmoid, softplus # static nonlinear functions used here
from returnmeans import simple_gaussian, return_mean_temporal,return_mean_fff, return_mean_spatial, return_mean_nonlin, fit_nonlin, return_mean_dg_peaks
#from axolotl.utils import events_binary,argclosest


def extract_flash_data(data,
                       stim=["FFF_uniform_20msimpulse_10int","FFF_uniform_40msimpulse_10int","FFF_uniform_80msimpulse_10int","FFF_uniform_160msimpulse_10int"],
                       detrend=False,
                       window=10,
                       baseline='start',
                       baseline_window = 5,
                       plot=False,
                       verbose=False,
                       skip_sweep_dict = {},
                       figsize=(4,4),
                       xlim=(-0.02,3),
                       ylim=(-0.02,0.03),
                       **kwargs):
    
    """ This function takes in a datajoint query object, extracts flashes and averages them 
    
    Args
    ----
    data: datajoint query object
    stim: list of strings for datajoint query e.g. stim = ["FFF_uniform_20msimpulse_10int"]
    detrend: Boolean, if True, apply scipy.signal.detrend(,type='linear') to entire trace
    window: int, number of seconds to cut from each sweep before averaging. Max is 10 for this stimulus paradigm
    baseline: str, 'start' or 'end', if start, average all sweeps from initial value. 
            If 'end', average sweeps after subtracting mean of last n (e.g. 5) seconds for each sweep
    baseline_window: int, last n (e.g. 5) seconds to be averaged for each sweep to then be subtracted
    plot: Boolean, if True plots individual traces used for average. This is useful for sanity check
    verbose: Boolean
    skip_sweep_dict: dict, keys are recording_id and subrecording number, values are "sweep" to skip.
        e.g. skip_sweep_dict={'200928-18':0,'200928-20':3}
    
    Returns
    -------
    flash_dict: Dictionary with keys 'mean' 'std' 'n' and 'sampling_rate' for each stimulus condition specified by input string list 'stim'
    
    A NOTE ABOUT DETREND: I am doing a linear detrend on the entire signal, however I don't think this has a strong
    effect, and can actually cause distortion when traces have large (and deviant) activity when the projector turns on
    before the stimulus starts
    
    """
    
    flash_dict = {}
    
    

    for s in stim:
        
        test = data & 'stimulus_name="'+s+'"'
        
        
        fff_data = (test).fetch(as_dict=True)

        if plot:
            fig,ax = plt.subplots(1,1,figsize=figsize)

        mn_all = []
        n = 1
        
        if len(fff_data) == 0: # empty list
            print('skip stim ' + s + ' (empyt list)')
            continue
            
        ff = fff_data[0]
        old_name = ff['recording_id']
        if verbose: print(old_name)
            
        
        for ff in fff_data:
            
            """ linear detrend of entire/full trace """
            if detrend:
                ephys_trace = signal.detrend(ff['ephys_trace'],type='linear')
            else:
                ephys_trace = ff['ephys_trace']

            new_name = ff['recording_id']
            if verbose: print('>>'+new_name+'-'+ff['subrecording_number']+ ' '+ff['bath_solution'])

            """ keep track of n cells """
            if old_name != new_name:
                n+=1
                old_name = new_name

            if verbose: print('n=',n)

            """ process photodiode for pulling out traces """
            photodiode = ff['photodiode_trace']
            sos = signal.butter(10, 15, 'lp', fs=1000, output='sos')
            filtered_photodiode = signal.sosfilt(sos, photodiode)
            filtered_photodiode /=np.max(filtered_photodiode)

            start_times = events_binary(filtered_photodiode,ff['timestamps'],direction='rising',sigma=10)
            pd_start_times = argclosest(ff['timestamps'],start_times)

            """ plot and save traces"""
            traces = []
            
            if verbose: print('>> '+ff['recording_id']+'-'+ff['subrecording_number'])

            
            for ii,t in enumerate(pd_start_times):
                
                """ unusual case of skipping one or more sweeps manually"""
                skip_key = ff['recording_id']+'-'+ff['subrecording_number']
                if (skip_key in skip_sweep_dict.keys()) and (skip_sweep_dict[skip_key] == ii):
                    if verbose: print('>> skip sweep {} for '.format(ii)+ff['recording_id']+'-'+ff['subrecording_number'])
                    continue
                    
                tmp = ephys_trace[t:t+int(window/0.0002)] # sampling rate hardcoded

                #plt.plot(trace - trace[0])
                
                """ baseline set at beginning or by average of end """
                if baseline == 'start':
                    traces.append(tmp-tmp[0])
                if baseline == 'end':
                    traces.append(tmp-np.mean(tmp[-int(baseline_window/0.0002):])) # baseline average of last 5 seconds
            mn = np.mean(traces,axis=0)
            mn_all.append(mn)

            time = np.arange(0,len(mn))*0.0002

            if plot:
                plt.plot(time,mn,linewidth=2,label=ff['cell_type']+'-'+ff['recording_id']+'-'+ff['subrecording_number']+ ' '+ff['bath_solution'])
                std = np.std(traces,axis=0)
                plt.fill_between(time,mn+std,mn-std,alpha=0.2)

        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']] = {}
        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['mean'] = np.mean(mn_all,axis=0)
        # should combine std of both here >>>>>>>
        #Tm4_flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['mean'] = np.mean(mn_all,axis=0)
        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['std'] = np.std(mn_all,axis=0)
        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['n'] = n

        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['sampling_rate'] = ff['sampling_rate']
        
        flash_dict[ff['stimulus_name']+' '+ff['bath_solution']]['all'] = mn_all

        if plot:
            
            if 'title' in kwargs.keys():
                title=title
            else:
                title=s[-9:]
                    
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.plot(time,np.mean(mn_all,axis=0),'k',linewidth=2)
            plt.title(title)
            plt.xlabel('time (s)')
            plt.legend(bbox_to_anchor=(1.1,1.1))
            plt.ylim(ylim)
            plt.xlim(xlim)

    return flash_dict





""" Make white noise prediction of falsh data """


def flash_wn_pred(wn_dict,flash_dict=None,plot=False,c = 'saddlebrown',**kwargs):
    
    """ 
    
    Make flash predictions with white noise filters
    
    This uses the complete linear-nonlinear filter for each cell individually and predicts responses to 20,40,80,160 ms
    by convolution in time, summation in space, and finally a static nonlinearity
    
    
    Args
    ----
    wn_dict: Dictionary, extracted from datajoint query. Example keys are 'recording_id','complete_filter'
    flash_dict: 
    plot: Boolean, if True plots wn prediction with and flash data
    c: color string (e.g. 'midnightblue' for Tm1)
    
    
    Returns
    -------
    wn_pred_dict: Dictionary, contains keys for 4 conditions: 20,40,80,160. Value for each key is a list of 1D temporal predictions
    """
    
    if (plot and flash_dict == None):
        raise Exception('For plotting, must specify flash_dict')
    
    wn_pred_dict = {}
    wn_pred_dict['20'] = []
    wn_pred_dict['40'] = []
    wn_pred_dict['80'] = []
    wn_pred_dict['160'] = []


    # loop throug WN filters
    for wn in wn_dict:

        if plot: fig,ax = plt.subplots(1,4,figsize=(12,3))

        """Plot WN filter prediction (without nonlinearity)"""

        r_id = wn['recording_id']
        #print(r_id) 

        #filt = wn['temporal_filter'].squeeze()
        filt = np.real(wn['complete_filter'])

        #pred10s_off = np.convolve(filt,-1*np.ones(10*100))
        t_pred = np.arange(0,3,0.01)
        pred20 = np.zeros((len(t_pred),filt.shape[1]))
        pred40 = np.zeros((len(t_pred),filt.shape[1]))
        pred80 = np.zeros((len(t_pred),filt.shape[1]))
        pred160 = np.zeros((len(t_pred),filt.shape[1]))

        for i in range(filt.shape[1]):
            pred20[:,i] = np.convolve(filt[:,i],-1*np.ones(2))[:len(t_pred)]
            pred40[:,i] = np.convolve(filt[:,i],-1*np.ones(4))[:len(t_pred)]
            pred80[:,i] = np.convolve(filt[:,i],-1*np.ones(8))[:len(t_pred)]
            pred160[:,i] = np.convolve(filt[:,i],-1*np.ones(16))[:len(t_pred)]

        if plot:
            ax[0].plot(t_pred,np.sum(pred20,axis=1),'k-',linewidth=2,label='20 wn filter linear pred')
            ax[1].plot(t_pred,np.sum(pred40,axis=1),'k-',linewidth=2,label='40 wn filter linear pred')
            ax[2].plot(t_pred,np.sum(pred80,axis=1),'k-',linewidth=2,label='80 wn filter linear pred')
            ax[3].plot(t_pred,np.sum(pred160,axis=1),'k-',linewidth=2,label='160 wn filter linear pred')




        # apply nonlin
        fit = fit_nonlin([wn],nonlin='softplus',sigma_weight = None,verbose=False)

        a = fit['popt'][0]
        b = fit['popt'][1]
        cc = fit['popt'][2]
        d = fit['popt'][3]
        k = fit['popt'][4]
        #plt.plot(x,sigmoid(x,k,L,x0,b),'o',linewidth=4)
        #plt.plot(x,sigmoid(x,k,L,x0,b),'-',color=c,linewidth=3,alpha=1)

        if plot:
            ax[0].plot(t_pred,softplus(np.sum(pred20,axis=1),a,b,cc,d,k),'r--',linewidth=2,label='wn filter nonlinear pred')
            ax[1].plot(t_pred,softplus(np.sum(pred40,axis=1),a,b,cc,d,k),'r--',linewidth=2)
            ax[2].plot(t_pred,softplus(np.sum(pred80,axis=1),a,b,cc,d,k),'r--',linewidth=2)
            ax[3].plot(t_pred,softplus(np.sum(pred160,axis=1),a,b,cc,d,k),'r--',linewidth=2)


        wn_pred_dict['20'].append(softplus(np.sum(pred20,axis=1),a,b,cc,d,k))
        wn_pred_dict['40'].append(softplus(np.sum(pred40,axis=1),a,b,cc,d,k))
        wn_pred_dict['80'].append(softplus(np.sum(pred80,axis=1),a,b,cc,d,k))
        
        
        # SPECIFIC TO TM9 - BE CAREFUL HERE
        
        """ something is wrong with 190812 nonlinearity, keep it linear..."""
        if r_id == '190812':
            wn_pred_dict['160'].append(np.sum(pred160,axis=1))
        else:
            wn_pred_dict['160'].append(softplus(np.sum(pred160,axis=1),a,b,cc,d,k))
            
            
        

        """ Plot flash Data """
        for k,tr in flash_dict.items():
            if 'saline' in k:

                time = np.arange(0,len(tr['mean']))*tr['sampling_rate'] # should be 0.0002

                #c = next(colors)
                if plot:
                    if '20' in k:
                        ax[0].plot(time,tr['mean'],label=k,color=c,linewidth=3)
                        ax[0].fill_between(time,tr['mean']+tr['std'],tr['mean']-tr['std'],color=c,alpha=0.2)
                        ax[0].set_title('20 ms')
                    if '40' in k:
                        ax[1].plot(time,tr['mean'],label=k,color=c,linewidth=3)
                        ax[1].fill_between(time,tr['mean']+tr['std'],tr['mean']-tr['std'],color=c,alpha=0.2)
                        ax[1].set_title('40 ms')
                    if '80' in k:
                        ax[2].plot(time,tr['mean'],label=k,color=c,linewidth=3)
                        ax[2].fill_between(time,tr['mean']+tr['std'],tr['mean']-tr['std'],color=c,alpha=0.2)
                        ax[2].set_title('80 ms')
                    if '160' in k:
                        ax[3].plot(time,tr['mean'],label=k,color=c,linewidth=3)
                        ax[3].fill_between(time,tr['mean']+tr['std'],tr['mean']-tr['std'],color=c,alpha=0.2)
                        ax[3].set_title('160 ms')




        if plot:
            for a in ax:
                a.set_ylim(-0.03,0.03)
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
                a.legend(frameon=False)
                a.get_legend().remove()

            plt.tight_layout()
            plt.show()

    
    return wn_pred_dict










###################################

"""Custom Funtions"""

def bp(t,tau1,tau2,scale,c=1):
    
    """Simple bandpass filter with 2 "lobes"
    the filter is scaled with c=1 such that the filter integrates to zero"""

    r = ((t/(tau1**2))*np.exp(-t/tau1) - c*(t/(tau2**2))*np.exp(-t/tau2))
    
    return scale*r

 ## CREATE SINE WAVE
def sine_wave1d(t,t_period,phase):

    stimulus = np.sin(2*np.pi*(1/t_period)* t + phase)

    return stimulus

def get_sine_tuning(filt,freqs,dt=0.01,num_cycles=4,max_start = 5,verbose=False):
    
    """
    for given list of frequencies, convolve filter with sine wave
    
    filt: vector of filter
    freqs: list of frequencies in Hz
    dt: timestep in seconds
    num_cycles: convolution 
    max_start:
    
    
    n.b. max amplitude should be taken for middle segments/portions of steady
    state responses, as there can be strange effects at the beginning and end of
    a convolution
    """
    
    # max_end = 15 # removed this
    
    tt = np.arange(0,num_cycles*1/np.min(freqs),dt)
    
    max_end = 0.9*num_cycles*1/np.min(freqs)
    
    if verbose: print('{} seconds, sampling {} Hz '.format(num_cycles*1/np.min(freqs),1/dt))
    
    if verbose: print('max calculation from {}:{} seconds'.format(max_start,max_end))
    
    
    resp = {}
    resp['stim_time'] = tt
    resp['response'] = []
    resp['max'] = []
    resp['freqs'] = freqs
    for freq in freqs:
        
        stimulus = sine_wave1d(tt,t_period=1/freq,phase=0)
        
        response = np.convolve(filt,stimulus)
        
        resp['response'].append(response)
        resp['max'].append(np.max(response[int(max_start/dt):int(max_end/dt)])) # take max from 5 seconds onwards, int(max_end/dt)
    
    return resp


""" Taken from Explore-Biphasic-Measure-2020-10-21.ipynb"""

def parameterize_frequency_tuning(trace_dict,
                                  p0,
                                  freq,
                                  dt,
                                  dF,
                                  dt_flash=None,
                                  mode='white_noise',
                                  end=2, # in seconds
                                  bounds=([0,0,-1,-3], [1, 3., 0.01,3]),
                                  bound_freq=0.1,
                                  verbose=False):
    
    """
    
    trace_dict should contain traces of cells for a single cell type
    
    I would like this to work for both white noise data and for flash data...
    
    Returns: dict of individual and mean parameterizations
    """
    
    parameterized_dict = {}
    
    numerical_freq = []
    area_under_curve = []
    max_freq_list = []
    popt_list = []
    temporal_filter_list = []
    fwhm_list = []
    r2_list = []

    
    for i,wn in enumerate(trace_dict):
        
        tdata = np.arange(0,end,dt)
        if mode == 'white_noise':
            ydata = np.squeeze(wn['temporal_filter'])[:int(end/dt)]
            tdata1 = np.arange(0,end,dt)
            
        if mode == 'pred_dict':
            ydata = wn[:int(end/dt)]
            tdata1 = np.arange(0,end,dt)
            
        if mode == 'flash':
            ydata = np.squeeze(wn)[:int(2/dt_flash)]
            tdata1 = np.arange(0,end,dt_flash)

        

        """
        1sigma` determines uncertainty in data
        A 1-d `sigma` should contain values of standard deviations of
                      errors in `ydata`. In this case, the optimized function is
                      ``chisq = sum((r / sigma) ** 2)``
        """
        # sigma=ystd # Determines the uncertainty in `ydata`
        sigma = None
        popt, pcov = curve_fit(bp, tdata1, ydata,sigma=sigma,p0=p0,bounds=bounds,maxfev=100000)
        
        if verbose: print(k,popt)

        r2 = r2_score(ydata,bp(tdata1, *popt))
        
        temporal_filter_list.append(ydata)
        popt_list.append(popt)
        r2_list.append(r2)
 
        
        
        """ Numerical predictions from parameterized filter """
        resp = get_sine_tuning(bp(np.arange(0,10,dt),*popt),freq,dt=dt,max_start=5) # custom function
        #ax[1].plot(freq,resp['max']/np.max(resp['max']),color=c,linewidth=5,alpha=1-i*cn) # 
        
        curve = resp['max']/np.max(resp['max'])
        numerical_freq.append(curve)
    
        """ Numerical area under curve """
        ind = np.where(freq>=bound_freq)[0][0] # only count above 0.1 Hz
        ind_max = np.argmax(curve[ind:])
        
        area_under_curve.append(np.sum(resp['max'][ind:]/np.max(resp['max']))*dF)
        
        """ Numerical FWHM """
        cc = curve[ind:]
        span = cc[cc >= 0.5] # only works if smooth
        fwhm_list.append(len(span)*dF)
        
        """ Max Frequency """
        max_freq_list.append(freq[ind+ind_max]) # because only looked for max above 0.1 Hz
        
        
    """ Parameterize Average temporal filter, """
    
    """ Average temporal convolved (sanity check) """
    if mode == 'white_noise':
        mn_temporal = return_mean_temporal(trace_dict)
    
        #tdata = np.arange(0,end,mn_temporal['dt'])
        sigma = None
        popt, pcov = curve_fit(bp, tdata1, mn_temporal['mean'][:len(tdata1)],sigma=sigma,p0=p0,bounds=bounds,maxfev=100000)
        r2 = r2_score(mn_temporal['mean'][:len(tdata1)],bp(tdata1, *popt))
        
    if mode == 'pred_dict':
        mn_temporal = np.mean(temporal_filter_list,axis=0)
    
        #tdata = np.arange(0,end,mn_temporal['dt'])
        sigma = None
        popt, pcov = curve_fit(bp, tdata1, mn_temporal[:len(tdata1)],sigma=sigma,p0=p0,bounds=bounds,maxfev=100000)
        r2 = r2_score(mn_temporal[:len(tdata1)],bp(tdata1, *popt))
        
            
    
    if mode == 'flash':
        mn_temporal = np.mean(temporal_filter_list,axis=0)
        sigma = None
        popt, pcov = curve_fit(bp, tdata1, mn_temporal[:len(tdata1)],sigma=sigma,p0=p0,bounds=bounds,maxfev=100000)
        r2 = r2_score(mn_temporal[:len(tdata1)],bp(tdata1, *popt))
        
    if verbose: print('parameterized mean temporal filter ',popt)

    
    resp = get_sine_tuning(bp(np.arange(0,10,dt), *popt),freq,dt=dt,max_start=5) # custom function
    numerical_freq_mn_temporal = resp['max']
        
  
    # Lists for each cell
    parameterized_dict['numerical_freq'] = numerical_freq
    parameterized_dict['area_under_curve'] = area_under_curve
    parameterized_dict['fwhm_list'] = fwhm_list
    parameterized_dict['max_freq_list'] = max_freq_list
    parameterized_dict['popt_list'] = popt_list # just to keep track
    parameterized_dict['temporal_filter_list'] = temporal_filter_list # just to keep track
    
    # Average
    parameterized_dict['mn_temporal_popt'] = popt
    parameterized_dict['mn_temporal'] = mn_temporal
    parameterized_dict['numerical_freq_mn_temporal'] = numerical_freq_mn_temporal # not normalized
    
    
    
    parameterized_dict['r2_list'] = r2_list
    parameterized_dict['tdata'] = tdata
    parameterized_dict['tdata1'] = tdata1
    parameterized_dict['freq'] = freq
    
    return parameterized_dict
    
    
    
    
    
    
    
    
    
    
    
