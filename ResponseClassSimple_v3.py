#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:32 2021

@author: Erin Barnhart
"""
import numpy as numpy
import scipy as scipy

class Response(object):
    """class attributes"""
    
    """instance attributes"""
    _instance_data = {'sample_number':None,
                      'sample_name':None,
                      'ROI_number':None,
                      'reporter':None,
                      'genotype':None,
                      'age':None,
                      'compartment':None,
                      'F':[],
                      'stimulus_name':None,
                      'global_time':[],
                      'stim_time':[],
                      'stim_state':[],
                      'stim_type':[],
                      'stim_time_avg':[],
                      'stim_type_avg':[],
                      'DFF_avg':[],
                      'stim_type_dict':{},
                      'time_step':None,
                      'units':'seconds',
                      'smoothed':False}
    
    def __init__(self, **kws):
        
        """self: is the instance, __init__ takes the instace 'self' and populates it with variables"""
        
        for attr, value in self._instance_data.items():
            if attr in kws:
                value = kws[attr]
                setattr(self,attr,value)
            else:
                setattr(self,attr,value)
        
    """instance methods"""
    
    
    """find median fluorescence over time"""
    def med(self):
        return np.median(self.F)
        #self.median.append(median)

    """smooth raw fluoresence"""
    def smooth(self,sigma = 1):
        smoothed = scipy.ndimage.gaussian_filter1d(self.F,sigma)
        self.F = smoothed
        self.smoothed = True
        return smoothed
        

    """measure DFF from binned images. ONLY WORKS FOR ONE STIM TYPE AT THE MOMENT"""
    def measure_dff_binned(self,epochs,epoch_length,base_t1,base_t2,t):
        """epochs is a list stim types (integers, starting at 1, incrementing by 1 for each new type
        epoch_length is the number of image frames per epoch
        base_t1 and base_t2 are the first and last time points for calculating the baseline F for each epoch"""
        
        min_index = 0
        max_index = epoch_length
        DFF=[]
        T_avg = []
        stim_types = []
        for e in epochs:
        	f = self.F[min_index:max_index]
        	baseline = numpy.average(self.F[int(base_t1/t)+min_index:int(base_t2/t)+min_index])
        	DFF.append((f-baseline)/baseline)
        	T_avg.append(list(numpy.arange(0,len(f)*t,t)))
        	stim_types.append(self.stim_type[min_index:max_index])
        	min_index = min_index+epoch_length
        	max_index = max_index+epoch_length

        self.DFF_avg = DFF
        self.stim_type_avg = stim_types
        self.stim_time_avg = T_avg
        #print(T_avg)

        return DFF, stim_types, T_avg


