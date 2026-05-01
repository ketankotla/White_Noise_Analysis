#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:00:35 2021

@author: Erin Barnhart
"""
import ResponseClassSimple_v3 as ResponseClassSimple
import numpy
from utility import *
import scipy.ndimage as ndimage
from pylab import *
from colour import Color
import scipy.stats as stats
import pandas
import seaborn



def count_frames(filename,input_dict,threshold=1): 
	"""Reads in a stimulus output file and assigns an image frame number to each stimulus frame."""
	rows,header = read_csv_file(filename) #import stim file
	R = numpy.asarray(rows,dtype='float') #convert stim file list to an array
	
	#set up the output array
	output_array=numpy.zeros((R.shape[0],R.shape[1]+3))
	output_array[:,:-3] = R
	header.extend(['delta_v','dt','frames']) 
	
	#replace intermediate values for AIN4 with 0 (if less 0.1) or 1 (if greater than 0.1)
	R[:,-1]=numpy.where(R[:,-1]<=0.1,0,R[:,-1])
	R[:,-1]=numpy.where(R[:,-1]>0.1,1,R[:,-1])

	#calculate change in voltage signal; vs==1 indicates the start of a new imaging frame
	vs = [0]
	vs.extend(R[1:,-1]-R[:-1,-1])
	output_array[:,-3]=vs # add change in voltage to output array
	output_array = output_array[numpy.argsort(output_array[:,-3])] #sort array by vs
	index = numpy.searchsorted(output_array[:,-3],1) #select rows where vs==1
	output_array = output_array[index:,:]
	output_array = output_array[numpy.argsort(output_array[:,int(input_dict['gt_index'])])] #sort by global time

	#calculate the time step between each image frame
	global_time = output_array[:,int(input_dict['gt_index'])]
	dt = [0]
	dt.extend(global_time[1:]-global_time[:-1])
	output_array[:,-2]=dt #add dt to the output array
	
	frames = numpy.arange(1,len(output_array)+1)
	output_array[:,-1] = frames #add frame numbers to the output array

	return output_array, header


def find_dropped_frames(frames,time_interval,stim_data,stim_data_OG,gt_index):
	stim_frames = stim_data[-1,-1]
	print('number of image frames = ' +str(frames))
	print('number of stimulus frames = '+str(int(stim_frames)))
	if stim_frames != frames:
		print('uh oh!')
		target_T = frames * time_interval
		stim_T = numpy.sum(stim_data[:,-2])
		print('total time should be '+str(target_T)+' seconds')
		print('total time from stim file is '+str(stim_T)+' seconds')
		max_t_step = numpy.max(stim_data[:,-2])
		if numpy.round(max_t_step/time_interval)>=2:
			print('stimulus dropped at least one frame!')
			OUT = []
			num_df = 0
			for row in stim_data:
				if numpy.round(row[-2]/time_interval)>=2:
					num_df=num_df+1
					gt_dropped = row[gt_index]-time_interval
					stim_frame = numpy.searchsorted(stim_data_OG[:,gt_index],gt_dropped)
					print('frame dropped in row '+str(stim_frame)+' of original stim file (maybe)')
					OUT.append(stim_data_OG[stim_frame])
					OUT.append(row[:-2])
				else:
					OUT.append(row[:-2])
			print('found '+str(num_df)+' potential dropped frames')
		else:
			print('stim frames and image frames do not match, but no dropped frames found...double check the stim file')
	else:
		print('looks fine!')



def parse_stim_file(stim_info_array,frame_index = -1,gt_index=0,rt_index = 1,st_index = None):
	"""Get frame numbers, global time, relative time per epoch, and stim_state (if it's in the stim_file)"""
	frames = stim_info_array[:,frame_index]
	global_time = stim_info_array[:,gt_index]
	rel_time = stim_info_array[:,rt_index]
	if st_index == 'None':
		stim_type = list(numpy.ones(len(frames),dtype = int))
	else:
		stim_type = stim_info_array[:,int(st_index)]
	return frames, global_time, rel_time, stim_type

def get_stim_position(stim_info_array,x_index = 3, y_index = 4):
	xpos = stim_info_array[:,x_index]
	ypos = stim_info_array[:,y_index]
	return xpos, ypos

def define_stim_state(rel_time,on_time,off_time):
	"""Define stimulus state (1 = ON; 0 = OFF) based on relative stimulus time."""
	stim_state = []
	for t in rel_time:
		if t>on_time and t<off_time:
			stim_state.extend([1])
		else:
			stim_state.extend([0])
	return stim_state

def segment_ROIs(mask_image):
	"""convert binary mask to labeled image"""
	labels = ndimage.measurements.label(mask_image)
	return labels[0]

def generate_ROI_mask(labels_image, ROI_int):
	return labels_image == ROI_int

def threshold_ROIs(image,labels,threshold):
	T = image >= threshold
	return T*labels

def measure_ROI_fluorescence(image,mask):
	"""measure average fluorescence in an ROI"""
	masked_ROI = image * mask
	return numpy.sum(masked_ROI) / numpy.sum(mask) 

def measure_ROI_ts(images,mask):
	out = []
	for image in images:
		out.append(measure_ROI_fluorescence(image,mask))
	return out

def measure_multiple_ROIs(images,mask_image):
	labels = segment_ROIs(mask_image)
	out = []
	num = []
	n = 1
	while n<=numpy.max(labels):
		mask = generate_ROI_mask(labels,n)
		out.append(measure_ROI_ts(images,mask))
		num.append(n)
		n=n+1
	return out,num,labels


def measure_multiple_thresholded_ROIs(images,mask_image,threshold):
	labels = segment_ROIs(mask_image)
	median_image = numpy.median(images,axis=0)
	thresholded_labels = threshold_ROIs(median_image,labels,threshold)
	out = []
	num = []
	n = 1
	while n<=numpy.max(thresholded_labels):
		mask = generate_ROI_mask(thresholded_labels,n)
		out.append(measure_ROI_ts(images,mask))
		num.append(n)
		n=n+1
	return out,num,labels

def measure_one_ROI(images,mask_image):
	labels = segment_ROIs(mask_image)
	mask = labels >= 1
	out=[measure_ROI_ts(images,mask)]
	num=[1]
	return out,num,labels

def get_input_dict(row,header):
	input_dict = {}
	for r,h in zip(row,header):
		input_dict[h]=r
	return input_dict



def bin_images(images,stim_data,input_dict):
	
	#select images frames
	I = images[int(input_dict['min_frame']):int(input_dict['max_frame'])]
	R = stim_data[int(input_dict['min_frame']):int(input_dict['max_frame'])]

	#set up time bins
	bins = numpy.arange(float(input_dict['min_t']),float(input_dict['max_t']),float(input_dict['t']))

	#set up list of epochs
	epochs = []
	n=1
	while n<= int(input_dict['max_epoch']):
		epochs.extend([n])
		n=n+1

	#sort images by 1)epoch type and 2) time relative to the start of the stimulus presentation
	#average images within each time bin and append to output list

	output_images = []

	#sort images and stim data array by epoch type
	sort_e = numpy.argsort(R[:,int(input_dict['st_index'])])
	RSE = R[sort_e]
	ISE = I[sort_e]
	print(ISE.shape)

	#get images in each epoch type
	for e in epochs:
		ei1 = numpy.searchsorted(RSE[:,int(input_dict['st_index'])],e)
		ei2 = numpy.searchsorted(RSE[:,int(input_dict['st_index'])],e+1)
		RE = RSE[ei1:ei2]
		IE = ISE[ei1:ei2]
		print(IE.shape)

		#sort images by time relative to stim start
		sort_b = numpy.argsort(RE[:,int(input_dict['rt_index'])])
		RESB = RE[sort_b]
		IESB = IE[sort_b]

		#get images in each time bin
		for b in bins:
			bi1 = numpy.searchsorted(RESB[:,int(input_dict['rt_index'])],b)
			bi2 = numpy.searchsorted(RESB[:,int(input_dict['rt_index'])],b+float(input_dict['t']))
			B = RESB[bi1:bi2]
			binned_images = IESB[bi1:bi2]
			#average images in each time bin and append average to output list
			output_images.append(numpy.average(binned_images,axis=0))
	OUT = numpy.asarray(output_images, dtype = 'float')
	return OUT,bins,epochs

def extract_response_objects_from_binned_images(binned_images,mask,epochs,input_dict):

	"""inputs are average images and a list of epochs, both generated by the bin_images function, as well as a mask and an input dictionary."""
	"""outputs a list of response objects."""

	#measure fluorscence intensities in each ROI
	if input_dict['ROI']=='one':
		responses,num,labels = measure_one_ROI(binned_images,mask)
	elif input_dict['ROI']=='thresholded':
		responses,num,labels = measure_multiple_thresholded_ROIs(binned_images,mask,float(input_dict['threshold']))
	else:
		responses,num,labels = measure_multiple_ROIs(binned_images,mask)

	print('number of ROIs = '+ str(numpy.max(num)))
	

	#get global_time, stim_time, stim_type, and stim_state

	gt = numpy.arange(1,len(binned_images)+1) #frames, not time actually
	rel_t = numpy.arange(float(input_dict['min_t']),len(binned_images)/float(input_dict['max_epoch'])*float(input_dict['t']),float(input_dict['t']))
	#print(rel_t)

	stim_time = []
	stim_type = []
	stim_state = []
	for e in epochs:
		stim_time.extend(rel_t)
		stim_type.extend(list(numpy.ones(len(rel_t))*e))

		ss = numpy.zeros(len(rel_t))
		ss[int(float(input_dict['on_time'])/float(input_dict['t'])):int(float(input_dict['off_time'])/float(input_dict['t']))]=1
		stim_state.extend(list(ss))

	#print(stim_time)
	#print(stim_type)
	#print(stim_state)

	#generate response objects
	response_objects = []
	for r, n in zip(responses, num):
		ro=ResponseClassSimple.Response(F=r,global_time = gt,stim_time = stim_time,stim_state = stim_state,ROI_number = n,stim_type = stim_type)
		response_objects.append(ro)

	return response_objects,labels


def measure_dff_binned(response_objects,epochs,input_dict):
	for ro in response_objects:
		ro.measure_dff_binned(epochs, int(input_dict['epoch_length']),float(input_dict['base_t1']),float(input_dict['base_t2']),float(input_dict['t']))

def save_raw_responses_dataframe(response_objects,filename):
	OUT = pandas.DataFrame([],columns=['ROI','global_time','stim_type','stim_time','stim_state','F'])
	for ro in response_objects:
		out = pandas.DataFrame(numpy.zeros((len(ro.F),6)),columns=['ROI','global_time','stim_type','stim_time','stim_state','F'])
		out['ROI']=ro.ROI_number
		out['global_time']=ro.global_time
		out['stim_type']=ro.stim_type
		out['stim_time']=ro.stim_time
		out['stim_state']=ro.stim_state
		out['F']=ro.F
		OUT = pandas.concat([OUT,out])
	OUT.to_csv(filename)
	return OUT

def save_avg_responses_dataframe(response_objects,filename):
	OUT = pandas.DataFrame([],columns=['ROI','stim_type','stim_time','avg_DFF'])
	for ro in response_objects:
		for dff,st,t in zip(ro.DFF_avg,ro.stim_type_avg,ro.stim_time_avg):
			out = pandas.DataFrame(numpy.zeros((len(dff),4)),columns=['ROI','stim_type','stim_time','avg_DFF'])
			out['ROI']=ro.ROI_number
			out['stim_type']=st
			out['stim_time']=t
			out['avg_DFF']=dff
			OUT = pandas.concat([OUT,out])
	OUT.to_csv(filename)
	return OUT

def plot_raw_responses(raw_df,filename):
	seaborn.lineplot(raw_df,x='global_time',y='F',hue='ROI',palette='tab10')
	savefig(filename)
	clf()

def plot_avg_responses(avg_df,filename,hue_variable='ROI',style_variable = 'stim_type'):
	seaborn.lineplot(avg_df,x='stim_time',y='avg_DFF',hue=hue_variable,style = style_variable,palette='tab10')
	savefig(filename)
	clf()









