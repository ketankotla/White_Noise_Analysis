
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:39:08 2022

@author: Katherine Delgado and Erin Barnhart
"""
import csv
import alignment 
import numpy 
import utility
import skimage.util as sku
import skimage.transform as skt
from pystackreg import StackReg
from PIL import Image, ImageSequence
import os

'''Script to parse images from lif files, align them, and output aligned images and an average projection for masking'''

'''Specify parent directory. This should be a folder that will house all z-plane directories'''

parent_dir = '../Mi4_forKetan/' #make sure to include the final slash

'''Specify CSV file name. This CSV should be in parent directory'''

csv_filename = parent_dir+'input_alignment.csv'
data,header = utility.read_csv_file(csv_filename)

'''Iterate over each line of the CSV where one line contains information from one image or job in the lif file'''
for d in data[:]:
    '''Each bracketed number corresponds to an index (column) of the CSV. 
        If you make changes to the CSV structure, change the indices below accordingly.''' 
    sample = d[0]
    lif_name = d[1]
    job_index = d[2]
    ch1_name = d[3]
    ch1_index = int(d[4])
    use_ch2 = d[5]
    ch2_name = d[6]
    ch2_index = int(d[7])
    use_target = d[8]
    target_name = d[9]
    target_start = int(d[10]) #in frames
    target_stop = int(d[11])
    save_avg = d[12]
    
    print(sample)

    folder_names = ['images','aligned_images','masks','measurements','plots','stim_files']
    for f in folder_names:
        new_dir = parent_dir+sample+'/'+f+'/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    
    #Check all filepaths below to make sure they match yours
    
    '''Parse lif file'''
    lifFileName = parent_dir+'lif_files/'+lif_name+'.lif'
    image = utility.loadLifFile(lifFileName)
    job = utility.getLifImage(image, job_index)
    
    '''Extract ch1 from hyperstack'''
    ch1 = job[0,:,ch1_index,:,:]
    '''Use target for alignment. Either use specified target or make target from ch1'''
    if use_target == 'TRUE':
        target_filename = parent_dir+sample+'/images/'+target_name+'.tif'
        target = utility.read_tif(target_filename)
        print('using designated target to align')
    else:
        subset_target = ch1[target_start:target_stop]
        print(subset_target.shape)
        # target = np.average(subset_target,axis=0)
        target = numpy.average(ch1,axis=0)
        print(target.shape)
        utility.save_tif(target, parent_dir+sample+'/images/'+target_name+'.tif')
        print('using automatically generated average projection to align')
    A1, tmat = utility.alignMultiPageTiff(target, ch1)
    A1=numpy.asarray(A1,dtype = 'uint8')
    utility.saveMultipageTif(ch1, parent_dir +sample+'/images/'+ch1_name+'.tif')
    utility.saveMultipageTif(A1, parent_dir +sample+'/aligned_images/'+ch1_name+'-aligned.tif')

    '''Make and save aligned average projections'''
    if save_avg == 'TRUE':
        print ('saving average projection for masking')
        AVG1 = numpy.average(A1, axis=0)
        utility.save_tif(AVG1, parent_dir+ sample+'/aligned_images/'+ch1_name+'-avg.tif')

    '''If ch2 exists, align to specified target'''
    if use_ch2 == 'TRUE':
        print ('will do ch2')
        ch2 = job[0,:,ch2_index,:,:]
        A2 = utility.alignFromMatrix(ch2, tmat)
        A2=numpy.asarray(A2,dtype = 'uint8')
        utility.saveMultipageTif(ch2, parent_dir +sample+'/images/'+ch2_name+'.tif')
        utility.saveMultipageTif(A2, parent_dir +sample+'/aligned_images/'+ch2_name+'-aligned.tif')
        if save_avg == 'TRUE':
            print ('saving average projection for masking')
            AVG2 = numpy.average(A2, axis=0)
            utility.save_tif(AVG2, parent_dir+ sample+'/aligned_images/'+ch2_name+'-avg.tif')
