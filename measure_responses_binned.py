import numpy
import scipy.ndimage as ndimage
import ResponseTools_v3 as rt
import utility
import os
from pylab import *

parent_dir = '../Leica_Processing/20251218_Mi1_ASAP/'

input_csv = parent_dir+'inputs_binned.csv'
rows,header = utility.read_csv_file(input_csv)

for row in rows[:]:
    
    #get input parameters for each sample
    input_dict = rt.get_input_dict(row,header)
    print('analyzing sample ' + input_dict['sample_name'] + ' ' + input_dict['output_name'] + ' ' + input_dict['mask_name'] + ' ' + input_dict['stimulus_name'])

    #set up paths
    sample_dir = parent_dir+input_dict['sample_name']
    image_dir = sample_dir+'/aligned_images/'
    mask_dir = sample_dir+'/masks/'
    stim_dir = sample_dir+'/stim_files/'
    plot_dir = sample_dir+'/plots/'
    output_dir = sample_dir+'/measurements/'

    #get file names for images, mask, and stim file
    if input_dict['aligned']=='TRUE':
        image_file = image_dir + input_dict['ch1_name']+'-aligned.tif'
    else:
        image_file = image_dir + input_dict['ch1_name']+'.tif'
    mask_file = mask_dir + input_dict['mask_name']+'.tif'
    stim_file = utility.get_file_names(stim_dir,file_type = 'csv',label = input_dict['stimulus_name'])[0]

    #load and parse stim file
    stim_data,stim_header = rt.count_frames(stim_file,input_dict)

    if input_dict['verbose']=='TRUE':
        parsed_stim_dir = stim_dir+'parsed/'
        if not os.path.exists(parsed_stim_dir):
            os.makedirs(parsed_stim_dir)
        utility.write_csv(stim_data,stim_header,parsed_stim_dir + 'parsed-'+input_dict['stimulus_name'] +'.csv')

    #load images and mask
    I = utility.read_tifs(image_file)
    mask = utility.read_tifs(mask_file)

    #bin images
    binned_images,T,epochs = rt.bin_images(I,stim_data,input_dict)

    #smooth images
    sigma_xy = float(input_dict['sigma_xy'])
    sigma_t = float(input_dict['sigma_t'])
    smoothed_images = ndimage.gaussian_filter(binned_images,sigma=(sigma_t,sigma_xy,sigma_xy))
    S = numpy.asarray(smoothed_images,dtype='uint8')
    utility.saveMultipageTif(S,image_dir+input_dict['ch1_name']+'-binned.tif')

    #generate response objects
    response_objects, labels = rt.extract_response_objects_from_binned_images(smoothed_images,mask,epochs,input_dict)

    raw_df = rt.save_raw_responses_dataframe(response_objects,output_dir+input_dict['output_name']+'-raw.csv')
    rt.plot_raw_responses(raw_df,plot_dir+input_dict['output_name']+'-raw.png')
    rt.measure_dff_binned(response_objects,epochs,input_dict)
    avg_df=rt.save_avg_responses_dataframe(response_objects,output_dir+input_dict['output_name']+'-DFF.csv')
    rt.plot_avg_responses(avg_df,plot_dir+input_dict['output_name']+'-average.png',hue_variable='stim_type',style_variable = 'ROI')
    utility.save_tif(labels[0],mask_dir+input_dict['mask_name']+'-labels.tif')


