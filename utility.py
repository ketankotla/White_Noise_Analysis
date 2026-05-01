#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:00:35 2021

@author: katherinedelgado and Erin Barnhart
"""
import glob
import csv
import numpy
from pylab import *
import scipy.ndimage as ndimage
from PIL import Image, ImageSequence
import skimage.util as sku
import skimage.transform as skt
from pystackreg import StackReg
import alignment
from readlif.reader import LifFile

def get_file_names(parent_directory,file_type = 'all',label = ''):
    if file_type == 'csv':
        file_names = glob.glob(parent_directory+'/*'+label+'*.csv')
    elif file_type == 'all':
        file_names = glob.glob(parent_directory+'/*'+label+'*')
    return file_names


def read_csv_file(filename, header=True):
    data = []
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            data.append(row)
    if header==True:
        out_header = data[0]
        out = data[1:]
        return out, out_header
    else:
        return out

def write_csv(data,header,filename):
    with open(filename, "w") as f:
        writer= csv.writer(f)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)


def get_input_dict(row,header):
    input_dict = {}
    for r,h in zip(row,header):
        input_dict[h]=r
    return input_dict

def read_tif(filename):
    tiff = Image.open(filename)
    return numpy.asarray(tiff)

def read_tifs(filename):
    # Open image as PIL ImageSequence
    tiffs = Image.open(filename)
    # Convert each image page to a numpy array, convert array stack into 3D array
    return numpy.array([numpy.array(page) for page in ImageSequence.Iterator(tiffs)], dtype=np.uint8)

def save_tif(image_array,filename):
    out = Image.fromarray(image_array)
    out.save(filename)

def alignMultiPageTiff(ref, img):
    tmat = []
    aligned_images = []
    for t in img:
        #print(t.shape)
        mat = alignment.registerImage(ref, t, mode="rigid") #transformation matrix
        a = alignment.transformImage(t,mat) #aligned image
        aligned_images.append(a)
        tmat.append(mat)
    A = numpy.asarray(aligned_images)
    return A, tmat

def alignFromMatrix(img, tmat):
    aligned_images = []
    for t,mat in zip(img, tmat):
        a = alignment.transformImage(t, mat)
        aligned_images.append(a)
    A = numpy.asarray(aligned_images)
    return A

def saveMultipageTif(numpyArray, saveFile):
   # use list comprehension to convert 3D numpyArray into 1D pilArray, which is a list of 2D PIL images (8-bit grayscale via mode="L")
   pilArray = [
       Image.fromarray(numpyArray[x], mode="L")
       for x in range(numpyArray.shape[0])]
   # saveFile is a complete string path in which to save your multipage image. Note, saveFile should end with ".tif"
   pilArray[0].save(
       saveFile, compression="tiff_deflate",
       save_all=True, append_images=pilArray[1:])

def loadLifFile(file):
    """
    Load entire lif file as an object from which to extract imaging samples
    @param file: path to .lif file to be loaded
    @type file: string
    @return: LifFile iterable that contains all imaged samples in lif file
    @rtype: readlif.reader.LifFile
    """
    lif = LifFile(file)
    return lif

def getLifImage(lif, idx, dtype=numpy.uint8):
    """
    Extract an imaged sample as a hyperstack from a pre-loaded .lif file
    @param lif: LifFile iterable that contains all imaged samples in lif file
    @type lif: readlif.reader.LifFile
    @param idx: index of desired image within lif file (0-indexed)
    @type idx: int
    @param dtype: data type of image array to be saved
    @type dtype: np.dtype, passed to np.ndarray.astype constructor
    @return: lif imaging sample converted to 5D hyperstack array
    @rtype: numpy.ndarray
    """
    image = lif.get_image(img_n=idx)
    stack = [[[numpy.array(image.get_frame(t=t, c=c))
               for c in range(image.channels)]
             for t in range(image.dims.t)]]
    stack = numpy.array(stack, dtype=dtype)
    return stack










