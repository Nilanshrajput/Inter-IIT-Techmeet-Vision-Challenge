import os
import numpy as np
import tifffile
import json
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
import cv2 as cv
from skimage import io
from skimage.segmentation import quickshift


# Load the tif file
#grounf truth
tif_data = tifffile.imread('interiit_techmeet/The-Eye-in-the-Sky-dataset/gt/13.tif')
plt.imshow(tif_data)

#conert to uint8 from uint16

#img = a.astype(np.uint8)
#print (a.dtype)
#print (a)
#plt.imshow(a)


#not correct image
#discard this cell above



# sat image
imgarr = io.imread('interiit_techmeet/The-Eye-in-the-Sky-dataset/sat/13.tif')

#takes first 3 band of image(rgb)
a_rgb=(imgarr[...,0:3])
print (a_rgb.shape)
print(a_rgb.dtype)
#print (imgarr)
#print (imgarr.shape)
#print (imgarr.dtype)


def scale_percentile(matrix):
    """Fixes the pixel value range to 2%-98% original distribution of values"""
    orig_shape = matrix.shape
    matrix = np.reshape(matrix, [matrix.shape[0]*matrix.shape[1], 3]).astype(float)
    
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    
    matrix = (matrix - mins[None,:])/maxs[None,:]
    matrix = np.reshape(matrix, orig_shape)
    matrix = matrix.clip(0,1)
    return matrix
fixed_im = scale_percentile(a_rgb)


plt.imshow(fixed_im)
print (fixed_im.dtype)
#seprating 3 band of input sat img
#band 1 (R)

band_r=fixed_im[...,0]
print(fixed_im.shape)
print (band_r)
print(band_r.shape)
#print (band_r.dtype)

band_g=fixed_im[...,1]
band_b=fixed_im[...,2]

# nir band uint16 
band_nir_16=imgarr[...,3]
#nir band converted to float64
band_nir_f64=scale_percentile(imgarr[...,1:4])[...,2]
print(band_nir_16)
#print (band_nir.flatten())
#flat_r=np.reshape(band_nir.flatten(),band_nir.shape)
print(flat_r)

#print(band_nir_f64)
-