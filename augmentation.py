import logging
import os
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import cv2
import random
from glob import glob
from datetime import datetime
from shutil import copyfile
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.misc import imsave, imread

from PIL import Image, ImageOps, ImageEnhance
import math
from math import floor, ceil
import scipy.ndimage

import utils
import image_utils
import configuration as config

def augmentation_function(images, labels):
    '''
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size. 
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction.     
    ''' 
    
    # Define in configuration.py which operations to perform
    do_rotations_range = config.do_rotations_range
    do_rotations_90 = config.do_rotations_90
    do_rotation_180 = config.do_rotations_180
    do_rotation_270 = config.do_rotation_270
    do_rotation_reshape = config.do_rotation_reshape
    do_rotation = config.do_rotation
    crop = config.crop
    do_fliplr = config.do_fliplr
    do_flipud = config.do_flipud
    RandomContrast = config.RandomContrast
    blurr = config.blurr
    SaltAndPepper = config.SaltAndPepper
    Multiply = config.Multiply
    ElasticTransformation = config.ElasticTransformation
    Pad = config.Pad
    
    # Probability to perform a generic operation
    prob = config.prob
    if 0.0 <= prob <= 1.0:
        
        new_images = []
        new_labels = []
        num_images = images.shape[0]
        
        for i in range(num_images):
            
            #extract the single image
            img = np.squeeze(images[i,...])
            lbl = np.squeeze(labels[i,...])
            
            # ROTATE (Min,Max)
            # The operation will rotate an image by a random amount, within a range
            # specified
            if do_rotations_range:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angles = config.rg
                    random_angle = np.random.uniform(angles[0], angles[1])
                    img = image_utils.rotate_image(img, random_angle)
                    lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

            # ROTATE 90°
            if do_rotations_90:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = 90
                    img = image_utils.rotate_image(img, angle)
                    lbl = image_utils.rotate_image(lbl, angle, interp=cv2.INTER_NEAREST)
        
            # ROTATE 180°
            if do_rotations_180:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = 180
                    img = image_utils.rotate_image(img, angle)
                    lbl = image_utils.rotate_image(lbl, angle, interp=cv2.INTER_NEAREST)
           
            # ROTATE 270°
            if do_rotations_270:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = 270
                    img = image_utils.rotate_image(img, angle)
                    lbl = image_utils.rotate_image(lbl, angle, interp=cv2.INTER_NEAREST)
          
            # ROTATE (generic angle with reshape)
            if do_rotation_reshape:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = config.angle
                    img = scipy.ndimage.rotate(img,angle)
                    lbl = scipy.ndimage.rotate(lbl,angle)
            
            # ROTATE (generic angle)
            if do_rotation:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = config.angle
                    img = scipy.ndimage.rotate(img,angle,reshape=False)
                    lbl = scipy.ndimage.rotate(lbl,angle,reshape=False)
            
            # RANDOM CROP SCALE
            if crop:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    augmenters = [iaa.Crop(px=config.offset)]
                    seq = iaa.Sequential(augmenters, random_order=True)
                    img = seq.augment_image(img)
                    lbl = seq.augment_image(lbl)

            # RANDOM FLIP Lelf/Right
            if do_fliplr:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    img = np.fliplr(img)
                    lbl = np.fliplr(lbl)
                
            # RANDOM FLIP  up/down
            if do_flipud:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    img = np.flipud(img)
                    lbl = np.flipud(lbl)
                
            # RANDOM CONTRAST
            # Random change the passed image contrast
            '''
            min_factor: The value between 0.0 and max_factor
             that define the minimum adjustment of image contrast.
             The value  0.0 gives s solid grey image, value 1.0 gives the original image.
            max_factor: A value should be bigger than min_factor.
             that define the maximum adjustment of image contrast.
             The value  0.0 gives s solid grey image, value 1.0 gives the original image.
            '''
            if RandomContrast:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    factor = np.random.uniform(config.min_factor, config.max_factor)
                    img = ImageEnhance.Contrast(img).enhance(factor)
            
            # BLURRING 
            # only on image not label
            if blurr:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    inter = config.sigma
                    sigma = np.random.uniform(inter[0],inter[1])
                    img = scipy.ndimage.gaussian_filter(img, sigma)
           
            # SaltAndPepper (image should be normalized between 0 and 1)
            if SaltAndPepper:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    dens = config.density
                    if (dens>1.00) | (dens<0.00):
                        dens = 0.05
                    coords = [np.random.randint(0, d - 1, int(np.ceil(dens * img.size * 0.5))) for d in img.shape]
                    img[coords] = 1
                    coords = [np.random.randint(0, d - 1, int(np.ceil(dens * img.size * 0.5))) for d in img.shape]
                    img[coords] = 0
            
            # Multiply
            if Multiply:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    augmenters = [iaa.Multiply(config.m)]
                    seq = iaa.Sequential(augmenters, random_order=True)
                    img = seq.augment_image(img)
            
            # ElasticTransformation
            '''
            The augmenter has the parameters ``alpha`` and ``sigma``. ``alpha`` controls the strength of the
            displacement: higher values mean that pixels are moved further. ``sigma`` controls the
            smoothness of the displacement: higher values lead to smoother patterns -- as if the
            image was below water -- while low values will cause indivdual pixels to be moved very
            differently from their neighbours, leading to noisy and pixelated images.
            A relation of 10:1 seems to be good for ``alpha`` and ``sigma`
            '''
            if ElasticTransformation:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    augmenters = [iaa.ElasticTransformation(alpha=config.alpha, sigma=config.sigma)]
                    seq = iaa.Sequential(augmenters, random_order=True)
                    img = seq.augment_image(img)
                    lbl = seq.augment_image(lbl)
            
            if Pad:
                coin_flip = np.random.uniform(low=0, high=1.0)
                if coin_flip < prob :
                    augmenters = [iaa.Pad(px=config.offset2)]
                    seq = iaa.Sequential(augmenters, random_order=True)
                    img = seq.augment_image(img)
                    lbl = seq.augment_image(lbl)
            
            
            new_images.append(img[...])
            new_labels.append(lbl[...])

        sampled_image_batch = np.asarray(new_images)
        sampled_label_batch = np.asarray(new_labels)

        return sampled_image_batch, sampled_label_batch
    
    else:
        logging.warning('Probability must be in range 0.0/1.0!!')
