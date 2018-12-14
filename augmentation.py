import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import cv2
import random

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
    do_scaleaug = config.do_rotations
    do_fliplr = config.do_fliplr
    do_flipud = config.do_flipud
    RandomContrast = config.RandomContrast
    blurr = config.blurr
    
    # Probability to perform a generic operation
    prob = config.prob

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for i in range(num_images):

        #extract the single image
        img = np.squeeze(images[i,...])
        lbl = np.squeeze(labels[i,...])

        # ROTATE (Min,Max)
        # The operation will rotate an image by an random amount, within a range
        # specified
        if do_rotations_range:
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                angles = config.range
                random_angle = np.random.uniform(angles[0], angles[1])
                img = image_utils.rotate_image(img, random_angle)
                lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # ROTATE 90°
        if do_rotations_90:
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                angles = 90
                img = image_utils.rotate_image(img, random_angle)
                lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
        
        # ROTATE 180°
        if do_rotations_180:
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                angles = 180
                img = image_utils.rotate_image(img, random_angle)
                lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
            
        # ROTATE 270°
        if do_rotations_270:
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                angles = 270
                img = image_utils.rotate_image(img, random_angle)
                lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
          
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
        if do_scaleaug:
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                offset = config.offset
                n_x, n_y = img.shape
                r_y = np.random.randint(n_y-offset, n_y)
                p_x = np.random.randint(0, n_x-r_y)
                p_y = np.random.randint(0, n_y-r_y)

                img = image_utils.resize_image(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y))
                lbl = image_utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

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
                sigma = config.sigma
                img = scipy.ndimage.gaussian_filter(img, sigma)



        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch
