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
# import imgaug as ia
# from imgaug import augmenters as iaa
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
    :param images: A numpy array of shape [batch, X, Y], normalized between 0-1
    :param labels: A numpy array containing a corresponding label mask     
    ''' 
    
    # Define in configuration.py which operations to perform
    do_rotation_90 = config.do_rotation_90
    do_fliplr = config.do_fliplr
    do_flipud = config.do_flipud
    
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
            
            # ROTATE
            # rotate an image 90°
            if do_rotation_90:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    angle = 90
                    imgg = image_utils.rotate_image(img, angle)
                    lbll = image_utils.rotate_image(lbl, angle, interp=cv2.INTER_NEAREST)
                    if (random.randint(0,1)):
                        x = random.randint(-11,11)
                        y = random.randint(-11,11)
                        M = np.float32([[1,0,x],[0,1,y]])
                        imgg = cv2.warpAffine(imgg,M,(212,212))
                        lbll = cv2.warpAffine(lbll,M,(212,212))
                    new_images.append(imgg[...])
                    new_labels.append(lbll[...])

            # FLIP Lelf/Right (+ rotation)
            if do_fliplr:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    imgg = np.fliplr(img)
                    lbll = np.fliplr(lbl)
                    new_images.append(imgg[...])
                    new_labels.append(lbll[...])
                    # rotation
                    angle = 90
                    imgg = image_utils.rotate_image(imgg, angle)
                    lbll = image_utils.rotate_image(lbll, angle, interp=cv2.INTER_NEAREST)
                    if (random.randint(0,1)):
                        x = random.randint(-11,11)
                        y = random.randint(-11,11)
                        M = np.float32([[1,0,x],[0,1,y]])
                        imgg = cv2.warpAffine(imgg,M,(212,212))
                        lbll = cv2.warpAffine(lbll,M,(212,212))
                    new_images.append(imgg[...])
                    new_labels.append(lbll[...])
                
            # FLIP  up/down (+ rotation)
            if do_flipud:
                coin_flip = np.random.uniform(low=0.0, high=1.0)
                if coin_flip < prob :
                    imgg = np.flipud(img)
                    lbll = np.flipud(lbl)
                    new_images.append(imgg[...])
                    new_labels.append(lbll[...])
                    # rotation
                    angle = 90
                    imgg = image_utils.rotate_image(imgg, angle)
                    lbll = image_utils.rotate_image(lbll, angle, interp=cv2.INTER_NEAREST)
                    if (random.randint(0,1)):
                        x = random.randint(-11,11)
                        y = random.randint(-11,11)
                        M = np.float32([[1,0,x],[0,1,y]])
                        imgg = cv2.warpAffine(imgg,M,(212,212))
                        lbll = cv2.warpAffine(lbll,M,(212,212))
                    new_images.append(imgg[...])
                    new_labels.append(lbll[...])
                    
            
            # FLIP up/down + FLIP felft/right (+ rotation)
            coin_flip = np.random.uniform(low=0.0, high=1.0)
            if coin_flip < prob :
                imgg = np.flipud(img)
                lbll = np.flipud(lbl)
                imgg = np.fliplr(imgg)
                lbll = np.fliplr(lbll)
                new_images.append(imgg[...])
                new_labels.append(lbll[...])
                angle = 90
                imgg = image_utils.rotate_image(imgg, angle)
                lbll = image_utils.rotate_image(lbll, angle, interp=cv2.INTER_NEAREST)
                if (random.randint(0,1)):
                    x = random.randint(-11,11)
                    y = random.randint(-11,11)
                    M = np.float32([[1,0,x],[0,1,y]])
                    imgg = cv2.warpAffine(imgg,M,(212,212))
                    lbll = cv2.warpAffine(lbll,M,(212,212))
                new_images.append(imgg[...])
                new_labels.append(lbll[...])
            
            # RANDOM CROPPING
  #          if crop:
  #              coin_flip = np.random.uniform(low=0.0, high=1.0)
  #              if coin_flip < prob :
  #                  augmenters = [iaa.Crop(px=config.offset)]
  #                  seq = iaa.Sequential(augmenters, random_order=True)
  #                  imgg = seq.augment_image(img)
  #                  lbll = seq.augment_image(lbl)
  #                  if (random.randint(0,1)):
  #                      x = random.randint(-11,11)
  #                      y = random.randint(-11,11)
  #                      M = np.float32([[1,0,x],[0,1,y]])
  #                      imgg = cv2.warpAffine(imgg,M,(212,212))
  #                      lbll = cv2.warpAffine(lbll,M,(212,212))
  #                  new_images.append(imgg[...])
  #                  new_labels.append(lbll[...])

            # ROTATION + CROP
  #          random_angle = np.random.uniform(angles[0], angles[1])
  #          imgg = image_utils.rotate_image(img, random_angle)
  #          lbll = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
  #          augmenters = [iaa.Crop(px=config.offset)]
  #          seq = iaa.Sequential(augmenters, random_order=True)
  #          imgg = seq.augment_image(imgg)
  #          lbll = seq.augment_image(lbll)
  #          if (random.randint(0,1)):
  #              x = random.randint(-11,11)
  #              y = random.randint(-11,11)
  #              M = np.float32([[1,0,x],[0,1,y]])
  #              imgg = cv2.warpAffine(imgg,M,(212,212))
  #              lbll = cv2.warpAffine(lbll,M,(212,212))
  #          new_images.append(imgg[...])
  #          new_labels.append(lbll[...])
            

        sampled_image_batch = np.asarray(new_images)
        sampled_label_batch = np.asarray(new_labels)

        return sampled_image_batch, sampled_label_batch
    
    else:
        logging.warning('Probability must be in range [0.0,1.0]!!')
