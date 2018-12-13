import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import cv2

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
    
    do_rotations = config.do_rotations
    do_rotations_90 = config.do_rotations_90
    do_rotation_180 = config.do_rotations_180
    do_scaleaug = config.do_rotations
    do_fliplr = config.do_fliplr
    do_flipud = config.do_flipud

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
        if do_rotations:
            angles = config.angles
            random_angle = np.random.uniform(angles[0], angles[1])
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # ROTATE 90°
        if do_rotations_90:
            angles = 90
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
        
        # ROTATE 180°
        if do_rotations_180:
            angles = 180
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
            
        # RANDOM CROP SCALE
        if do_scaleaug:
            offset = config.offset
            n_x, n_y = img.shape
            r_y = np.random.randint(n_y-offset, n_y)
            p_x = np.random.randint(0, n_x-r_y)
            p_y = np.random.randint(0, n_y-r_y)

            img = image_utils.resize_image(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y))
            lbl = image_utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

        # RANDOM FLIP Lelf/Right
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                lbl = np.fliplr(lbl)
                
        # RANDOM FLIP  up/down
        if do_flipud:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.flipud(img)
                lbl = np.flipud(lbl)


        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch
