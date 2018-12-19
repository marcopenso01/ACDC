import os
from glob import glob
import time
import re
import argparse
import pandas as pd
import h5py
import tensorflow as tf
import shutil

import SimpleITK as sitk
from multiprocessing import pool
import pickle
import numpy as np
import logging

import utils
import image_utils
import model as model
import acdc_data
import configuration as config
import augmentation as aug

# Load data
data = acdc_data.load_and_maybe_process_data(
        input_folder=config.input_folder,
        preprocessing_folder=config.preprocessing_folder,
        mode=config.data_mode,
        size=config.image_size,
        target_resolution=config.target_resolution,
        force_overwrite=False,
        split_test_train=config.split_test_train 
)

# the following are HDF5 datasets, not numpy arrays
images_train = data['images_train']
labels_train = data['masks_train']

if config.split_test_train:
        images_val = data['images_test']
        labels_val = data['masks_test']

logging.info('Data summary:')
logging.info('Images:')
logging.info(images_train.shape)
logging.info(images_train.dtype)
logging.info('Labels:')
logging.info(labels_train.shape)
logging.info(labels_train.dtype)        
logging.info('Before data_augmentation the number of images is:')
logging.info(images_train.shape[0])

#augmentation
sampled_image_batch, sampled_label_batch = aug.augmentation_function(images_train,labels_train)
images_train = np.concatenate((images_train,sampled_image_batch))
labels_train = np.concatenate((labels_train,sampled_label_batch))

logging.info('After data_augmentation the number of images is:')
logging.info(images_train.shape[0])

#shuffle the dataset
n_images = images_train.shape[0]
random_indices = np.arange(n_images)
np.random.shuffle(random_indices)
