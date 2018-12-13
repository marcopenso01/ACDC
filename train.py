import os
from glob import glob
import time
import re
import argparse
import pandas as pd
import h5py
import tensorflow as tf

import SimpleITK as sitk
from multiprocessing import pool
import pickle
import numpy as np

import utils
import image_utils
import acdc_data
import configuration as config

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

