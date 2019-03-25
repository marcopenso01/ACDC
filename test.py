import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
import cv2

import configuration as config
import utils
import acdc_data
import image_utils
import model
import logging

def main(config):

    # Load data
    data = acdc_data.load_and_maybe_process_data(
        input_folder=config.test_data_root ,
        preprocessing_folder=config.preprocessing_folder,
        mode=config.data_mode,
        size=config.image_size,
        target_resolution=config.target_resolution,
        force_overwrite=False
    )

    batch_size = 1

    image_tensor_shape = [batch_size] + list(config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, config)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()


    with tf.Session() as sess:

        sess.run(init)

        checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        saver.restore(sess, checkpoint_path)

        while True:

            ind = np.random.randint(data['images_test'].shape[0])

            x = data['images_test'][ind,...]
            y = data['masks_test'][ind,...]
            
            for img in x:
                if config.standardize:
                    img = image_utils.standardize_image(img)
                if config.normalize:
                    img = cv2.normalize(img, dst=None, alpha=config.min, beta=config.max, norm_type=cv2.NORM_MINMAX)


            x = image_utils.reshape_2Dimage_to_tensor(x)
            y = image_utils.reshape_2Dimage_to_tensor(y)

            feed_dict = {
                images_pl: x,
            }

            mask_out, softmax_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)

            fig = plt.figure()
            ax1 = fig.add_subplot(241)
            ax1.imshow(np.squeeze(x), cmap='gray')
            ax2 = fig.add_subplot(242)
            ax2.imshow(np.squeeze(y))
            ax3 = fig.add_subplot(243)
            ax3.imshow(np.squeeze(mask_out))

            ax5 = fig.add_subplot(245)
            ax5.imshow(np.squeeze(softmax_out[...,0]))
            ax6 = fig.add_subplot(246)
            ax6.imshow(np.squeeze(softmax_out[...,1]))
            ax7 = fig.add_subplot(247)
            ax7.imshow(np.squeeze(softmax_out[...,2]))
            ax8 = fig.add_subplot(248)
            ax8.imshow(np.squeeze(softmax_out[...,3]))
            
            plt.axis('off')
            plt.show()

    data.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a 2D network on slices from the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()
    
    base_path = config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    logging.info(model_path)

   
    init_iteration = main(config)
