import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
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
        input_folder=config.data_root ,                  #data_root    test_data_root
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

            #x = cv2.normalize(x, dst=None, alpha=config.min, beta=config.max, norm_type=cv2.NORM_MINMAX)

            x = image_utils.reshape_2Dimage_to_tensor(x)
            y = image_utils.reshape_2Dimage_to_tensor(y)
            logging.info('x')
            logging.info(x.shape)
            logging.info(x.dtype)
            logging.info(x.min())
            logging.info(x.max())
            plt.imshow(np.squeeze(x))
            plt.gray()
            plt.axis('off')
            plt.show()
            logging.info('y')
            logging.info(y.shape)
            logging.info(y.dtype)
            
            feed_dict = {
                images_pl: x,
            }

            mask_out, softmax_out = sess.run([mask_pl, softmax_pl], feed_dict=feed_dict)
            logging.info('mask_out')
            logging.info(mask_out.shape)
            logging.info('softmax_out')
            logging.info(softmax_out.shape)
            fig = plt.figure(1)
            ax1 = fig.add_subplot(241)
            ax1.set_axis_off()
            ax1.imshow(np.squeeze(x), cmap='gray')
            ax2 = fig.add_subplot(242)
            ax2.set_axis_off()
            ax2.imshow(np.squeeze(y))
            ax3 = fig.add_subplot(243)
            ax3.set_axis_off()
            ax3.imshow(np.squeeze(mask_out))
            ax1.title.set_text('a')
            ax2.title.set_text('b')
            ax3.title.set_text('c')
           
            ax5 = fig.add_subplot(245)
            ax5.set_axis_off()
            ax5.imshow(np.squeeze(softmax_out[...,0]))
            ax6 = fig.add_subplot(246)
            ax6.set_axis_off()
            ax6.imshow(np.squeeze(softmax_out[...,1]))
            ax7 = fig.add_subplot(247)
            ax7.set_axis_off()
            ax7.imshow(np.squeeze(softmax_out[...,2]))
            ax8 = fig.add_subplot(248)
            ax8.set_axis_off()
            ax8.imshow(np.squeeze(softmax_out[...,3]))  #cmap=cm.Blues
            ax5.title.set_text('d')
            ax6.title.set_text('e')
            ax7.title.set_text('f')
            ax8.title.set_text('g')
            plt.gray()
            plt.show()
            
            logging.info('mask_out type')
            logging.info(mask_out.dtype)

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
