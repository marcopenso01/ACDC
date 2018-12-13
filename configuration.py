import tensorflow as tf

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
nlabels = 4
split_test_train = False

# Training settings
batch_size = 10
learning_rate = 0.01
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'weighted_crossentropy'  # crossentropy/weighted_crossentropy/dice/dice_onlyfg

# Augmentation settings
augment_batch = False
do_rotations = True
angles = (-15,15)    #define range (min,max)
do_rotations_90 = False
do_rotation_180 = False
do_scaleaug = False
do_fliplr = False

# Paths settings (we need to mount our drive before)
input_folder = '/content/drive/My Drive/ACDC_challenge/train'
preprocessing_folder = '/content/drive/My Drive/preproc_data'
