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
do_rotations_range = True  #random rotation in range "range" (min,max)
range = (-15,15)     
do_rotations_90 = False     #rotation 90°
do_rotation_180 = False     #rotation 180°
do_rotation_270 = False     #rotation 270°
do_rotation_reshape = False #rotation of a specific 'angle' with reshape
do_rotation = False         #rotation of a specific 'angle'
angle = 45
do_scaleaug = False         #crop scale
offset = 30
do_fliplr = False           #Flip array in the left/right direction
do_flipud = False           #Flip array in the up/down direction.
RandomContrast= False       #Random change contrast of an image
min_factor = 1.0
max_factor = 1.0
blurr = False
sigma = 0.5

prob = 0.5                  #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (we need to mount our drive before)
input_folder = '/content/drive/My Drive/ACDC_challenge/train'
preprocessing_folder = '/content/drive/My Drive/preproc_data'
