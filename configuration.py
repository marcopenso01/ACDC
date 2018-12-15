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
do_rotations_range = True   #random rotation in range "rg" (min,max)
rg = (-15,15)     
do_rotations_90 = False     #rotation 90°
do_rotation_180 = False     #rotation 180°
do_rotation_270 = False     #rotation 270°
do_rotation_reshape = False #rotation of a specific 'angle' with reshape
do_rotation = False         #rotation of a specific 'angle'
angle = 45
crop = False                #crops/cuts away pixels at the sides of the image
offset = (10, 30)           #The number of pixels to crop away on each side of the image
                            #crops EACH side by a random value from the range (min,max) pixel
do_fliplr = False           #Flip array in the left/right direction
do_flipud = False           #Flip array in the up/down direction.
RandomContrast= False       #Random change contrast of an image
min_factor = 1.0
max_factor = 1.0
blurr = False               #Blurring the image with gaussian filter with random 'sigma'
sigma = (0.5,1.0)           #generate a random sigma in range(min,max)
SaltAndPepper = False
density = 0.05              #Noise density for salt and pepper noise, specified as a numeric scalar.
Multiply = False            #Multiply all pixels in an image with a specific value (m)
m = 1
ElasticTransformation = False #Moving pixels locally around using displacement fields.
alpha = (0.0, 70.0)         #alpha and sigma can be a number or tuple (a, b)
sigma = 5.0                 #If tuple a random value from range ``a <= x <= b`` will be used
Pad = False                 #Pad image, i.e. adds columns/rows to them
offset2 = (10,30)           #number of pixels to crop away on each side of the image (a,b)
                            #each side will be cropped by a random amount in the range `a <= x <= b`
  
prob = 0.5                  #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (we need to mount our drive before)
input_folder = '/content/drive/My Drive/ACDC_challenge/train'
preprocessing_folder = '/content/drive/My Drive/preproc_data'
