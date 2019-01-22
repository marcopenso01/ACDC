import model_structure
import tensorflow as tf

experiment_name = 'unet2D'
# experiment_name = 'enet'

# Model settings
model_handle = model_structure.unet2D_bn_modified
# model_handle = model_structure.enet

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (212, 212)
target_resolution = (1.36719, 1.36719)
nlabels = 4
split_test_train = True   #divide data in train (80%) and test (20%)
train_on_all_data = False 

# Training settings
batch_size = 10
learning_rate = 0.01
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
# loss can be 'weighted_crossentropy'/'crossentropy'/'dice'/'dice_onlyfg'/'crossentropy_and_dice'
loss_type = 'weighted_crossentropy'

# Augmentation settings
do_rotation_range = True   #random rotation in range "rg" (min,max)
rg = (-30,30)     
do_rotation_90 = False      #rotation 90°
do_rotation_180 = False     #rotation 180°
do_rotation_270 = False     #rotation 270°
do_rotation_reshape = False #rotation of a specific 'angle' with reshape
do_rotation = False         #rotation of a specific 'angle'
angle = 45
crop = True                #crops/cuts away pixels at the sides of the image
offset = (10, 30)           #The number of pixels to crop away on each side of the image
                            #crops EACH side by a random value from the range (min,max) pixel
do_fliplr = True           #Flip array in the left/right direction
do_flipud = True           #Flip array in the up/down direction.
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
  
prob = 1                    #Probability [0.0/1.0] (0 no augmentation, 1 always)

# Paths settings (we need to mount MyDrive before)
input_folder = '/content/drive/My Drive/ACDC_challenge/train'
#input_folder = '/content/drive/My Drive/ACDC_challenge/test'
preprocessing_folder = '/content/drive/My Drive/preproc_data'
log_root = '/content/drive/My Drive/logdir'

# Pre-process settings
standardize = False
normalize = True
equalize = False
clahe = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_epochs = 20000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
