"""
Usage:
    main.py [--epochs=<path--yaml-config-file>] [--default_cell_timeout=<time in second>]
    main.py -h|--help
    main.py -v|--version
Options:
    -h --help  Show this screen.
    -v --version  Show version.
    --yamlspec path of yaml file referring to indepth analysis report and plot control
    --default_cell_timeout: Default cell time out for executing in jupyter notebook 
"""
import os
import sys
import random
import warnings #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from joblib import Memory
from metrics import my_iou_metric
try:
  os.mkdir('mem_cache')
except:
  pass
cachedir = 'mem_cache'
mem = Memory(cachedir)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../assets/U_NET/train/'
TEST_PATH = '../assets/U_NET/validation/'
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

model_output_path = 'py_model.h5'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
@mem.cache # Avoid computing the same thing twice
def resize_training_sets(train_ids):
  print('Getting and resizing training images ... ')
  X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
  Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
          
  # Re-sizing our training images to 128 x 128
  # Note sys.stdout prints info that can be cleared unlike print.
  # Using TQDM allows us to create progress bars
  sys.stdout.flush()
  for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
      path = TRAIN_PATH + id_
      img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
      img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_train[n] = img
      mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
      
      # Now we take all masks associated with that image and combine them into one single mask
      for mask_file in next(os.walk(path + '/masks/'))[2]:
          mask_ = imread(path + '/masks/' + mask_file)
          mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                        preserve_range=True), axis=-1)
          mask = np.maximum(mask, mask_)
      # Y_train is now our single mask associated with our image
      Y_train[n] = mask

  return X_train, Y_train

X_train, Y_train = resize_training_sets(train_ids)

@mem.cache # Avoid computing the same thing twice
def resize_target_sets(test_ids):
  # Get and resize test images
  X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
  sizes_test = []
  print('Getting and resizing test images ... ')
  sys.stdout.flush()

  # Here we resize our test images
  for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
      path = TEST_PATH + id_
      img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
      sizes_test.append([img.shape[0], img.shape[1]])
      img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
      X_test[n] = img
  return X_test

X_test = resize_target_sets(test_ids)

print('Prepared, Ready for modelling!')
from plots import *
train_illustrate(X_train, Y_train)
#### TRAIN
def train_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
  from model.unet import Unet
  # prepare unet modela
  _unet= Unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
  inputs, outputs = _unet.model()
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
  print(model.summary())

  ### FIT

  # Initialize our callbacks
  model_path = model_output_path # TODO change  to ->model_output_path
  checkpoint = ModelCheckpoint(model_path,
                               monitor="val_loss",
                               mode="min",
                               save_best_only = True,
                               verbose=1)

  print('Crossedd checkpoint')
  earlystop = EarlyStopping(monitor = 'val_loss', 
                            min_delta = 0, 
                            patience = 5,
                            verbose = 1)
                            # restore_best_weights = True)

  # Fit our model 
  results = model.fit(X_train, Y_train, validation_split=0.1,
                      batch_size=16, epochs=10, 
                      callbacks=[earlystop, checkpoint])
  return model_path

model_path = train_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
# pred
# Predict on training and validation data
# Note our use of mean_iou metri
model = load_model(model_path, 
                   custom_objects={'my_iou_metric': my_iou_metric})

# the first 90% was used for training
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

# the last 10% used as validation
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

#preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

## Plots
# Ploting our predicted masks
ix = random.randint(0, 602)
plt.figure(figsize=(20,20))

# Our original training image
plt.subplot(131)
imshow(X_train[ix])
plt.title("Image")

# Our original combined mask  
plt.subplot(132)
imshow(np.squeeze(Y_train[ix]))
plt.title("Mask")

# The mask our U-Net model predicts
plt.subplot(133)
imshow(np.squeeze(preds_train_t[ix] > 0.5))
plt.title("Predictions")
plt.savefig('validation_prediction.png')




# test set predictions
preds_test = model.predict(X_test, verbose=1)
# Ploting our predicted masks
ix = random.randint(0, 50)
plt.figure(figsize=(20,20))

# Our original training image
plt.subplot(131)
imshow(X_test[ix])
plt.title("TEst")

plt.subplot(133)
imshow(np.squeeze(preds_test[ix] > 0.5))
plt.title("Predictions")
plt.savefig('Test_prediction.png')

ix = 25 # temp
from metrics import ClassificationReport
cf = ClassificationReport(ix)
cf.generate()
print("Classification report mean : {}".format())