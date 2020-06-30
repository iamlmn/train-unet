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
from plots import *
try:
  os.mkdir('mem_cache')
except:
  pass
cachedir = 'mem_cache'
mem = Memory(cachedir)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../data/U_NET/train/'
TEST_PATH = '../data/U_NET/validation/'
MODEL_OUTPUT_PATH = 'py_model.h5'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore')
seed = 42
random.seed = seed
np.random.seed = seed


class TrainUnet:
  def __init__(self, train_path, test_path,  img_height, img_width, img_channels, model_output_path):
    self.train_path = train_path
    self.test_path = test_path
    self.img_height = img_height
    self.img_width = img_width
    self.img_channels = img_channels
    self.model_output_path = model_output_path
    self.train_ids = next(os.walk(self.train_path))[1]
    self.test_ids = next(os.walk(self.test_path))[1]
    

  # @mem.cache # Avoid computing the same thing twice
  def resize_training_sets(self, combine_masks = True):
    print('Getting and resizing training images ... ')
    X_train = np.zeros((len(self.train_ids), self.img_height, self.img_width, self.img_channels), dtype=np.uint8)
    Y_train = np.zeros((len(self.train_ids), self.img_height, self.img_width, 1), dtype=np.bool)
            
    # Re-sizing our training images to 128 x 128
    # Note sys.stdout prints info that can be cleared unlike print.
    # Using TQDM allows us to create progress bars
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(self.train_ids), total=len(self.train_ids)):
        path = self.train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:self.img_channels]
        img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((self.img_height, self.img_width, 1), dtype=np.bool)
        
        if combine_masks:
          # Now we take all masks associated with that image and combine them into one single mask
          for mask_file in next(os.walk(path + '/masks/'))[2]:
              mask_ = imread(path + '/masks/' + mask_file)
              mask_ = np.expand_dims(resize(mask_, (self.img_height, self.img_width), mode='constant', 
                                            preserve_range=True), axis=-1)
              mask = np.maximum(mask, mask_)
        # Y_train is now our single mask associated with our image
        Y_train[n] = mask
    self.X_train = X_train
    self.Y_train = Y_train
    return X_train, Y_train


  # @mem.cache # Avoid computing the same thing twice
  def resize_target_sets(self):
    # Get and resize test images
    X_test = np.zeros((len(self.test_ids), self.img_height, self.img_width, self.img_channels), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()

    # Here we resize our test images
    for n, id_ in tqdm(enumerate(self.test_ids), total=len(self.test_ids)):
        path = self.test_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:self.img_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (self.img_height, self.img_width), mode='constant', preserve_range=True)
        X_test[n] = img

    self.X_test = X_test
    return X_test

  def train_illustrate(self):
    return train_illustrate(self.X_train, self.Y_train)


  def train_model(self):
    '''
    Train a unet model
    '''
    from model.unet import Unet
    # prepare unet modela
    _unet= Unet(self.img_height, self.img_width, self.img_channels)
    inputs, outputs = _unet.model()
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
    print(model.summary())

    ### FIT
    # Initialize our callbacks
    model_path = self.model_output_path # TODO change  to ->model_output_path
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

  def load_and_predict(self):
    # model_path = unet_test.train_model(img_height, img_width, img_channels)
    # pred
    # Predict on training and validation data
    # Note our use of mean_iou metri
    model = load_model(self.model_output_path, 
                       custom_objects={'my_iou_metric': my_iou_metric})

    # the first 90% was used for training
    preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    # the last 10% used as validation
    preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    # test set predictions
    preds_test = model.predict(X_test, verbose=1)
    self.preds_train = preds_train
    self.preds_val  = preds_val
    self.preds_test = preds_test
    return preds_train, preds_val, preds_test

  def plot_random_comparisons(self, preds_train_t, preds_val_t, preds_test, _save = True):
    # Our original training image
    random_prediction_plots_on_training_set(self.X_train, self.Y_train, preds_train_t, save = _save)
    # validation test predicion comparison
    random_prediction_plots_on_validation_set(self.X_train, preds_val_t, save = _save)
    # test set predictions
    random_prediction_plots_on_test_set(self.X_test, preds_test, save = _save)
    return preds_train, preds_val, preds_test

  def classification_report(self, ix):
    from reports import ClassificationReport
    cf = ClassificationReport(ix)
    cf_mean = cf.generate( Y_train, preds_train_t)
    print("Classification report mean : {}".format(cf_mean))


if __name__ == '__main__':
  unet_test = TrainUnet(TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, MODEL_OUTPUT_PATH)
  X_train, Y_train = unet_test.resize_training_sets(combine_masks = True)
  X_test = unet_test.resize_target_sets()
  print('Inputs prepared, Ready for modelling!')
  unet_test.train_illustrate() # illustrate every 10th training and masked images
  #model_path = unet_test.train_model()
  # pred
  preds_train, preds_val, preds_test = unet_test.load_and_predict()
  # Threshold predictions
  preds_train_t = (preds_train > 0.5).astype(np.uint8)
  preds_val_t = (preds_val > 0.5).astype(np.uint8)
  ## Plots
  unet_test.plot_random_comparisons(preds_train_t, preds_val_t, preds_test, _save = True)
  ix = 25 # temp
  unet_test.classification_report(ix)