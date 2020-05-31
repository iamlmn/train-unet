import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images

# Illustrate the train images and masks
def train_illustrate(X_train, Y_train, save = True):
	plt.figure(figsize=(20,16))
	x, y = 12,4
	for i in range(y):  
	    for j in range(x):
	        plt.subplot(y*2, x, i*2*x+j+1)
	        pos = i*120 + j*10
	        plt.imshow(X_train[pos])
	        plt.title('Image #{}'.format(pos))
	        plt.axis('off')
	        plt.subplot(y*2, x, (i*2+1)*x+j+1)
	        
	        #We display the associated mask we just generated above with the training image
	        plt.imshow(np.squeeze(Y_train[pos]))
	        plt.title('Mask #{}'.format(pos))
	        plt.axis('off')
	# TODO NEED TO CONVERT TO FIGURE SAVE
	if save:
		plt.savefig('trains.png')
	else:
		plt.show()


def random_prediction_plots_on_training_set(X_train, Y_train, preds_train_t, save = True):
	# TODO Eithr take a lis of indexes or generate random indexs and save plots for them
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
	if save:
		plt.savefig('rnd-prediction-training-set.png')
	else:
		plt.show()

	# TODO NNEED TO SAVE INSTEAD OF SHOW


def random_prediction_plots_on_validation_set(X_train, preds_val_t, save = True):
	# Ploting our predicted masks
	ix = random.randint(602, 668)
	plt.figure(figsize=(20,20))

	# Our original training image
	plt.subplot(121)
	imshow(X_train[ix])
	plt.title("Image")

	# The mask our U-Net model predicts
	plt.subplot(122)
	ix = ix - 603
	imshow(np.squeeze(preds_val_t[ix] > 0.5))
	plt.title("Predictions")
	if save:
		plt.savefig('rnd-validationset-prediction-set.png')
	else:
		plt.show()


def random_prediction_plots_on_test_set(X_test, preds_test, save = True):
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
	if save:
		plt.savefig('rnd-prediction-test-set.png')
	else:
		plt.show()