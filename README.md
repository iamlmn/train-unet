# Train Unet

Aim is to create a low-code easy to use python library for training CNN models using Unet architecture and custom metrics like IoU (Intersection over Union) at ease for semantic segmentation of medical images.

## Appliccations in Medical Imaging
- Many medical applications necessitates finding and accurately labeling things found in medical scans.
- This is often done using advanced software to assist medical technicians and doctos. However, this task still requires human intervention and such as, can be tedious,slow, expensive and prone to human error.
- There's huge initiative for use Computer Vision and Deep Learning to automate many of these tasks. 

Analyzing medical scans such as
 - CAT scans
 - X-rays
 - Ultra sound 
 - PET
 - NMR

***

<!-- ## What is image segmentation?
 The goal of segmentation is seperate different parts of image, into sensible coherent parts where we are doing a pixel wise predictions.

 Two types of segmentation:
 1. Instance segmentation.
 	Pixel classificcation combined with classifying object entities e.g. Seperate persons, cars. etc.
 2. Semantic segmentation.
 	> Pixel classifications based on defined classes e.g roads, persons, cars, trees etc.
 -->
***

## U-net  - Image segmentation using CNN's.

U-net has become very popular end-to-end encoder-decoder network for Semantic segmentation.

It has a unique Up-down archtiecture which has a contracting path and an expansive path. 

U-NET archtiectURE: 
![alt text][unet1]

[unet1]: https://github.com/iamlmn/train-unet/blob/master/assets/unet.png "Unet Arch"

U-NET design: 
![alt text][unet2]

[unet2]: https://github.com/iamlmn/train-unet/blob/master/assets/unet-design.png "Unet Design"


## Intersectionn over union metrics
IoU is basically a measure of overlap.

Iou = Size_of_union / Size of Intersection

 - Typically IoU over 0.5 is acceptable.
 - higher the IoU better the prediction.

***

## Installation and Usage


```
git clone https://github.com/iamlmn/train-unet.git
cd train-unet
python3 main.py
```
Can be used like a package like below as well

Requires Training sets & test sets are expected to be in the below format.

Input struct on single train and test set: 
![alt text][unet2]

[unet2]: https://github.com/iamlmn/train-unet/blob/master/assets/input-struct.png "Input Struct"

***

### Sameple dataset is in data folder. Finding the nuclei in Divergent images. #Spot Nuclei. Speed Cures. (The Kaggle Data Science bowl 2018 Challenge).


##### use it from Python:
```python
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../data/U_NET/train/'
TEST_PATH = '../data/U_NET/validation/'
MODEL_OUTPUT_PATH = 'py_model.h5'
from train_unet import TrainUnet
unet_test = TrainUnet(TRAIN_PATH, TEST_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, MODEL_OUTPUT_PATH) # Create Unet object
X_train, Y_train = unet_test.resize_training_sets(combine_masks = True) # prep training data
X_test = unet_test.resize_target_sets() # prep Target sets
unet_test.train_illustrate() # illustrate every 10th training and masked images
model_path = unet_test.train_model() # Traing
preds_train, preds_val, preds_test = unet_test.load_and_predict() # predict
unet_test.plot_random_comparisons(preds_train_t, preds_val_t, preds_test, _save = True) # comparison plots on random images
ix = 25 # temp
unet_test.classification_report(ix)

```

***

TODOs and completed work : 
- [x] Base module class
- [ ] Generalization
- [ ] Brainstorm ideas
- [ ] 3d Segmentations?
- [ ] Unit tests

***

[Repo]: https://github.com/iamlmn/train-unet