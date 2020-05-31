# first line: 50
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
