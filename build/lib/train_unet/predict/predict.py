class predict(self):
	def __init__(self):
		self.metrics = None
		train_split = 90/100
		validation_split = 1 - train_split
		# Predict on training and validation data
		# Note our use of mean_iou metri
		model = load_model('/home/deeplearningcv/DeepLearningCV/Trained Models/nuclei_finder_unet_2.h5', # output_model_path
		                   custom_objects={'my_iou_metric': my_iou_metric})

		# the first 90% was used for training
		preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

		# the last 10% used as validation
		preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

		#preds_test = model.predict(X_test, verbose=1)

		# Threshold predictions
		preds_train_t = (preds_train > 0.5).astype(np.uint8)
		preds_val_t = (preds_val > 0.5).astype(np.uint8)

		return preds_train_t, preds_val_t