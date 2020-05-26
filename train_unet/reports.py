from metrics import iou_metric
import numpy as np

class ClassificationReport(self):
	def __init__(self):
		ix = self.ix

	def generate(self):
		return iou_metric(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]), print_table=True)