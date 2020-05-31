from metrics import iou_metric
import numpy as np

class ClassificationReport:
	def __init__(self, ix):
		ix = self.ix

	def generate(self):
		return iou_metric(np.squeeze(Y_train[self.ix]), np.squeeze(preds_train_t[self.ix]), print_table=True)