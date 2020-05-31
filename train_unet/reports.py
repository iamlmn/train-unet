from metrics import iou_metric
import numpy as np

class ClassificationReport:
	def __init__(self, ix):
		self.ix = ix
		

	def generate(self, Y_train, preds):
		return iou_metric(np.squeeze(Y_train[self.ix]), np.squeeze(preds[self.ix]), print_table=True)