from abc import ABC, abstractmethod
import numpy as np

# Adapters to be able to write generic evaluation code
# for Flair and TFLite Models

class Classifier(ABC):

	@abstractmethod
	def predict(self, sample):
		"""Predict the label of a sample"""
		pass

	@abstractmethod
	def predict_multiple(self, samples):
		"""Predict the label of a sample"""
		pass

	# self.index_to_label

	def predict_label(self, text):
		"""predicts the label string of the text"""
		label_index = np.argmax(self.predict(text))
		return self.index_to_label[label_index]

	def predict_label_probability(self, text, label):
		"""returns the probability that a text has a certain label"""
		label_index = self.index_to_label.index(label)
		return self.predict(text)[label_index]


# class Sample(ABC):
# 	@property
# 	@abstractmethod
# 	def label(self):
# 		"""return some label (string, int, …)"""
# 		pass

# 	@property
# 	@abstractmethod
# 	def value(self):
# 		"""return the value of a sample (e.g. the original sentence string) """
# 		pass


class Dataset(ABC):
	@abstractmethod
	def get_samples(self):
		"""Generator for the content of a Dataset"""
		pass

	@property
	@abstractmethod
	def size(self) -> int:
		"""Number of samples in the Dataset"""
		pass
