# TODO rename

import pandas as pd
import csv
import sys
import os
# from timebudget import timebudget
# timebudget.set_quiet() # don't show measurements as they happen
# timebudget.report_at_exit() # Generate report when the program exits

from abstract_classes import Dataset, Classifier

class PredatorDataset(Dataset):
	def __init__(self, data_dir, dataset_indicator, dataset_type):
		csv.field_size_limit(sys.maxsize) # there are very large rows in our dataset
		self.ds = pd.read_csv(
			os.path.join(data_dir, '%s-%s.csv' % (dataset_indicator, dataset_type)),
			# sep = '\t',
			# encoding='latin-1',
			usecols=['label', 'chatName', 'segment'],
			converters={'label': str, 'chatName': str, 'segment': str} # https://stackoverflow.com/a/52156896/1974674
		)

	def get_samples(self):
		return ((self.ds.segment[i], self.ds.label[i], self.ds.chatName[i])
			for i in range(self.size))

	@property
	def size(self) -> int:
		return len(self.ds.segment)

	def __getitem__(self, subscript):
		return list(self.ds.values[subscript][0])



from tensorflow_examples.lite.model_maker.core.task import model_util
from official.nlp.data import classifier_data_lib # best library
import json
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier
from util import loadPredatorDatasetFromCSV
from argparse import Namespace
import setup_tensorflow
import tensorflow as tf


def getTFLiteModel(args):
	# get model_spec
	spec = model_spec.get(args.model_indicator)
	spec.seq_len = args.seq_len # TODO try 1000 and see what happens # DONE you get an error
	# spec.model_dir = args.run_dir

	# get dataset
	test_data = loadPredatorDatasetFromCSV(
		args.data_dir, args.dataset_indicator, "test", spec)

	# get model
	model = text_classifier.create(
		test_data,
		model_spec=spec,
		do_train=False,
	)
	return model

class TFLiteClassifier(Classifier):
	def __init__(self, run_id, model_version, base_dir="./", model=None):
		f = open("%s/resources/%s/run_settings.json" % (base_dir, run_id))
		args = Namespace(**json.load(f))
		tflite_filepath = "%s/%s/model.tflite" % (args.run_dir, model_version)

		# use preprocessed model if exists
		self.model = model if model is not None else getTFLiteModel(args)
		self.model_spec = self.model.model_spec
		self.lite_runner = model_util.get_lite_runner(
			tflite_filepath, self.model_spec)
		self.index_to_label = self.model.index_to_label


	# @timebudget  # Record how long this function takes
	def predict(self, text): # weird function but it works
		# with timebudget("preprocessing prediction"):
		# wrap text as an InputExample
		example = classifier_data_lib.InputExample(None, text, None, None)

		# initializes the tokenizer if not built already
		self.model_spec.build()

		# Convert InputExample to the input feature which BERT models expect
		feature = classifier_data_lib.convert_single_example(
			0, example, [],
			self.model_spec.seq_len, self.model_spec.tokenizer
		)

		# adapt feature to what our tflite model expects as input
		# see also: self.model.model_spec.select_data_from_record
		converted_feature = {
			'input_word_ids': tf.constant([feature.input_ids], dtype=tf.int32),
			'input_mask': tf.constant([feature.input_mask], dtype=tf.int32),
			'input_type_ids': tf.constant([feature.segment_ids], dtype=tf.int32)
		}

		probabilities = self.lite_runner.run(converted_feature)[0]
		probabilities = [float(f) for f in probabilities] # convert to float
		# print(probabilities)
		# print(type(probabilities))
		# prediction_index = np.argmax(probabilities)
		# print("prediction: %s – %s" % (self.index_to_label[prediction_index], probabilities))
		return probabilities

	# expected input for lite_runner.run:
	# 	{
	# 		input_word_ids: TensorShape([1, 128], dtype=int32),
	# 		input_mask: TensorShape([1, 128], dtype=int32),
	# 		input_type_ids: TensorShape([1, 128], dtype=int32)
	# 	}

	def predict_multiple(self, texts):
		return [self.predict(text) for text in texts]



from flair.models import TextClassifier
from flair.data import Sentence
import numpy as np

class FlairClassifier(Classifier):
	def __init__(self, run_dir, model_version):
		self.index_to_label = ["non-predator", "predator"]
		model_path = os.path.join(run_dir, model_version, "final-model.pt")
		self.textClassifier = TextClassifier.load(model_path)

	##
	## classifies a list of texts as positive or negative
	##
	## :param      texts: list of texts
	## :type       texts:  list of strings
	##
	## :returns:   [negative probabiility, positive probability] for each text
	## :rtype:     list of numpy arrays
	##
	def predict_multiple(self, texts):
		sentences = [Sentence(text) for text in texts]
		# with timebudget("flair prediction"):
		self.textClassifier.predict(sentences) # annotates the sentences with predictions
		first_labels = [s.get_labels()[0] for s in sentences]
		ret = [self.label_to_class_probabilities(label) for label in first_labels]
		assert len(ret) == len(texts), \
			"return list has different length to text list (%s vs %s respectively)" % (len(ret), len(texts))
		return ret

	##
	## classifies a text as positive or negative
	##
	## :param      text:  The text
	## :type       text:  string
	##
	## :returns:   [negative probabiility, positive probability]
	## :rtype:     numpy array
	##
	def predict(self, text):
		return self.predict_multiple([text])[0]

	##
	## converts a flair label prediction to a list of class probabilities
	##
	## :param      label:       The label
	## :type       label:       flair label of predator or non-predator with probability
	##
	## :returns:   [non-predator probability, predator probability]
	## :rtype:     numpy array
	##
	## :raises     ValueError:  the label given is neither predator nor non-predator
	##
	def label_to_class_probabilities(self, label):
		if label.value == "predator":
			return np.array([1-label.score, label.score])
		elif label.value == "non-predator":
			return np.array([label.score, 1-label.score])
		else:
			raise ValueError("undefined label ``%s''" % label)
