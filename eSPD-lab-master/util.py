import sys
import os
from datetime import datetime, timezone
import argparse
from argparse import Namespace


import json
from pathlib import Path

def getTimestamp():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
def getUNIXTimestamp():
	return int(datetime.now().replace(tzinfo=timezone.utc).timestamp()*1000)


# to access like bunch.args instead of bunch["args"]

def getTrainArgs():
	parser = argparse.ArgumentParser(description='Train a model')
	parser.add_argument(
		"--model",
		dest='model_indicator',
		# choices=["bert_classifier", "mobilebert_classifier"],
		help="the kind of model to use, e.g. bert_classifier or mobilebert_classifier if using modelmaker and bert-large-uncased if using flair",
		required=True
	)
	parser.add_argument(
		"--seq_len",
		dest='seq_len',
		type=int,
		help="maximum sequence length for input text",
		required=True
	)
	parser.add_argument(
		"--dataset",
		dest='dataset_indicator',
		help="which dataset to train on",
		required=True
	)
	parser.add_argument(
		"--project",
		dest='project',
		choices=["modelmaker", "flair"],
		help="which project this model belongs to",
		required=True
	)
	args = parser.parse_args()

	run_settings = vars(args)

	# used to identify the specific training run for this model configuration
	run_settings["run_id"] = '%s__%s_on_%s_with_seq-len-%s' \
		% (getTimestamp(), args.model_indicator, args.dataset_indicator, args.seq_len)

	# directory of the dataset
	run_settings["data_dir"] = os.path.join(
		"datasets/%s/" % args.dataset_indicator)

	# directory where the model files should be saved
	run_settings["run_dir"] = os.path.join(
		"resources/%s/" % run_settings["run_id"])

	# create folder if it doesn't exist already
	Path(run_settings["run_dir"]).mkdir(parents=True, exist_ok=True)
	# dump run settings to file
	with open("%s/run_settings.json" % run_settings["run_dir"], "w") as file:
		json.dump(run_settings, file)

	print("\n---            Run settings            ---")
	for key, val in run_settings.items(): print("%20s: %s" % (key, val))

	return Namespace(**run_settings)



def loadPredatorDatasetFromCSV(data_dir, dataset_indicator, dataset_type, spec):
	import setup_tensorflow
	from tflite_model_maker.text_classifier import DataLoader
	import csv

	csv.field_size_limit(sys.maxsize) # there are very large rows in our dataset

	# VTPAN
	return DataLoader.from_csv(
		filename=os.path.join(os.path.join(data_dir, '%s-%s.csv' % (
			dataset_indicator, dataset_type
		))),
		text_column='segment',
		label_column='label',
		model_spec=spec,
		# delimiter='\t',
		is_training=False
	)
