import setup_tensorflow
from eval_util import get_classifier, getEvalArgs
import argparse
from timebudget import timebudget
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Estimate the inference latency of a model')
args = getEvalArgs(parser)

classifier = get_classifier(args.project, args.run_dir, args.run_id, args.model_version)

for i in tqdm(range(1,21)):
	with timebudget("inference"):
		classifier.predict("foobar "*i)



