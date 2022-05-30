from util import getTimestamp
from eval_util import getEvalArgs, getSegments, getWarningLatency, MasterClassifier, isNonemptyMsg, Score, get_speed, breakUpChat
from pathlib import Path
import json
import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
	"--skepticism",
	dest='skepticism',
	help="skepticism level of the master classifier",
	type=int,
	default=5
)
parser.add_argument(
	"--window_size",
	dest='window_size',
	help="window_size used for prediction annotation",
	type=int,
	default=50
)
parser.add_argument(
	"--eval_mode",
	dest='eval_mode',
	help="whether to evaluate with complete chats or segments of complete chats. This applies to the complete positive chats from CC2.",
	choices=["full", "full_fast"],
	default="full_fast"
)
args = getEvalArgs(parser)

# get the datapack
out_dir = "%s/%s/message_based_eval/" % (args.run_dir, args.model_version)
datapackPath = "%s/annotated-datapack-%s-test-eval_mode-%s--window_size-%s.json" % (out_dir, args.dataset_indicator, args.eval_mode, args.window_size)
with open(datapackPath, "r") as file: datapack = json.load(file)
chatNames = sorted(list(datapack["chats"].keys()))
# information about datapacks can be found in the chat-visualizer repo

# if in full mode, only evaluate on complete positive chats.
# because negative chats are always just segments, full mode is the same as
# segment mode for them.
chatNames = [name for name in chatNames
	if datapack["chats"][name]["className"] == "predator"]

def evaluateForSkepticism(skepticism):
	latencies = []
	for chatName in chatNames:
		chat = datapack["chats"][chatName]
		is_positive = chat["className"] == "predator"

		assert is_positive, "chat in datapack was negative"

		# get latency
		nonempty_messages = [ct for ct in chat["content"] if isNonemptyMsg(ct)]
		latency = getWarningLatency(nonempty_messages, skepticism)
		latencies.append(latency if latency else -1)
		if not latency: print("!!! the chat %s was not detected as positive" % chatName)
		# we only count the latency of true positives warnings

	return latencies

# for skepticism in tqdm(range(1, 10+1), desc="Evaluating for skepticism"):
print("For skepticism = %s" % args.skepticism)
latencies = evaluateForSkepticism(args.skepticism)

print("len(latencies) = %s" % len(latencies))
print("len(chatNames) = %s" % len(chatNames))
print("len(latencies)/len(chatNames) = %s" % (len(latencies)/len(chatNames)))

print("np.median(latencies) = %s" % np.median(latencies))

# save warning latencies for args.skepticism
latency_dir = os.path.join(out_dir, "full_length_latencies/")
Path(latency_dir).mkdir(parents=True, exist_ok=True)
latencyFile = "%s/latencies__skepticism-%s.txt" % (latency_dir, args.skepticism)
with open(latencyFile, "w") as file:
	for val in latencies: file.write("%s\n" % str(val))
