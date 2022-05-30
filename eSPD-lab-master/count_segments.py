import setup_tensorflow
from util import getTimestamp
from concrete_classes import TFLiteClassifier
from eval_util import getEvalArgs, iterate_multi_threaded, getSegments, contentToString, isNonemptyMsg, isGood, breakUpChat
from pathlib import Path
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
	"--eval_mode",
	dest='eval_mode',
	help="whether to evaluate with complete chats or segments of complete chats. This applies to the complete positive chats from CC2.",
	choices=["segments", "full"],
	default="segments",
	required=False
)
args = getEvalArgs(parser)

# get the datapack
datapackPath = os.path.join(args.data_dir,
	'datapack-%s-test.json' % args.dataset_indicator)
with open(datapackPath, "r") as file:
	datapack = json.load(file)
chatNames = sorted(list(datapack["chats"].keys()))
# information about datapacks can be found in the chat-visualizer repo

if args.eval_mode == "full":
	chatNames = [name for name in chatNames
		if datapack["chats"][name]["className"] == "positive"]

WINDOW_SIZE = 50

def annotateSlice(dataset_slice, step):
	count = 0
	for chatName in chatNames[dataset_slice]:
		step()
		for content in breakUpChat(datapack["chats"][chatName], args):
			count += 1
	return count

# len(chatNames)
counts = iterate_multi_threaded(len(chatNames), args.threads, annotateSlice)
print(counts)
print(sum(counts))

print("all done\n")

# # dump annotated datapack

# datapack["description"] += "Annotated with predictions by %s (seq_len=%s, %s)" % (args.model_indicator, args.seq_len, args.model_version)

# eval_dir = os.path.join(args.run_dir, "%s/message_based_eval/" % args.model_version)
# Path(eval_dir).mkdir(parents=True, exist_ok=True) # might not exist yet

# with open("%s/info.txt" % eval_dir, "w") as file:
# 	file.write("Generated at %s\n" % getTimestamp())

# outFile = eval_dir + "annotated-datapack-%s-test.json" % args.dataset_indicator
# with open(outFile, "w") as file: json.dump(datapack, file, indent=4)
