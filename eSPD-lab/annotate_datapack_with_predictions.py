import argparse
import json
import os
from pathlib import Path

from concrete_classes import getTFLiteModel
from eval_util import getEvalArgs, iterate_multi_threaded, contentToString, isNonemptyMsg, breakUpChat, get_classifier, \
    MasterClassifier
from util import getUNIXTimestamp

parser = argparse.ArgumentParser(description='Evaluate a model')
parser.add_argument(
    "--eval_mode",
    dest='eval_mode',
    help="whether to evaluate with complete predator chats or (predator and non-predator) segments of chats. *_fast modes are recommended. They speeds up the respecitve mode by only analyzing until the first warning is raised.",
    choices=["segments", "segments_fast", "full", "full_fast"],
    required=False
)
parser.add_argument(
    "--window_size",
    dest='window_size',
    help="we look at the last `window_size` messages during classification",
    type=int,
    default=50
)

args = getEvalArgs(parser)

# get the datapack
datapackPath = os.path.join(args.data_dir, 'datapack-%s-test.json' % args.dataset_indicator)
with open(datapackPath, "r") as file:
    datapack = json.load(file)

chatNames = sorted(list(datapack["chats"].keys()))
# information about datapacks can be found in the chat-visualizer repo

# if in full mode, only evaluate on complete positive chats.
# because negative chats are always just segments, full mode is the same as
# segment mode for them.
if args.eval_mode.startswith("full"):
    chatNames = [name for name in chatNames if datapack["chats"][name]["className"] == "predator"]

# if we use model maker, cache the model so we don't have to load it for each thread
model = getTFLiteModel(args) if args.project == "modelmaker" else None


def annotateExtract(extract, classifier):
    # In fast mode, we only annotate until the master classifier with maximum
    # skepticism 10 raises a warning. Later in evaluation, if we annotate this
    # way, there will always be enough annotated messages to evaluate for all
    # skepticisms
    mc = MasterClassifier(10)

    nonempty_messages = [ct for ct in extract if isNonemptyMsg(ct)]

    for i, msg in enumerate(nonempty_messages):
        # last args.window_size messages up to message with index i
        window = nonempty_messages[max(0, i + 1 - args.window_size):i + 1]

        text = contentToString(window)
        prediction = classifier.predict_label_probability(text, "predator")

        # Annotate the message. This modifies the referenced message
        # that is then also modified in our datapack.
        msg["prediction"] = prediction

        # in fast mode
        mc_raised_warning = mc.add_prediction(prediction >= 0.5)

        if mc_raised_warning and args.eval_mode.endswith("_fast"):
            return  # stop annotating when warning is raised


def annotateSlice(dataset_slice, step):
    # get the classifier for the thread, initialize with existing model info
    classifier = get_classifier(args.project, args.run_dir, args.run_id, args.model_version, model=model)

    for chatName in chatNames[dataset_slice]:
        for extract in breakUpChat(datapack["chats"][chatName], args):
            # extract -> segment or full chat
            annotateExtract(extract, classifier)
        step()


print("Starting work on %s chats (which might have multiple segments each)" % len(chatNames))
iterate_multi_threaded(len(chatNames), args.threads, annotateSlice)

print("all done\n")

# dump annotated datapack

suffix = "eval_mode-%s--window_size-%s" % (args.eval_mode, args.window_size)

datapack["datapackID"] += "--" + suffix

if datapack["description"] == None:
    datapack["description"] = ""

datapack["description"] += "Annotated with predictions by %s (seq_len=%s, %s)" % (
    args.run_id, args.seq_len, args.model_version)

eval_dir = os.path.join(args.run_dir, "%s/message_based_eval/" % args.model_version)

Path(eval_dir).mkdir(parents=True, exist_ok=True)  # might not exist yet

datapack["generatedAtTime"] = getUNIXTimestamp()

# outFile = eval_dir + "annotated-datapack-%s-test-%s.json" % ( # should be --%s.json
outFile = eval_dir + "annotated-datapack-%s-test-%s.json" % (
    args.dataset_indicator, suffix)

with open(outFile, "w") as file:
    json.dump(datapack, file, indent=4)
