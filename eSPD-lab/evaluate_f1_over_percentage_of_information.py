import os
from pathlib import Path

from concrete_classes import PredatorDataset, getTFLiteModel
from eval_util import getEvalArgs, Score, iterate_multi_threaded, get_classifier
from util import getTimestamp

# globals
args = getEvalArgs()

# get the dataset
test_data = PredatorDataset(args.data_dir, args.dataset_indicator, "test")
samples = list(test_data.get_samples())
print("test_data.size = %s" % test_data.size)

# if we use model maker, cache the model so we don't have to load it for each thread
model = getTFLiteModel(args) if args.project == "modelmaker" else None

FULL_MODE = False  # takes too long because BERT is slow
percentages = range(1, 101) if FULL_MODE else list(range(10, 110, 10))  # 10,20,…,100


def getPrefix(text, percentage):
    prefixLen = int(percentage / 100 * len(text))  # at most len(text)

    # Whole words only: if cut were in a word, cut before the word instead
    # Example where "|" is the cut:
    # 1. baz fo|o bar
    # 2. baz f|oo bar
    # 3. baz |foo bar
    # 4. baz| foo bar
    while prefixLen < len(text) and prefixLen > 0:
        # if zero, the prefix is the emptystring
        indexAfterCut = prefixLen
        if text[indexAfterCut].isspace(): break
        prefixLen -= 1
    # prefixLen+=1; # this would put the cut after the word

    return text[:prefixLen]  # might be ""


def getScore(dataset_slice, step):
    # get the classifier for the thread, initialize with existing model info
    classifier = get_classifier(args.project, args.run_dir, args.run_id, args.model_version, model=model)

    # by calculating the scores seperately for each thread, we don't need locks
    percentage_scores = [Score() for i in percentages]  # i=1,…,100

    for text, label, name in samples[dataset_slice]:
        step()
        # with timebudget("all %s percentages for a single sample" % len(percentages)):

        for i, score in enumerate(percentage_scores):
            percentage = percentages[i]  # percentage = 1,2,…,100 or 1,10,…,100

            prefix = getPrefix(text, percentage)

            # happens when len(text)<100
            # for example for the segment 066e8849c064badb9668c3fa39227195
            # it makes sense to skip these because we can't evaluate such prefixes
            if len(prefix) == 0: continue

            predicted_label = classifier.predict_label(prefix)
            predicted_positive = predicted_label == "predator"
            is_positive = label == "predator"

            score.add_prediction(predicted_positive, is_positive)

        # print("percentage = %s" % percentage)
        # print("prefix = %s" % prefix)
        # print("predicted_label = %s" % predicted_label)
        # print("ground truth label = %s" % label)

    return percentage_scores


future_percentage_scores = iterate_multi_threaded(test_data.size, args.threads, getScore)
# add all returned percentage scores
percentage_scores = [sum(scores_of_percentage)
                     for scores_of_percentage in zip(*future_percentage_scores)]

# print and save to file

# scores
f = [score.f1 for score in percentage_scores]
p = [score.precision for score in percentage_scores]
r = [score.recall for score in percentage_scores]

print("all done\n")
print("Score for 100% of information:")
print(percentage_scores[len(percentage_scores) - 1])

print("\nNumber of samples for each percentage:")
number_of_samples = [s.number_of_samples for s in percentage_scores]
print(number_of_samples)

# expected to monotonously increase
# monotonously_increasing = all(x<=y for x, y in zip(number_of_samples, number_of_samples[1:]))
# if not monotonously_increasing: print("\n---\nwarning: number of samples is not monotonously increasing!\n---\n")

print("\nF1 for each percentage:\n%s" % f)

# save scores to files

eval_dir = os.path.join(args.run_dir, "%s/percentage_of_information_eval/" % args.model_version)
Path(eval_dir).mkdir(parents=True, exist_ok=True)  # might not exist yet

with open("%s/info.txt" % eval_dir, "w") as file:
    file.write("Generated this percentage-of-information evaluation at %s\n" % getTimestamp())

for name, scores in ({"f1": f, "precision": p, "recall": r}).items():
    with open("%s/%s.txt" % (eval_dir, name), "w") as file:
        # files have 100 lines: the values of the metrics at each percentage
        for val in scores: file.write("%s\n" % str(val))
