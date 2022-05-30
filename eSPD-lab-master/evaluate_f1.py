import setup_tensorflow
from util import getTimestamp
from eval_util import getEvalArgs, Score, iterate_multi_threaded, get_classifier
from concrete_classes import PredatorDataset, getTFLiteModel

# globals
args = getEvalArgs()

# get the dataset
test_data = PredatorDataset(args.data_dir, args.dataset_indicator, "test")
samples = list(test_data.get_samples())
print("test_data.size = %s" % test_data.size)

# if we use model maker, cache the model so we don't have to load it for each thread
model = getTFLiteModel(args) if args.project == "modelmaker" else None

def getScore(dataset_slice, step):
	# get the classifier for the thread, initialize with existing model info
	classifier = get_classifier(args.project, args.run_dir, args.run_id, args.model_version, model=model)

	# by calculating the scores seperately for each thread, we don't need locks
	score = Score()

	for text, label, name in samples[dataset_slice]:
		step()

		predicted_label = classifier.predict_label(text)
		predicted_positive = predicted_label == "predator"
		is_positive = label == "predator"

		score.add_prediction(predicted_positive, is_positive)

		# print("percentage = %s" % percentage)
		# print("prefix = %s" % prefix)
		# print("predicted_label = %s" % predicted_label)
		# print("ground truth label = %s" % label)

	return score

future_scores = iterate_multi_threaded(test_data.size, args.threads, getScore)
score = sum(future_scores) # add all returned percentage scores

print("Final score:")
print(score)

# save to file
with open("%s/%s/%s-score.txt" % (args.run_dir, args.model_version, args.split), "w") as file:
	file.write("%s\n\n" % score)
	file.write("Generated this accuracy evaluation at %s\n" % getTimestamp())
