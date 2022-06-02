# With code from the following tutorial:
# https://www.tensorflow.org/lite/tutorials/model_maker_text_classification

import os

from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import text_classifier

from util import getTrainArgs, loadPredatorDatasetFromCSV

# Globals

args = getTrainArgs()

# use mobilebert
spec = model_spec.get(args.model_indicator)
# The model configuration is a variant of Transformer with L=24 hidden layers (i.e., Transformer blocks), a hidden size of H=128, B=512 as bottleneck size, A=4 attention heads, and F=4 inner feed-forward layers.
# https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1

spec.seq_len = args.seq_len
# TODO try 1000 and see what happens # DONE you get an error
# TODO find out why android crashes with 512 and 256 but not with 128
spec.model_dir = args.run_dir
# spec.learning_rate = 0.005
# spec.dropout_rate = 0.5

"""
The MobileBERT model parameters you can adjust are:

* `seq_len`: Length of the sequence to feed into the model.
* `initializer_range`: The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
* `trainable`: Boolean that specifies whether the pre-trained layer is trainable.

The training pipeline parameters you can adjust are:

* `model_dir`: The location of the model checkpoint files. If not set, a temporary directory will be used.
* `dropout_rate`: The dropout rate.
* `learning_rate`: The initial learning rate for the Adam optimizer.
* `tpu`: TPU address to connect to.
"""

"""Step 2.   Load train and test data specific to an on-device ML app and preprocess the data according to a specific `model_spec`."""

print("\n--- Loading Dataset ---\n")

# VTPAN

train_data = loadPredatorDatasetFromCSV(args.data_dir, args.dataset_indicator, "train", spec)
test_data = loadPredatorDatasetFromCSV(args.data_dir, args.dataset_indicator, "test", spec)

# Train the model
print("\n--- Training the model ---\n")

model = text_classifier.create(
    train_data,
    model_spec=spec,
    # epochs = 4,
    # batch_size = 48, # the number of samples to use in one training step.
    # do_train = False,
)

# print a summary of the model
# model.summary()

# Evaluate the model
print("\n--- Evaluating ---\n")

loss, acc = model.evaluate(test_data)
print("loss = %s" % loss)
print("acc = %s" % acc)

# Export as a TensorFlow Lite model with [metadata](https://www.tensorflow.org/lite/convert/metadata).
print("\n--- Exporting non_quantized TensorFlow Lite model with metadata ---\n")

model.export(
    export_dir=os.path.join(args.run_dir, 'non_quantized/')
)

print("\n--- Exporting quantized TensorFlow Lite model with metadata ---\n")
""" Since MobileBERT is too big for on-device applications, use [dynamic range quantization](https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization) on the model to compress it by almost 4x with minimal performance degradation. """

model.export(export_dir=os.path.join(args.run_dir, 'quantized/'))

print("also export metadata (this is also packed with the .tflite though)")
model.export(
    export_dir=args.run_dir,
    export_format=[ExportFormat.LABEL, ExportFormat.VOCAB]
)

"""
The allowed export formats can be one or a list of the following:

*   `ExportFormat.TFLITE`
*   `ExportFormat.LABEL`
*   `ExportFormat.VOCAB`
*   `ExportFormat.SAVED_MODEL`

By default, it just exports TensorFlow Lite model with metadata. You can also selectively export different files. For instance, exporting only the label file and vocab file as follows:
export_format=[ExportFormat.LABEL, ExportFormat.VOCAB]
"""

# print("\n--- Evaluating non_quantized tflite model ---\n")

# """You can evalute the tflite model with `evaluate_tflite` method."""
# acc = model.evaluate_tflite(
# 	os.path.join(args.run_dir, 'non_quantized/model.tflite'),
# 	test_data
# )
# print("non_quantized_lite_loss = %s" % loss)
# print("non_quantized_lite_acc = %s" % acc)

# print("\n--- Evaluating quantized tflite model ---\n")

# acc = model.evaluate_tflite(
# 	os.path.join(args.run_dir, 'quantized/model.tflite'),
# 	test_data
# )
# print("quantized_lite_loss = %s" % loss)
# print("quantized_lite_acc = %s" % acc)


"""
After executing the 5 steps above, you can further use the TensorFlow Lite model file in on-device applications using [BertNLClassifier API](https://www.tensorflow.org/lite/inference_with_metadata/task_library/bert_nl_classifier) in [TensorFlow Lite Task Library](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview).
"""
