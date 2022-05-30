# import this before tensorflow

# disable TF info logging
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import tensorflow as tf
assert tf.__version__.startswith('2')

def dontUseWholeGPU(tf):
	# dont use the whole gpu
	# # https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
	# # import tensorflow as tf
	# from tensorflow.keras.backend import set_session
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
	# config.log_device_placement = False  # to log device placement (on which device the operation ran)
	# sess = tf.Session(config=config)
	# set_session(sess)  # set this TensorFlow session as the default session for Keras

	# https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if not gpus:
		print("no gpu available")
		return

	for gpu in gpus:
		try:
			tf.config.experimental.set_memory_growth(gpu, True)
		except RuntimeError as e:
			print(e)

dontUseWholeGPU(tf)
