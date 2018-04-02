import numpy as np
import tensorflow as tf

from interface import cnn_model_fn
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


def log_tensors():
	""""
	Determines which variables we log

	"""
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
	  tensors=tensors_to_log, every_n_iter=50)
	return logging_hook	

## Defintiions

def main(unused_argv):
	# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)		
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)



	temp_num = 1000
	# train_data = train_data[1:temp_num]
	# train_labels = train_labels[1:temp_num]
	# eval_data = eval_data[1:temp_num]
	# eval_labels = eval_labels[1:temp_num]
	BATCH_SIZE = 100
	STEPS = 20000

	##Create Esimator object. Used for high level training, evaluation and inference
	mnist_classifier = tf.estimator.Estimator(
		model_fn = cnn_model_fn, model_dir='..//tmp/mnist_convnet_model')
	# print('Created Estimator')
	## Setup our logging functions
	logging_hook = log_tensors()


	# Train the model


	## Train Model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y = train_labels,
		batch_size= BATCH_SIZE,
		num_epochs=None,
		shuffle=True)
	mnist_classifier.train(
		input_fn = train_input_fn,
		steps=STEPS,
		hooks= [logging_hook])
  # Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x": eval_data},
	  y=eval_labels,
	  num_epochs=1,
	  shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)
	# print('Begin Evaluating Model')
	# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	# 	x={"x":eval_data},
	# 	y=eval_labels,
	# 	num_epochs=1,
	# 	shuffle=False)
	# eval_results= mnist_classifier.evaluate(input_fn=eval_input_fn)
	# print(eval_results)

if __name__ == "__main__":
	tf.app.run() # handles all flags. then runs code
