"""
Used to get mammogram objects ready for tensorflow calls




"""

import numpy as np
import tensorflow as tf
import os
import re
import mammogram_input

# NUM_CLASSES = 3

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size',32,"""Number of images processed in each batch""")
tf.app.flags.DEFINE_string('data_dir','/Volumes/ExternalDrive/Mammograms/',"""Path to Mammogram Images""")

IMAGE_SIZE = mammogram_input.IMAGE_SIZE
NUM_CLASSES = mammogram_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = mammogram_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

## Hyperparameters
MOVING_AVG_DECAY = 0.9999 #Decay to use for moving average
NUM_EPOCHS_PER_DECAY = 350.0 #Epochs after which learning rate decays
LR_DECAY_FACTOR = 0.1 #Learning rate decay factor
INITIAL_LR = 0.1 #initial learning rate

TOWER_NAME = 'tower'

def _activation_summary(x):

	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activation',x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype=tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _add_loss_summaries(total_loss):
	"""
	Gnerates moving averages for all losses and associated summaries for visualizing the performance of the network

	Input:
		total_loss: Total loss from loss()
	Output:
		loss_avg_op : op for generating moving averages of losses
	"""

	loss_avg = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_avg_os = loss_avg.apply(losses+[total_loss])

	#Attach scalar summary to all individual losses and the total loss
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_avg.average(l))
	return loss_avg_os

def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  
  # with tf.device('/cpu:0'):
  # 	var = tf.get_variable(name, shape,
  # 		initializer= tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  var = _variable_on_cpu(
  	name, shape,
  	tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def inputs(eval_data=False):
	if not FLAGS.data_dir:
		raise ValueError('Supply a data_dir')

	# data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')

	images, labels = mammogram_input.inputs(
		eval_data=eval_data,
		data_dir = FLAGS.data_dir,
		batch_size = FLAGS.batch_size)
	print(images)
	print(labels)

	return images, labels


def inference(images):

	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
			# shape=[5,5,3,64],
			shape=[5,5,1,64], #change since grayscale
			stddev=5e-2,
			wd=None)
		conv = tf.nn.conv2d(images,kernel, [1,1,1,1], padding='SAME')

		with tf.device('/cpu:0'):
			biases = tf.get_variable('biases',[64],
			 initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv,biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv1)


	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
		padding='SAME', name='pool1')

	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

	#Conv 2

	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
			shape=[5,5,64,64],
			stddev=5e-2,
			wd=None)
		conv = tf.nn.conv2d(norm1,kernel, [1,1,1,1], padding='SAME')
		with tf.device('/cpu:0'):
			biases = tf.get_variable('biases',[64],
				initializer=tf.constant_initializer(0.1), dtype=tf.float32)
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
		_activation_summary(conv2)


	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
	                name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
	                     strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
	# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
		                                      stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
		                                      stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)

	# linear layer(WX + b),
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
		                                      stddev=1/192.0, wd=None)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
		                          tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear

def loss(logits, labels):
	"""
	
	Inputs:
		logits: Logits from inference()
		labels: labels from inputs. 1-D tensor of shape [batch_size]
	Outputs:
		Log tensor of type float

	"""
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses',cross_entropy_mean)

	#Total loss is cross entropy loss plus all weight decay terms (L2 terms)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, global_step):
	"""

	Inputs:
		total_loss: Total loss from loss()
		global_step: Integer variable counting total number of training steps processed
	Outputs:
		train_op: op for training

	"""
	# num_batch = NUM_PER_EPOCH/ FLAGS.batch_size
	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps= int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	#Decay learning rate exponentially based on number of steps
	lr = tf.train.exponential_decay(INITIAL_LR,
		global_step,
		decay_steps,
		LR_DECAY_FACTOR,
		staircase=True)
	tf.summary.scalar('learning_rate',lr) #save variable for TensorBoard

	#calculate moving averages
	loss_avg_op = _add_loss_summaries(total_loss)

	#compute gradients
	with tf.control_dependencies([loss_avg_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	#apply gradients
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	#Add histograms for trainable variables
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients',grad)

	#Track the moving averages of all trainable variables
	variable_avg= tf.train.ExponentialMovingAverage(
		MOVING_AVG_DECAY, global_step)
	variables_avg_op = variable_avg.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_avg_op]):
		train_op = tf.no_op(name='train')

	return train_op