import tensorflow as tf
import sys


sys.path.insert(0, 'Classifier/')
sys.path.insert(0, 'imagePrep/')

from classifier import *
# from imagePrep import *
import imagePrep

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os


## Set flags for 
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('use_ex_data',True,
	"""Choose whether to use MNIST dataset instead of DDSM""")
tf.app.flags.DEFINE_string('data_dir','//Volumes/ExternalDrive/Mammogram',
	"""Set the root folder containing Bening, Cancer and Normal cases""")
tf.app.flags.DEFINE_boolean('prep_images',False,
	"""Sets whether to pre-process dataset. Otherwise load last saved numpy array""")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore CPU instructions warning since we're using GPU


## Use the mnist dataset if you want to see how the program works. Much faster 
if (FLAGS.use_ex_data):
	IMAGE_SIZE=28
	NUM_CLASSES = 10
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	print('Using MNIST dataset')
else:
	IMAGE_SIZE = 500 # Will be used to resize the images
	NUM_CLASSES=3 #Cancer, benign, normal
	print('Reading from DDSM dataset')


#Learning variables
LR = 0.0001
epochs = 10
batch_size = 100


def main():
	x = tf.placeholder(tf.float32,[None,IMAGE_SIZE*IMAGE_SIZE])
	x_shaped = tf.reshape(x,[-1,IMAGE_SIZE,IMAGE_SIZE,1]) #convd2d and max_pool take in 4D arrays, so we reshape our data
	# in shape [i,j,k,l]
	# i=number of training samples, j=height of image, k=weight, l=channel number
	#-1 will dynamically reshape based on number of training samples

	y = tf.placeholder(tf.float32,[None,NUM_CLASSES])

	layer1 = create_new_conv_layer(x_shaped,1,32,[5,5],[2,2],name='layer1')
	layer2 = create_new_conv_layer(layer1,32,64,[5,5],[2,2],name='layer2')

	new_size = IMAGE_SIZE/4

	flattened = tf.reshape(layer2, [-1,new_size*new_size*64])

	wd1 = tf.Variable(tf.truncated_normal([new_size*new_size*64,1000], stddev=0.03),name='wd1')
	bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
	dense_layer1 = tf.matmul(flattened,wd1) + bd1
	dense_layer1=  tf.nn.relu(dense_layer1)

	#more softmax activation

	wd2 = tf.Variable(tf.truncated_normal([1000, NUM_CLASSES], stddev=0.03),name='wd2')
	bd2 = tf.Variable(tf.truncated_normal([NUM_CLASSES], stddev=0.01), name='bd2')
	dense_layer2 = tf.matmul(dense_layer1,wd2) + bd2

	y_ = tf.nn.softmax(dense_layer2)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))

	### Stats stuff

	optimiser = tf.train.AdamOptimizer(learning_rate=LR).minimize(cross_entropy)

	#define accuracy aseeseement operation
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#setup initialisation operator
	init_op = tf.global_variables_initializer()


################################## GET IMAGES READY ######################################

	## Use DDSM Dataset
	if not FLAGS.use_ex_data:

		ddsm= imagePrep.prepDDSM(PREP_IMAGES)
		train_labels = ddsm.train_labels
		train_images = ddsm.train_images
		test_labels  = ddsm.test_labels
		test_images  = ddsm.test_images


	## Use MNIST Dataset
	else:
		train_labels = mnist.train.labels
		train_images = mnist.train.images
		test_labels = mnist.test.labels
		test_images = mnist.test.images
##############################################################################################

	## Only use CPU
	config = tf.ConfigProto(
		device_count = {'GPU':0}
		)


	with tf.Session(config=config) as sess:
		sess.run(init_op)
		total_batch = int(len(train_labels)/batch_size)
		print('Total batch:'+ str(total_batch))

		for epoch in range(epochs):
			avg_cost = 0

			# Run through multiple different batches
			for i in range(total_batch):
				if FLAGS.use_ex_data:
					batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
				else:
					batch_x, batch_y = imagePrep.next_batch(batch_size, train_images, train_labels)

				_, c = sess.run([optimiser, cross_entropy],
					feed_dict={x:batch_x, y:batch_y})
				avg_cost += c/total_batch
			if (FLAGS.use_ex_data):
				test_acc = sess.run(accuracy,
					feed_dict={x:mnist.test.images, y:mnist.test.labels})

			else:
				test_acc = sess.run(accuracy,
					feed_dict={x:test_images, y:test_labels})

			print("epoch:", (epoch+1), "cost=","{:.3f}".format(avg_cost), 
				"test accuracy: {:.3f}".format(test_acc))
		print("\nTraining complete!")

		# Use test data to determine accuracies
		if (FLAGS.use_ex_data):
			print(sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels}))
		else:
			print(sess.run(accuracy,feed_dict={x:test_images, y:test_labels}))

if __name__ == "__main__":
	main()
