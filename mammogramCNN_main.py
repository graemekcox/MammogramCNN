import tensorflow as tf
import sys


sys.path.insert(0, 'Classifier/')
sys.path.insert(0, 'imagePrep/')

from classifier import *
from imagePrep import *

from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Ignore CPU instructions warning since we're using GPU

##Parameters to set before running
USE_EX_DATA = True
# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms'
# root =  '//Volumes/SeagateBackupPlusDrive/Mammograms'
root = '/Volumes/ExternalDrive/Mammograms/'

if (USE_EX_DATA):
	IMAGE_SIZE=28
	NUM_CLASSES = 10
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	print('Using MNIST dataset')
else:
	IMAGE_SIZE = 500
	NUM_CLASSES=3
	print('Reading from DDSM dataset')


#Learning variables
LR = 0.0001
epochs = 10
batch_size = 50


def next_batch(num, data, labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

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
	# images, labels = prepImages(root, IMAGE_SIZE, IMAGE_SIZE,1)

	if not(USE_EX_DATA):
	# images = np.load('/Users/graemecox/Documents/ResearchProject/Code/Data/1000_1000/image_1000_1000.npy')
		images = np.load('Data/images.npy')
		labels = np.load('Data/labels.npy')

		train_images, test_images, train_labels, test_labels = train_test_split(
			images, labels, test_size=0.2, random_state=0)

		print('Length of training images: ' + str(len(train_images)))
		print('Length of training labels: ' + str(len(train_labels)))

		print(train_labels.shape)
		print(test_labels.shape)
		print(train_images.shape)
		print(test_images.shape)

##############################################################################################

	with tf.Session() as sess:
		sess.run(init_op)

		if (USE_EX_DATA):
			total_batch = int(len(mnist.test.labels)/batch_size)
		else:
			total_batch = int(len(train_labels)/batch_size)

		print('Total batch:'+ str(total_batch))

		for epoch in range(epochs):
			avg_cost = 0
			for i in range(total_batch):
				if USE_EX_DATA:
					batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
				else:
					batch_x, batch_y = next_batch(batch_size, train_images, train_labels)

				_, c = sess.run([optimiser, cross_entropy],
					feed_dict={x:batch_x, y:batch_y})
				avg_cost += c/total_batch
			if (USE_EX_DATA):
				test_acc = sess.run(accuracy,
					feed_dict={x:mnist.test.images, y:mnist.test.labels})
				print(mnist.test.images.shape)

			else:
				test_acc = sess.run(accuracy,
					feed_dict={x:test_images, y:test_labels})

			print("epoch:", (epoch+1), "cost=","{:.3f}".format(avg_cost), 
				"test accuracy: {:.3f}".format(test_acc))
		print("\nTraining complete!")


		if (USE_EX_DATA):
			print(sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels}))
		else:
			print(sess.run(accuracy,feed_dict={x:test_images, y:test_labels}))

if __name__ == "__main__":
	main()
