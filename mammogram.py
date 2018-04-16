"""
Used to get mammogram objects ready for tensorflow calls




"""

import numpy as np
import tensorflow as tf
import os

NUM_CLASSES = 3

def _getFilenames(root):
	print('------ Reading in images from %s ------' % root)

	filenames=[]

	for root, dirs, files in os.walk(root, topdown=False):
		for file in files:
			if file.endswith(".jpg"):
				filenames.append(os.path.join(root, file))


	return filenames

def inputs(eval_data, data_dir, batch_size):

	filenames = _getFilenames(data_dir)

	#Get filename queue
	# string_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)
	# tf.random_shuffle(string_tensor)

	# fq =tf.FIFOQueue(capacity=NUM_CLASSES, dtypes=tf.string)
	fq = tf.train.string_input_producer(filenames)
	fq_op = fq.enqueue_many([string_tensor])

	tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_op]*1))

	#Set up jpeg reader
	im_reader = tf.WholeFileReader()
	_, im = im_reader.read(fq)


def main():
	root = '/Volumes/ExternalDrive/Mammograms/'
	filenames = _getFilenames(root)

	#Get filename queue
	string_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)
	tf.random_shuffle(string_tensor)

	# fq =tf.FIFOQueue(capacity=NUM_CLASSES, dtypes=tf.string)

	fq = tf.train.string_input_producer(filenames)
	fq_op = fq.enqueue_many([string_tensor])


	tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_op]*1))

	#Set up jpeg reader
	im_reader = tf.WholeFileReader()
	_, im = im_reader.read(fq)

	#Decode as jpeg into a Tensor

	image = tf.image.decode_jpeg(im)

	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		#Coordinate loading of image files
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		image_tensor = np.array(sess.run([image]))
		print(image_tensor.shape)

		#Finish filename queue coordinator
		coord.request_stop()
		coord.join(threads)

main()
# root = '/Volumes/ExternalDrive/Mammograms/'
# print(len(_getFilenames(root)))