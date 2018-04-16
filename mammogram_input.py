import tensorflow as tf
import os
import numpy as np

IMAGE_SIZE = 500 #Placeholder. Resize to 500 by 500, but should zero pad to max set size
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000

def read_mammograms(filename_queue,labels):
	"""

	Input:
		filename_queue: A queue of strings with the filename to read from

	Output:
		An Object with the following fields:
			height: number of rows
			width: number of columns
			key: a scalar string Tensor describing the filename and record number for this example
			label: an int32 Tensor with label in the range 0-2
			uint8image: a uint8 Tensor with the Image data
	"""
	class MammogramRecord(object):
		pass
	result = MammogramRecord()

	result.height = 500
	result.width  = 500
	# label_bytes   = 1
	# image_bytes = result.height*result.width #grayscale, so only 1 byte deep
	# record_bytes = label_bytes*image_bytes

	#Read a record, getting filenames from filename queue.
	# reader = tf.FixedLenghtRecordReader(record_bytes = record_bytes)
	reader = tf.WholeFileReader()
	result.key, value = reader.read(filename_queue)
	result.image = tf.image.decode_jpeg(value)

	result.labels = tf.cast(labels,tf.int32)
	return result

	# #Convert from string to uint8 vector
	# record_bytes = tf.


def _getFilename(root):
	"""
	Gets all filenames of jpg images

	Input:
		root: folder containing 'Cancer','Benign' and 'Normal' folders.
	"""
	print('Getting filenames in %s' % root)

	filenames = []
	labels= []
	classes = ['Cancer','Benign','Normal']

	for root, dirs, files in os.walk(root, topdown=False):
		for file in files:
			if file.endswith(".jpg"):
				fn = os.path.join(root, file) 
				filenames.append(fn)

				# Add label
				# temp_labels= [0,0,0]
				for i in range(len(classes)):
					if classes[i] in fn:
						# temp_labels[i] = 1
						# labels.append(temp_labels)
						labels.append(i)

	return filenames,labels

def _generate_image_and_label_batch(image,label,min_queue_examples, batch_size, shuffle):
	num_preprocess_threads = 16
	# temp = np.array(label)
	# temp = temp.reshape(-1)

	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3*batch_size,
			min_after_dequeue = min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image,label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+3*batch_size)

	tf.summary.image('images',images)

	return images, label_batch
	# return images, tf.reshape(label_batch, [batch_size])
	# print(label_batch.shape)
	# return images

def inputs(eval_data, data_dir, batch_size):

	if not eval_data:
		print('------ Get Training Set ------')
		filenames, labels = _getFilename(data_dir)
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN


	else:
		print('------ Get Testing Set ------')

	#Read in images now
	with tf.name_scope('input'):
		string_tensor = tf.convert_to_tensor(filenames, dtype=tf.string)
		# tf.random_shuffle(string_tensor)
		#Create queue
		fq = tf.FIFOQueue(capacity=NUM_CLASSES, dtypes=tf.string)
		#Create queue op
		fq_op = fq.enqueue_many([string_tensor])
		#Create QueueRunner and add to add to queue runner list
		tf.train.add_queue_runner (tf.train.QueueRunner(fq, [fq_op]*1))
		# print(fq.dequeue())
		read_input = read_mammograms(fq,labels)
		reshaped_image = tf.cast(read_input.image, tf.float32)

		#Resize Image
		height = 500
		width = 500
		float_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
		float_image.set_shape([height, width, 3])
		# read_input.labels.set_shape([1])

		#Make sure we're shuffling properly
		min_fraction_of_examples_in_queue= 0.4
		min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

		return _generate_image_and_label_batch(float_image, read_input.labels,
			min_queue_examples, batch_size, shuffle=False)


		# filename_queue= tf.train.string_input_producer(filenames)




root= '/Volumes/ExternalDrive/Mammograms/'
images, labels = inputs(False, root, 128)

print(images)
print(labels)


