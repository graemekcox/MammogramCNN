import numpy as np
import os
import cv2
import glob
from sklearn.utils import shuffle
import pickle


# Functions for reading in CIFAR10 dataset
def unpickle(file):
	import cPickle
	with open(file,'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def create_label(i,classes):
	label = np.zeros(len(classes))
	label[i] = 1
	return label

def create_img(img):
  # img = data['data'][index]
  x = 32
  y = 32
  imsize = x*y#Each channel is 1024 parts (32x32)

  r = img[0:imsize]
  g = img[imsize:imsize*2]
  b = img[imsize*2:imsize*3]

  img = np.dstack((np.reshape(r,(x,y)),
    np.reshape(g,(x,y)),
    np.reshape(b,(x,y))))

  img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

  return img

## Out labels
def create_classes():
  dict = unpickle('/Users/graemecox/Documents/DeepLearning/CNN/CIFAR10/cifar-10-batches-py/batches.meta')
  classes= dict['label_names']
  return classes

## Read in all data_batch files and save images and labels
def create_train_data(classes):
  training_data = []
  folder = '/Users/graemecox/Documents/DeepLearning/CNN/CIFAR10/cifar-10-batches-py'
  listd = os.listdir(folder)

  # Get all 'data_batch' files in specified directory
  matching = [s for s in listd if "data_batch" in s]
  print('Reading Training Data')

  for file in matching:
    fn = folder + '/' + file
    data = unpickle(fn)
    for i in range(len(data['labels'])):
      img = create_img(data['data'][i])    
      training_data.append([np.array(img), create_label(data['labels'][i], classes)])

  shuffle(training_data)
  np.save('train_data.npy',training_data)
  return training_data

def create_test_data(classes):
  testing_data = []
  batchPath = '/Users/graemecox/Documents/DeepLearning/CNN/CIFAR10/cifar-10-batches-py'
  # batchPath = '\Users\graemecox\Documents\DeepLearning\CNN\CIFAR10\cifar-10-batches-py'

  newFn = batchPath + '/' + 'test_batch'

  data = unpickle(newFn)

  print('Reading Testing Data')
  for i in range(len(data['labels'])):
    img = create_img(data['data'][i])
    testing_data.append([np.array(img), create_label(data['labels'][i], classes)])

  shuffle(testing_data)
  np.save('test_data.npy',testing_data)
  return testing_data


class DataSet(object):
	def __init__(self):
		labels = create_classes()
		self.test_data = create_test_data(labels)
		self.train_data  = create_train_data(labels)

		self._num_examples = len(self.test_data[0])
		self._images = self.test_data
		# self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

		self.start = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def return_images(self):
		return self.train_data, self.test_data

	def return_classes(self):
		return create_classes()

	def next_batch_train(self,batch_size):
		# if self._epoch_completed == 0 and self.start == 0:
		# 	batch_images = self.train_data[0, 0:batch_size-1]
		# 	batch_labels = self.train_data[1, 0:batch_size-1]
		# 	self._index_in_epoch
		# 	self.start = 1
		if (self.start == 0):
			self.start = 1
		else:
			self._epochs_completed += 1


		index = self._index_in_epoch
		batch_images = self.train_data[index:index+batch_size-1][0]
		batch_labels = self.train_data[index:index+batch_size-1][1]
		self._index_in_epoch += batch_size
		return batch_images, batch_labels

	# def next_batch(self,batch_size):
	# 	if self._epochs_completed == 0 and start == 0 and shuffle:
	# 		perm0 = numpy.arange(self._num_examples)
	# 		numpy.random.shuffle(perm0);
	# 		self._images = self.images[perm0]
	# 		self._labels = self.labels[perm0]

	# 	if start + batch_size > self._num_examples:
	# 		#finished epoch
	# 		self._epochs_completed += 1
	# 		#get rest of examples in epoch
	# 		rest_num_exmaples = self._num_examples - start
	# 		images_rest_part = self._images[start:self._num_examples]
	# 		labels_rest_part = self._labels[start:self._num_examples]

	# 		if shuffle:
	# 			perm = numpy.arange(self._num_examples)
	# 			numpy.random.shuffle(perm)
	# 			self._images = self.images[perm]
	# 			self._labels = self.labels[perm]
	# 		#start next epoch
	# 		start = 0
	# 		self._index_in_epoch = batch_size - rest_num_exmaples
	# 		end = self._index_in_epoch
	# 		images_new_part = self._images[start:end]
	# 		labels_new_part = self._labels[start:end]
	# 		return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part,labels_new_part),axis=0)

	# 	else:
	# 		self._index_in_epoch += batch_size
	# 		end = self._index_in_epoch
	# 		return self._images[start:end], self._labels[start:end]




