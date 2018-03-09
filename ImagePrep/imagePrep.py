import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, '/Users/graemecox/Documents/ResearchProject/Code/imagePrep')
from imageResizer import * 

## Reads in all images in root, add saves labels
def prepImages(root, max_x=100, max_y=100,save=0):
	print('Reading in images')

	## Reads in images from directories
	train_images = []
	images = []
	labels = []
	
	classes = ['Cancer','Benign','Normal']
	# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/'
	for root, dirs, files in os.walk(root,topdown=False):
		for file in files:
		# for name in file:
	   		if file.endswith(".jpg"):
	   			fn = os.path.join(root,file)
				# fn = os.path.join(root,file)
				print(fn)
				image = Image.open(fn)
				image= np.array(image.resize((max_x, max_y)))

				train_images.append(image)

				temp_labels= [0,0,0]
				for i in range(len(classes)):
					if classes[i] in fn:
						temp_labels[i] = 1
						labels.append(temp_labels)
		      	# images.append(fn)
		      	# # images.append(fn)
		      	# #Resize image
		      	# image = Image.open(fn)
		      	# image= np.array(image.resize((max_x, max_y)))

		      	# train_images.append(image)

		      	# temp_labels = [0,0,0]
		      	# for i in range(len(classes)):
			      # 	if classes[i] in fn:
			      # 		temp_labels[i] = 1
			      # 		labels.append(temp_labels)


	train_images = np.array(train_images)

	train_images = train_images.reshape(len(train_images), max_x*max_y)

	labels = np.array(labels)

	if save:
		np.save('/Users/graemecox/Documents/ResearchProject/Code/Data/image.npy',train_images)
		np.save('/Users/graemecox/Documents/ResearchProject/Code/Data/labels.npy',labels)

	return train_images, labels


def findBiggestImage(root):
	##Find  max dimensions of images
	print('Reading all images to find biggest size')
	max_x = 0
	max_y = 0

	for root, dirs, files in os.walk(root, topdown=False):
		for name in files:
			if name.endswith(".jpg"):
				fn = os.path.join(root,name)
				image = np.array(Image.open(fn))

				if image.shape[1] > max_x or image.shape[0] > max_y:
					max_x = image.shape[1]
					max_y = image.shape[0]

	return max_x, max_y


# fn = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/Benign/benign_01/case0029/C_0029_1.LEFT_CC.jpg'
# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/'
# root = '/Volumes/SeagateBackupPlusDrive/Mammograms/'

# # max_x, max_y = findBiggestImage(root)
# # print(max_x, max_y)

# images, labels = prepImages(root, 256,256,1)
# print(images.shape)
# print(labels.shape)
# root =  '//Volumes/SeagateBackupPlusDrive/Mammograms'
# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms'

# images,labels  = prepImages(root)

# print(len(images))