import numpy as np
from PIL import Image
import os
import sys
from skimage import color
import cv2

# sys.path.insert(0, 'imagePrep')
# from imageResizer import * 

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
		# i_shuf = np.random.permutation(np.arange(len(train_images)))
		print('----- Saved images and labels ------')
		np.save('../Data/images.npy',train_images)
		np.save('../Data/labels.npy',labels)
		subfolder = '../Data/'+str(max_x)+'_'+str(max_y)+'/'
		print(subfolder)
		if not os.path.exists(subfolder): # If folder doesn't exist, create it

			os.makedirs(subfolder)

		saveInBatch(train_images, labels,subfolder)

		print('----- Finished writing files -----')

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


def saveInBatch(features, labels, subfolder):
	i_shuf = np.random.permutation(np.arange(len(features)))

	full_len = len(i_shuf)

	numBatch = 4

	print('----- Started writing batch files -----')
	for i in range(numBatch):
		# i1 = (i-1)*full_len/numBatch 
		# i2 = i*full_len/numBatch -1
		# ind = [i1:i2]
		ind = i_shuf[ (i-1)*full_len/numBatch : i*full_len/numBatch -1]

		np.save(subfolder+'image_b'+str(i)+'.npy', features[ind])
		np.save(subfolder+'labels_b'+str(i)+'.npy', labels[ind])



def padArray(data,max_x, max_y):

	x,y,depth = data.shape
	print(x,y)

	if (depth != 1):
		print('This is not a b&w image. Converting now')
		data = convertToBW(data)

	new_im = np.zeros(shape=(max_x,max_y))
	
	x1 = (max_x - x)/2
	x2 = x1+x
	print(x1,x2)

	y1 = (max_y - y)/2
	y2 = y1+y

	new_im[x1:x2, y1:y2] = data


def convertToBW(data):
	gray = color.rgb2gray(data)
	# gray =np.array(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY))
	# gray =np.array(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY))
	return gray


### Uncomment to prep images
# root = '/Volumes/ExternalDrive/Mammograms/'
# max_x = 500
# max_y = 500

# images, labels = prepImages(root, max_x,max_y,1)
# print(images.shape)
# print(labels.shape)
