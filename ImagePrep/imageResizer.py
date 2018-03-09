## Image Resizer
import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color

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

	# cv2.imshow('gray',gray)
	# cv2.waitKey(0)
	# cv2.destoryAllWindows()
	

# labels= ['Cancer', 'Benign', 'Normal']

# # main_labels = np.empty((0,3),dtype='int')
# main_labels = []

# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/Normal/normal_02/case0203/A_0203_1.LEFT_CC.jpg'

# print(len(labels))

# for i in range(len(labels)):
# 	# temp_labels = np.zeros((3,),dtype='int')
# 	temp_labels = [0,0,0]
# 	if labels[i] in root:
# 		temp_labels[i] = 1


# print(temp_labels)

# main_labels.append(temp_labels)
# main_labels.append(temp_labels)
# main_labels.append(temp_labels)
# main_labels = np.array(main_labels)
# print(main_labels.shape)




# print(np.append(main_labels, temp_labels))
# print(main_labels)

# fn = '/Users/graemecox/Documents/My Pictures/FullSizeRender.jpg'

# image = Image.open(fn)
# image = np.array(image)

# padArray(image, 1000,1000)

