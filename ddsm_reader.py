import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

## Uses a LJPEG to JPEG converter found at the following link:
#.      https://github.com/nicholaslocascio/ljpeg-ddsm

def readPatients():
	root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/'
	patients = []
	folders = os.listdir(root)
	folders.remove('.DS_Store')

	for folder in folders:
		temp = root+folder
		subfolders = os.listdir(temp)
		if '.DS_Store' in subfolders:
			subfolders.remove('.DS_Store')

		for subfolder in subfolders:
			tempSubfolder = temp+'/'+subfolder
			cases = os.listdir(tempSubfolder)
			if '.DS_Store' in cases:
				cases.remove('.DS_Store')

			for case in cases:
				
				#get all files in folder
				case = tempSubfolder+'/'+case + '/'
				files = os.listdir(case)
				#convert from ljpeg to jpeg
				# if ".jpg" in files:
				if '.jpg' not in '\t'.join(files):
					print('Convert files from LJPEG to jpg')
					convertLJPEG(case)
				#Find files with the .ics extension
				matching = [s for s in files if '.ics' in s]
				# print(fn_ics)
				patients.append(readICS(folder,case,matching[0].strip('.ics')))

	np.save('patient_data.npy', patients)
	return patients
				
	# patients= np.load('patient_data.npy')


	# # typeim = 'Normal/'
	# print(len(patients))
	# print(patients[10].pathology)



class Mammogram(object):
	def __init__(self):
		self._patients = readPatients()
		self._classes = ['Benign','Cancer','Normal']
	
	@property
	def patients(self):
		return self._patients

	@property
	def classes(self):
		return self._classes

class Patient(object):
	def __init__(self,l_cc,l_mlo,r_cc, r_mlo,fn,density,age, pathlogy):
		self._l_cc = l_cc #From Scan objects
		self._l_mlo = l_mlo
		self._r_cc = r_cc
		self._r_mlo = r_mlo

		self._name = fn
		self._density = density #Breast tissue density ACR
		self._age=  age
		self._pathology = pathlogy

	@property
	def name(self):
		return self._name

	@property
	def density(self):
		return self._density

	@property
	def age(self):
		return self._age

	@property
	def left_cc(self):
		return self._l_cc

	@property
	def left_mlo(self):
		return self._l_mlo

	@property
	def right_cc(self):
		return self._r_c

	@property
	def right_mlo(self):
		return self._r_mlo

	@property
	def pathology(self):
		return self._pathology

class Scan(object):
	def __init__(self,title,xres,yres,BITS_PER_PIXEL,res,data, overlay):
		self._name = title
		self._xres = xres
		self._yres = yres
		self._bpp = BITS_PER_PIXEL
		self._res = res
		self.data = data
		self._overlay = overlay #0 or 1 depending on if overlay file is included.
		# overlay files are present if radiologist found any abnormalities

	@property
	def name(self):
		return self._name	

	@property
	def xres(self):
		return self._xres

	@property
	def yres(self):
		return self._yres

	@property
	def bpp(self):
		return self._bpp
	@property
	def res(self):
		return self._res
	@property
	def overlay(self):
		return self._overlay



	def displayImage(self):
		cv2.imshow(self._name, self.data)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def parseFile(x,name,folder):
	tempSplit = x.split(" ")
	xdim = tempSplit[2]
	ydim = tempSplit[4]
	bpp = tempSplit[6]
	res = tempSplit[8]
	overlay = 'OVERLAY' == tempSplit[9]
	fn_final = folder + name + '.jpg'
	im = cv2.imread(fn_final)
	if im is None: # check if image was read in properly
		print('Error occurred when reading in ' + fn_final + '\n')
		return

	return Scan(fn_final,xdim,ydim,bpp,res,im,overlay)


def convertLJPEG(path):
	# path_to_ddsm = "/Volumes/Storage/figment.csee.usf.edu/pub/DDSM/"

	for root, subFolders, file_names in os.walk(path):
	    for file_name in file_names:
	        if ".LJPEG" in file_name:
	            ljpeg_path = os.path.join(root, file_name)
	            out_path = os.path.join(root, file_name)
	            out_path = out_path.split('.LJPEG')[0] + ".jpg"
	            
	            cmd = '/Users/graemecox/Documents/ResearchProject/Data/ljpeg-ddsm/ljpeg.py "{0}" "{1}" --visual --scale 1.0'.format(ljpeg_path, out_path)
	            os.system(cmd)

	print('done')


def readICS(rootfolder,subfolder,fn):

	#Read in ICS
	icsFn = subfolder + fn + '.ics'
	with open(icsFn) as f:
		content = f.readlines()
	content = [x.strip() for x in content]

	##Convert all LJEG to jpeg now

	fn = fn.replace('-','_')
	# fn = fn.remove('.ics')

	for x in content:
		if "DATA_OF_STUDY" in x:
			tempSplit = x.split(" ")
			date = tempSplit[1]
		if "filename" in x:
			tempSplit = x.split(" ")
			name = tempSplit[1]
		if "PATIENT_AGE" in x:
			tempSplit = x.split(" ")
			age = tempSplit[1]
		if "DENSITY" in x:
			tempSplit = x.split(" ")
			density= tempSplit[1]
		if "LEFT_CC" in x:
			imName = subfolder+fn
			left_cc = parseFile(x, '.LEFT_CC', imName)
			if left_cc is None:
				print("Error occurred when creating Scan object: left_cc")

		if "LEFT_MLO" in x:
			imName = subfolder+fn
			left_mlo = parseFile(x, '.LEFT_MLO', imName)
			if left_mlo is None:
				print("Error occurred when creating Scan object: left_mlo")

		if "RIGHT_CC" in x:
			imName = subfolder+fn
			right_cc = parseFile(x, '.RIGHT_CC', imName)
			if right_cc is None:
				print("Error occurred when creating Scan object: right_cc")

		if "RIGHT_MLO" in x:
			imName = subfolder+fn
			right_mlo = parseFile(x, '.RIGHT_MLO', imName)
			if right_mlo is None:
				print("Error occurred when creating Scan object: right_mlo")

	patient = Patient(left_cc,
		left_mlo,
		right_cc,
		right_mlo,
		name,
		density,
		age,
		rootfolder)

	return patient





# # root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/Normal/normal_01/'
# case = 'case0003/'
# fn = 'A-0003-1.'

# fn_final = root+case+fn+'ics'

# fn = 'A_0003_1.' #LJPEG to JPEG conversion file changes '-' to '_'
# patient_1 = readICS(fn)

# print(patient_1.name)
# patient_1.left_cc.displayImage()
# with open(fn_final,'r') as f:
# 	content = f.readlines()
# 	print(content)
# 	return
# print(xdim)
# print(ydim)
# print(content.__len__())
#print(tempSplit)
# print(l_cc_xdim)