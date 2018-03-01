## Convert LJPEG to JPEG given a root folder
import os


def readFolders(root):
	# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/'
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
				cases.remove('.DS_Store') #get rid of that useless item

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
				# matching = [s for s in files if '.ics' in s]
				# print(fn_ics)
				# patients.append(readICS(folder,case,matching[0].strip('.ics')))

	# np.save('patient_data.npy', patients)
	return patients


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

root = '/Volumes/SeagateBackupPlusDrive/Mammograms/'
# root = '/Users/graemecox/Documents/ResearchProject/Data/Mammograms/'
readFolders(root)