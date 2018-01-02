import os, sys, csv
import numpy as np
import nibabel as nib
from PIL import Image

DIR_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
SAVE_PATH = DIR_PATH + 'nifti_converted/'
if not os.path.exists(SAVE_PATH): os.mkdir(SAVE_PATH)

def convert_nifti(paths, files):
	for i,v in enumerate(paths):
		save_path = SAVE_PATH + files[i][:-7]
		X = nib.load(v)
		X.set_data_dtype(np.uint8)
		X = np.array(X.dataobj)
		X = X.reshape(X.shape[0], X.shape[1])
		X = X.astype(np.uint8)
		X[X == 1] = 255
		im = Image.fromarray(X)
		im = im.convert('RGB')
		im.save(save_path + '.png')

def main():
	if len(sys.argv) != 2: 
		print('Enter path to files to convert.')
		filepath = input()
	else:
		filepath = sys.argv[1]

	if os.path.isabs(filepath): FILE_PATH = filepath
	else: FILE_PATH = os.path.abspath(filepath)

	files = os.listdir(FILE_PATH)
	files.sort()
	paths = [FILE_PATH + '/' + _ for _ in files]
	convert_nifti(paths, files)

if __name__ == '__main__':
	main()
