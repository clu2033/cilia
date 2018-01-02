import os, sys, csv
import numpy as np
import nibabel as nib
from PIL import Image


def write_coordinates(files):
	with open('coordinates.csv', 'w') as f:
		writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		for f in files:
			img = Image.open(f).convert('L')
			X = np.array(img).astype(np.uint8)
			coordinates = [(a,b) for a,b in zip(np.nonzero(X)[0],np.nonzero(X)[1])]
			writer.writerow(coordinates)

def main():
	if len(sys.argv) != 2: 
		print('Enter path to files.')
		filepath = input()
	else:
		filepath = sys.argv[1]

	if os.path.isabs(filepath): FILE_PATH = filepath
	else: FILE_PATH = os.path.abspath(filepath)

	files = os.listdir(FILE_PATH)
	files.sort()
	paths = [FILE_PATH + '/' + _ for _ in files]
	write_coordinates(paths)

if __name__ == '__main__':
	main()
