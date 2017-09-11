# code for utility function for convenience.


import os, sys
from os.path import isfile, join
import shutil
from distutils.dir_util import copy_tree
from PIL import Image

## Progress bar.
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()
	
	# usage example.
	#for i in range(0, 100):
	#	printProgress(i, 100, 'Progress:', 'Complete', 1, 50)



##I/O
def make_directory(path):
	if not os.path.exists(path):
		os.makedirs(path)
def remove_directory(path):
	if os.path.exists(path):
		shutil.rmtree(path)

def refresh_directory(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	make_directory(path)

def file_exists(path):
	if os.path.exists(path):
		return True
	else:
		return False
		

def refresh_file(path):
	if (file_exists(path)):
		os.remove(path)
	f = open(path, 'w')
	f.close()

def copy_directory(_from, _to):
	copy_tree(_from, _to)

def get_folder_name(path, index='None'):
	mylist = path.split('/')
	if index == 'None':
		return mylist[-1]
	else : 
		return mylist[index]

def write_txt(path, fname, contents, flag=False):
	make_directory(path)
	full_path = os.path.join(path,fname)
	log = open(full_path, 'a')
	log.write(contents + '\n')
	log.close()
	if (flag):
		print '-> [%s] has been written in %s.' % (contents, fname)

def readlines_txt(file):
	lines = []
	f = open(file, 'r')
	for line in f:
		lines.append(line)
	f.close()
	return lines



# return file list in the folder. (it return only files.)
def get_file_list(path):
	onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
	return onlyfiles

# image I/O
def load_img(path):
	im = Image.open(path)
	return im

# arr : [W  x H x C]
def numpy2img(arr):
	im = Image.fromarray(arr, 'RGB')
	return im

def crop_img(img, opt='center'):
	if opt == 'center':
		return 'center_cropped.'






