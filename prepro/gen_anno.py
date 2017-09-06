# this code preprocess the input image and save to the Tensor vector.
# last modified : 2017.07.12, nashory
# requirements : tps library, MIT lecense. (https://github.com/olt/thinplatespline)

import os, sys
from os.path import join
import dlib
sys.path.insert(0, '../tools')
from PIL import Image
from skimage import io
import json
import argparse
import h5py
import numpy as np
import mcutils as utils
from tps import TPS, TPSError, from_control_points
import random


def tps_warp(width, height, ctlpts, stdev):
	npts = len(ctlpts['_x'][0])
	tps_pts = []
	for i in range(npts):
		for j in range(npts):
			_varx = int(random.random() * width * stdev)
			_vary = int(random.random() * height * stdev)
			tps_pts.append((ctlpts['_x'][i][j], ctlpts['_y'][i][j], _varx, _vary))

	t = TPS(tps_pts)

	_wfx = np.zeros((width,height), dtype='float32')
	_wfy = np.zeros((width,height), dtype='float32')
	for w in range(width):
		for h in range(height):
			_wfx[w][h] = t.transform(w,h)[0]
			_wfy[w][h] = t.transform(w,h)[1]
	warpfield = {'_wfx':_wfx, '_wfy':_wfy}

	return warpfield


# create control points randomly.
def control_points(width, height, offset, npts):
	# (src_x, src_y, height_x, height_y) --> (displacement-x, displacement-y)
	if not ((width-2*offset)<npts and (height-2*offset)<npts):
		w = width - 2*offset
		h = height - 2*offset

	_x = np.zeros((npts, npts), dtype ='uint32')
	_y = np.zeros((npts, npts), dtype ='uint32')
	for i in range(npts):
		for j in range(npts):
			_x[i][j] = offset +  int(i*(w/float(npts-1)))
			_y[i][j] = offset +  int(j*(w/float(npts-1)))

	ctlpts = {'_x':_x, '_y':_y}
	return ctlpts




def main(params):
	# parsing parameters.
	data_root = params['data_root']
	save_h5 = params['save_h5']
	imsize = params['imsize']
	dbsize = params['dbsize']
	anno_root = params['anno_root']


	# make directories.
	utils.make_directory(utils.get_folder_name(save_h5, -2))

	# parameter settings.
	npts = 4
	offset = int(imsize/8.0)
	stdev = 0.12			# 0.08 0.12

	
	# get flow field using TPS.
	print '[Step 1. Generate TPS warp matrix ... ]'
	g_mat_x = []
	g_mat_y = []
	for iter in range(dbsize):
		cropsize = int(imsize*(1.0 - 1/8.0))
		ctl_pts = control_points(cropsize,cropsize,offset,npts)
		g = tps_warp(cropsize,cropsize,ctl_pts,stdev)
		g_mat_x.append(g['_wfx'])
		g_mat_y.append(g['_wfy'])
		print 'generating ... ', iter , '-th warpfield.'
	
	# save to h5 file.
	f = h5py.File(params['save_h5'], "w")
	f.create_dataset("_wfx", dtype='float32', data=g_mat_x)
	f.create_dataset("_wfy", dtype='float32', data=g_mat_y)
	print 'wrote', params['save_h5']
	
	
	
	# annotations (ldmk)
	adata = utils.readlines_txt(anno_root)
	info = []
	for idx in range(len(adata)):
		if idx >= 2:			# remove first two lines.
			targ = adata[idx]
			targ = targ.replace('\r\n', '')
			targ = targ.replace('   ', '\t')
			targ = targ.replace('  ', '\t')
			targ = targ.replace(' ', '\t')
			info.append(targ.split('\t'))

	cnt = 0
	success = 0
	utils.refresh_file('save/imlist.txt')
	if (params['detect']):
		for mdata in info:
			cnt = cnt+1
			im_path = os.path.join(data_root, mdata[0])
			im = io.imread(im_path)

			dets = detector(im, 1)
			if len(dets)>=1:
				contents = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" % (mdata[0], mdata[1], mdata[2], mdata[3], mdata[4], mdata[5], mdata[6], mdata[7], mdata[8], mdata[9], mdata[10], dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom())
				utils.write_txt('save', 'imlist.txt', contents)
				print '[' + str(cnt) + ']	' + mdata[0] + '	(' + str(success) + ' success so far.)'
				success = success + 1
			else:
				print '[' + str(cnt) + ']	' + mdata[0] + '	(failed)'
	else:
		for mdata in info:
			cnt = cnt+1
			im_path = os.path.join(data_root, mdata[0])
			im = io.imread(im_path)
			
			width, height, channel = im.shape
			contents = "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s" % (mdata[0], mdata[1], mdata[2], mdata[3], mdata[4], mdata[5], mdata[6], mdata[7], mdata[8], mdata[9], mdata[10], 0, 0, width-1, height-1)
			utils.write_txt('save', 'imlist.txt', contents, False)
			print '[' + str(cnt) + ']	' + mdata[0] + '	(no face detect)'
			success = success + 1
	
	# save to json file.
	out = {}
	out['_imlen'] = success
	out['_dbsize'] = params['dbsize']
	out['_imsize'] = params['imsize']
	out['_data_root'] = params['data_root']
	out['_anno_root'] = params['anno_root']
	json.dump(out, open(params['save_json'], 'w'))
	print 'wrote', params['save_json']
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# options
	parser.add_argument('--data_root', default='/home1/work/nashory/data/CelebA/Img/img_align_celeba')
	parser.add_argument('--anno_root', default='/home1/work/nashory/data/CelebA/Anno/list_landmarks_align_celeba.txt')
	parser.add_argument('--save_h5', default='./save/prepro_data.h5')
	parser.add_argument('--save_json', default='./save/prepro_data.json')

	parser.add_argument('--imsize', default=96, type=int, help='image width/height for resize.')
	parser.add_argument('--dbsize', default=1000, type=int, help='# of warpfields for pre-calculation.')
	parser.add_argument('--detect', default=False, type=bool, help='true: face detection. false: no face detection.')
	
	args = parser.parse_args()
	params = vars(args)		# convert to ordinary dict
	print 'parsed input parameters : '
	print json.dumps(params, indent = 4)
	main(params)



