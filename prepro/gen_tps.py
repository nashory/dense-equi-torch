# this code preprocess the input image and save to the Tensor vector.
# last modified : 2017.07.12, nashory
# requirements : tps library, MIT lecense. (https://github.com/olt/thinplatespline)

import os, sys
from os.path import join
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
	sampleSize = params['sampleSize']
	loadSize = params['loadSize']
	tpslen = params['tpslen']


	# make directories.
	os.system('mkdir -p prepro/save')

	# parameter settings.
	npts = 4
	offset = int((loadSize-sampleSize)*0.5)
	stdev = 0.12			# 0.08 0.12

	
	# get flow field using TPS.
	print 'generating tps warp matrix ...'
	g_mat_x = []
	g_mat_y = []
	for iter in range(tpslen):
		ctl_pts = control_points(sampleSize,sampleSize,offset,npts)
		g = tps_warp(sampleSize,sampleSize,ctl_pts,stdev)
		g_mat_x.append(g['_wfx'])
		g_mat_y.append(g['_wfy'])
		print 'generating ... ', iter , '-th warpfield.'
	
	# save to h5 file.
	f = h5py.File('prepro/save/tps.h5', "w")
	f.create_dataset("_wfx", dtype='float32', data=g_mat_x)
	f.create_dataset("_wfy", dtype='float32', data=g_mat_y)
	print 'wrote prepro/save/tps.h5'
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# options
	parser.add_argument('--loadSize', default=96, type=int, help='image width/height for resize.')
	parser.add_argument('--sampleSize', default=84, type=int, help='image width/height for resize.')
	parser.add_argument('--tpslen', default=100, type=int, help='# of warpfields for pre-calculation.')


	args = parser.parse_args()
	params = vars(args)		# convert to ordinary dict
	print 'parsed input parameters : '
	print json.dumps(params, indent = 4)
	main(params)



