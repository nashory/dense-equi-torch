-- Dataloader layer.
-- perform transformation(rotation/tps_warping/transition) and feed network.
-- last modified : 2017.07.17, nashory

require 'hdf5'
require 'math'
require 'image'
local Threads = require 'threads'
local utils = require 'tools.mcutils'
Threads.serialization('threads.sharedserialize')

local DataLoaderReg = torch.class('DataLoaderReg')

local opts = require 'script.opts'
local opt = opts.parse(arg)


-- for random # generation
math.randomseed(os.time())

-- create data loader.
function DataLoaderReg:create(opt)
	local loader = M.DataLoaderReg(opt)
	return loader
end


function DataLoaderReg:__init(opt)

	-- create and load train/test image and ldmk annotations
	paths.dofile('../data/gen_reg.lua')
	print('Loading image dataset ... (this may take several mins.)')
	self.trainset = torch.load('../data/save/reg_train.t7')
	self.testset = torch.load('../data/save/reg_train.t7')
	self.batchsize = opt.batchSize

	-- tps warp matrix(g).
	local h5_file = hdf5.open('data/save/tps.h5')
	self.wfx = h5_file:read('/_wfx'):all()
	self.wfy = h5_file:read('/_wfy'):all()
end

function DataLoaderReg:getTotalIter()
	return math.ceil(self.imlen/self.batchsize)
end

-- No rotation for regression. Only center crop applied.
function DataLoaderReg:transform(im)
	-- center crop
	im = self:crop(im, 'center')
	return im
end

function DataLoaderReg:load_im(iid)
	local data = self.dataset[iid]['data']
	return data
end

function DataLoaderReg:load_anno(iid)
	local anno = torch.Tensor(10):zero()
	anno[{1}] = self.dataset[iid]['l_eye_x']
	anno[{2}] = self.dataset[iid]['l_eye_y']
	anno[{3}] = self.dataset[iid]['r_eye_x']
	anno[{4}] = self.dataset[iid]['r_eye_y']
	anno[{5}] = self.dataset[iid]['nose_x']
	anno[{6}] = self.dataset[iid]['nose_y']
	anno[{7}] = self.dataset[iid]['l_mouth_x']
	anno[{8}] = self.dataset[iid]['l_mouth_y']
	anno[{9}] = self.dataset[iid]['r_mouth_x']
	anno[{10}] = self.dataset[iid]['r_mouth_y']
	
	return anno
end


function DataLoaderReg:warp(im, idx)
	local warpfield = torch.DoubleTensor(2, self.cropsize, self.cropsize)
	warpfield[1] = self.wfx[idx]
	warpfield[2] = self.wfy[idx]
	im_warp = image.warp(im, warpfield)
	return im_warp, warpfield
end

-- lim (degree.)
function DataLoaderReg:rotate(im, lim)
	local ang = (3.141592/180) * math.random(-lim, lim)
	im = image.rotate(im, ang)
	return im
end

function DataLoaderReg:crop(im, option)
	im = utils.crop(im, self.cropsize, option)
	return im
end

function DataLoaderReg:adjust_anno(anno, warp)
	-- adjust crop
	local offset = (self.imsize - self.cropsize)/2
	anno:csub(offset)
	
	return anno
end

function DataLoaderReg:get_batch(batchsize)
	local data = {}			-- save { im1, im2, g }
	--self.cropsize = self.imsize
	local batch1 = torch.Tensor(batchsize, 3, self.cropsize, self.cropsize):zero()
	local batch2 = torch.Tensor(batchsize, 3, self.cropsize, self.cropsize):zero()
	local g_matrix = torch.Tensor(batchsize, 2, 40, 40):zero()
	local annotation = torch.Tensor(batchsize, 10):zero()
	
	-- create random iid index list.
	local iid_list = {}
	local t = 0
	for k = 1, batchsize do
		-- local iid = math.random(1,self.imlen)
		local iid = math.random(1, 5000)
		local idx1 = math.random(1,self.dbsize)
		local idx2 = math.random(1,self.dbsize)
		table.insert(iid_list, iid)

		-- bias ver.
		local im = self:load_im(iid)
		local anno = self:load_anno(iid)
		im = self:transform(im)
		im1, warpfield1 = self:warp(im, idx1)
		anno = self:adjust_anno(anno, warpfield1)
		batch1[{k, {}, {}, {}}] = im
		batch2[{k, {}, {}, {}}] = im1
		g_matrix[{k, {}, {}}] = image.scale(warpfield1, 40, 40):div(84.0/40)
		annotation[{k, {}}] = anno
	end

	table.insert(data, {batch1, batch2, g_matrix, annotation})
	return data
end


-- return one sample input tuple {im1, im2, g}
function DataLoaderReg:get_sample(trgidx)
	local idx1 = math.random(1,self.dbsize)
	local idx2 = math.random(1,self.dbsize)
	
	local im = self:load_im(trgidx)
	local anno = self:load_anno(trgidx)
	im = self:transform(im)

	-- bias ver.
	im1, warpfield1 = self:warp(im, idx1)
	anno = self:adjust_anno(anno, warpfield1)
	g_matrix = image.scale(warpfield1, 40, 40):div(2)
	im = torch.repeatTensor(im, 1, 1, 1, 1)
	im1 = torch.repeatTensor(im1, 1, 1, 1, 1)
	g_matrix = torch.repeatTensor(g_matrix, 1, 1, 1, 1)
	anno = torch.repeatTensor(anno, 1, 1)
	
	return {im, im1, g_matrix, anno}
end


--return DataLoader
return DataLoaderReg










