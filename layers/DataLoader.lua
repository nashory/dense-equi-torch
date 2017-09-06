-- Multi-threaded Dataloader layer.
-- code brought and modified using DCGAN torch code.
-- perform transformation(rotation/tps_warping/transition) and feed network.
-- last modified : 2017.09.06, nashory

require 'hdf5'
require 'math'
require 'image'
require 'torch'
local utils = require 'tools.mcutils'
local DataLoader = torch.class('DataLoader')


-- for random # generation
math.randomseed(os.time())

-- get options
local opts = require 'script.opts'
local opt = opts.parse(arg)

-- DataLoader
function DataLoader:__init()
	-- load multi-thread dataloader.
	local loader = paths.dofile('../data/data.lua')
	self.dataset = loader.new(opt.nthreads, opt)

	-- load options
	self.batchSize = opt.batchSize
	self.loadSize = opt.loadSize
	self.sampleSize = opt.sampleSize
	self.outputSize = opt.outputSize
	self.tpslen = opt.tpslen
	self.imlen = self.dataset:size()

	-- generate tps warp matrix and load.
	os.execute('python prepro/gen_tps.py')

	-- load generated tps matrix
	local h5_file = hdf5.open(opt.tps_h5)
	self.wfx = h5_file:read('/_wfx'):all()
	self.wfy = h5_file:read('/_wfy'):all()
end

function DataLoader:getTotalIter()
	return math.ceil(self.imlen/self.batchSize)
end


function DataLoader:transform(im)
	-- rotation
	local lim = 25
	im = self:rotate(im, lim)
	-- crop
	im = self:crop(im)

	return im
end

function DataLoader:load_im(iid)
	local data = self.dataset[iid]['data']
	return data
end


function DataLoader:warp(im, idx)
	local warpfield = torch.DoubleTensor(2, self.cropsize, self.cropsize)
	warpfield[1] = self.wfx[idx]
	warpfield[2] = self.wfy[idx]
	im_warp = image.warp(im, warpfield)
	return im_warp, warpfield
end

-- lim (degree.)
function DataLoader:rotate(im, lim)
	local ang = (3.141592/180) * math.random(-lim, lim)
	im = image.rotate(im, ang)
	return im
end

function DataLoader:crop(im)
	im = utils.crop(im, self.cropsize, 'random')
	return im
end

function DataLoader:get_batch(batchSize)
	local data = {}			-- save { im1, im2, g }
	
	local batch1 = torch.Tensor(batchSize, 3, self.sampleSize, self.sampleSize):zero()
	local batch2 = torch.Tensor(batchSize, 3, self.sampleSize, self.sampleSize):zero()
	local g_matrix = torch.Tensor(batchSize, 2, self.outputSize, self.outputSize):zero()
	
	-- create random iid index list.
	local iid_list = {}
	for k = 1, batchSize do
		local iid = math.random(1, self.imlen)
		local idx1 = math.random(1,self.tpslen)
		local idx2 = math.random(1,self.tpslen)
		table.insert(iid_list, iid)
	
		-- unlike stated in the paper, we performed deformation only once since it has no harm.
		local im = self.dataset:getim(iid)
		im = self:transform(im)
		im1, warpfield1 = self:warp(im, idx1)
		
		batch1[{k, {}, {}, {}}] = im
		batch2[{k, {}, {}, {}}] = im1
		g_matrix[{k, {}, {}}] = image.scale(warpfield1, self.outputSize, self.outputSize):div(self.sampleSize/self.outputSize)
	end

	
	table.insert(data, {batch1, batch2, g_matrix})
	return data
end


-- return one sample input tuple {im1, im2, g}
function DataLoader:get_sample(trgidx)
	local idx1 = math.random(1,self.tpslen)
	local idx2 = math.random(1,self.tpslen)
	
	local im = self.dataset:getim(trgidx)
	im = self:transform(im)

	-- unlike stated in the paper, we performed deformation only once since it has no harm.
	im1, warpfield1 = self:warp(im, idx1)
	g_matrix = image.scale(warpfield1, self.outputSize, self.outputSize):div(self.sampleSize/self.outputSize)

	-- resize.
	im = torch.repeatTensor(im, 1, 1, 1, 1)
	im1 = torch.repeatTensor(im1, 1, 1, 1, 1)
	g_matrix = torch.repeatTensor(g_matrix, 1, 1, 1, 1)

	return {im, im1, g_matrix}
end

return DataLoader










