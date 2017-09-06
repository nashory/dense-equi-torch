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

-- for random # generation
math.randomseed(os.time())

-- create data loader.
function DataLoaderReg:create(opt)
	local loader = M.DataLoaderReg(opt)
	return loader
end


function DataLoaderReg:__init(opt)

	-- load training image and ldmk annotations from .t7
	print('Loading image dataset... ' .. opt.data_t7 .. '  (this may take several mins.)')
	self.dataset = torch.load(opt.data_t7)
	self.batchsize = opt.batch_size


	-- load json, h5
	print('Loading json,h5 files... ' .. opt.data_json .. '  '  .. opt.data_h5)
	local json_file = utils.read_json(opt.data_json)
	self.dbsize = json_file._dbsize
	self.imsize = json_file._imsize
	self.imlen = json_file._imlen
	self.cropsize = math.ceil(self.imsize*(1.0-1/8.0))
	
	local h5_file = hdf5.open(opt.data_h5)
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

-- Note that this is function calcuates mean/std per channel based on 1 batch.
function DataLoaderReg:normalize(data)
	local mean = {}
	local std = {}
	local channel = {'r','g','b'}
	for i, ch  in ipairs(channel) do
		-- normalize each channel globally.
		mean[i] = data[{{},i,{},{}}]:mean()
		std[i] = data[{{},i,{},{}}]:std()
		data[{{},i,{},{}}]:add(-mean[i])
		data[{{},i,{},{}}]:div(std[i])
	end

	return data
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
	
	-- adjust warp (uncomment this when nobias version is used.)
	--for i = 1, 10, 2 do
	--	anno[{i}] = anno[{i}] - warp[{2, anno[{i+1}], anno[{i}]}]
	--	anno[{i+1}] = anno[{i+1}] - warp[{1, anno[{i+1}], anno[{i}]}]
	--end

	-- rescale (ratio = 1.0)
	--anno:div(self.cropsize)

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

	--[[
		-- no bias ver.
		local im = self:load_im(iid)
		local anno = self:load_anno(iid)
		im = self:transform(im)
		im1, warpfield1 = self:warp(im, idx1)
		im2, warpfield2 = self:warp(im1, idx2)
		anno = self:adjust_anno(anno, warpfield1)
		batch1[{k, {}, {}, {}}] = im1
		batch2[{k, {}, {}, {}}] = im2
		g_matrix[{k, {}, {}}] = image.scale(warpfield2, 40, 40):div(2)
		annotation[{k, {}}] = anno
	]]--
		--[[
		-- test
		local display = image.drawRect(im1, anno[{1}], anno[{2}], anno[{1}]+1, anno[{2}]+1)
		display = image.drawRect(display, anno[{3}], anno[{4}], anno[{3}]+1, anno[{4}]+1)
		display = image.drawRect(display, anno[{5}], anno[{6}], anno[{5}]+1, anno[{6}]+1)
		display = image.drawRect(display, anno[{7}], anno[{8}], anno[{7}]+1, anno[{8}]+1)
		display = image.drawRect(display, anno[{9}], anno[{10}], anno[{9}]+1, anno[{10}]+1)
		image.save('test/' .. k ..'.png', display)
		]]--

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

	-- normalize batch images.
	--batch1 = self:normalize(batch1)
	--batch2 = self:normalize(batch2)

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

--[[
	-- no-bias ver.
	im1, warpfield1 = self:warp(im, idx1)
	im2, warpfield2 = self:warp(im1, idx2)
	anno = self:adjust_anno(anno, warpfield1)
	g_matrix = image.scale(warpfield2, 40, 40):div(2)
	im1 = torch.repeatTensor(im1, 1, 1, 1, 1)
	im2 = torch.repeatTensor(im2, 1, 1, 1, 1)
	g_matrix = torch.repeatTensor(g_matrix, 1, 1, 1, 1)
	anno = torch.repeatTensor(anno, 1, 1)
	return {im1, im2, g_matrix, anno}
]]--
	
	-- normalize batch images.
	--im1 = self:normalize(im1)
	--im2 = self:normalize(im2)
	
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










