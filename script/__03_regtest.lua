-- training face landmark regressor.
-- Last modified : 2017.08.07, nashory


require 'torch'
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'optim'
require 'nn'
require 'math'
require 'image'
require 'layers.DataLoaderReg'
require 'layers.GridUnit'
require 'layers.SpatialGridSrch'
require 'misc.trainer'
local opts = require 'misc.opts'
local utils = require 'tools.mcutils'

-- for random number generation.
math.randomseed(123)

-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

local loader = DataLoaderReg{data_t7 = opt.data_t7, img_h5 = opt.img_h5, data_json = opt.data_json, data_h5 = opt.data_h5, batch_size = opt.batch_size}


-------------------- GET FEATURE FROM PRETRIANED MODEL ------------------------------
-- load pretrained model
local model_num = opt.pretrain_modelIter
local model_iter = opt.regressor_modelIter
local exp = 'EXP5'
local model_path = 'repo/pretrain/' .. exp  .. '/3-conv_Iter' .. model_num .. '.t7'
local pretrain_model = torch.load(model_path)
local model = pretrain_model['model']
model:evaluate()

-- load regressor.
local reg = torch.load('repo/regressor/reg_M'.. model_num  ..'Iter' .. model_iter .. '.t7')
reg:evaluate()


-- test for N times.
local times = 100
local mse_sum = 0
local iod_sum = 0
for t = 1, times do
	local trgidx = math.random(15001, 20000)
	--local trgidx = math.random(1, 15000)
	local data = loader:get_sample(trgidx)
	local feature = model:forward({data[1]:cuda(), data[2]:cuda()})
	local input = {feature[1]:cuda(), feature[2]:cuda(), data[3]:cuda()}
	local predict = reg:forward(input)
	predict = predict:float()


	-- calculate iterocular distance error (IOD)
	local iod_gt = math.sqrt(data[4][{{},{1,2}}]:clone():csub(data[4][{{},{3,4}}]):pow(2):sum())
	local iod_pred = math.sqrt(predict[{{},{1,2}}]:clone():csub(predict[{{},{3,4}}]):pow(2):sum())
	local iod_err = utils.float(math.abs(iod_gt-iod_pred)/iod_gt*100, 2)
	iod_sum = iod_sum + iod_err


	-- calculate mean MSE
	local mse_le = utils.float(math.sqrt(torch.csub(predict[{{},{1,2}}], data[4][{{},{1,2}}]):pow(2):sum()), 3)
	local mse_re = utils.float(math.sqrt(torch.csub(predict[{{},{3,4}}], data[4][{{},{3,4}}]):pow(2):sum()), 3)
	local mse_no = utils.float(math.sqrt(torch.csub(predict[{{},{5,6}}], data[4][{{},{5,6}}]):pow(2):sum()), 3)
	local mse_lm = utils.float(math.sqrt(torch.csub(predict[{{},{7,8}}], data[4][{{},{7,8}}]):pow(2):sum()), 3)
	local mse_rm = utils.float(math.sqrt(torch.csub(predict[{{},{9,10}}], data[4][{{},{9,10}}]):pow(2):sum()), 3)
	local mse_avg = (mse_le + mse_re + mse_no + mse_lm + mse_rm) / 5.0

	local log = 'LE: ' .. mse_le .. '	RE: ' .. mse_re .. '	NO: ' .. mse_no .. '	LM: ' .. mse_lm .. '	RM: ' .. mse_rm .. '	AVG: ' .. mse_avg
	print(log)
	mse_sum = mse_sum + mse_avg


	-- save images with ldmk.
	local display = data[1]:squeeze():clone()
	data[4] = torch.floor(data[4])
	predict = torch.floor(predict)

	-- Ground Truth (GT)
	display = image.drawRect(display, data[4][{1,1}], data[4][{1,2}], data[4][{1,1}]+1, data[4][{1,2}]+1, {color = {0,255,0}})
	display = image.drawRect(display, data[4][{1,3}], data[4][{1,4}], data[4][{1,3}]+1, data[4][{1,4}]+1, {color = {0,255,0}})
	display = image.drawRect(display, data[4][{1,5}], data[4][{1,6}], data[4][{1,5}]+1, data[4][{1,6}]+1, {color = {0,255,0}})
	display = image.drawRect(display, data[4][{1,7}], data[4][{1,8}], data[4][{1,7}]+1, data[4][{1,8}]+1, {color = {0,255,0}})
	display = image.drawRect(display, data[4][{1,9}], data[4][{1,10}], data[4][{1,9}]+1, data[4][{1,10}]+1, {color = {0,255,0}})
	
	-- prediction
	display = image.drawRect(display, predict[{1,1}], predict[{1,2}], predict[{1,1}]+1, predict[{1,2}]+1, {color = {255,0,0}})
	display = image.drawRect(display, predict[{1,3}], predict[{1,4}], predict[{1,3}]+1, predict[{1,4}]+1, {color = {255,0,0}})
	display = image.drawRect(display, predict[{1,5}], predict[{1,6}], predict[{1,5}]+1, predict[{1,6}]+1, {color = {255,0,0}})
	display = image.drawRect(display, predict[{1,7}], predict[{1,8}], predict[{1,7}]+1, predict[{1,8}]+1, {color = {255,0,0}})
	display = image.drawRect(display, predict[{1,9}], predict[{1,10}], predict[{1,9}]+1, predict[{1,10}]+1, {color = {255,0,0}})
	display = image.scale(display, 128, 128)
	image.save('repo/output/' .. t .. '.png', display)



end
print('Mean MSE : ' .. mse_sum/times)
print('Mean IOD Error : ' .. iod_sum/times)















