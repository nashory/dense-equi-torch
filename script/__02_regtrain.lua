-- training face landmark regressor.
-- Last modified : 2017.08.07, nashory


require 'torch'
require 'optim'
require 'layers.DataLoaderReg'
require 'layers.SpatialGridSrch'
require 'script.trainer'
local opts = require 'misc.opts'

-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

local loader = DataLoaderReg{data_t7 = opt.data_t7, img_h5 = opt.img_h5, data_json = opt.data_json, data_h5 = opt.data_h5, batch_size = opt.batch_size}


-------------------- GET FEATURE FROM PRETRIANED MODEL ------------------------------
-- load pretrained model
local model_num = opt.pretrain_modelIter			-- EXP1: 55726, EXP2: 7599, EXP3: 81040, EXP4:10130, EXP5:60780
local exp = opt.expName
local model_path = 'repo/pretrain/' .. exp  .. '/'.. exp  ..'_Iter' .. model_num .. '.t7'
local pretrain_model = torch.load(model_path)
local model = pretrain_model['model']
model:evaluate()





-------------------- DEFINE NETWORK FOR REGRESSION  ------------------------------
-- define network (3-layered fully connected)
local channel = 3
local units = 4			-- units per each grid section.
local grid = 5			-- total points = grid x grid x units
local ndim = 40
local batchsize = opt.batchSize
local reg = nn.Sequential()

-- 1-st layer
reg:add(nn.SpatialGridSrch(grid, units))
reg:add(nn.Linear(channel*grid*grid*units, 256, true))
reg:add(nn.BatchNormalization(256))
reg:add(nn.ReLU(true))

-- 2-nd layer
reg:add(nn.Linear(256, 32, true))
reg:add(nn.BatchNormalization(32))
reg:add(nn.ReLU(true))

-- 3-rd layer
reg:add(nn.Linear(32, 10, true))

-- MSE Criterion
local crit = nn.MSECriterion()
reg:cuda()
crit:cuda()

----------------------------- FORWARD AND BACKWARD / UPDATE PARAMS -------------------
-- get model parameters
local params, gradParams = reg:getParameters()
local optim_state = {}

-- training.
local epoch = 50
local totalIter = loader:getTotalIter()
local lr = 0.005
local snapshot_every = 2000
local loss_history = {}
reg:training()

function avg_filter(loss, data, nelement)
	if #loss < nelement then
		table.insert(loss, data)
	else
		table.remove(loss, 1)
		table.insert(loss, data)
	end

	local sum = 0
	for i = 1, #loss do
		sum = sum + loss[i]
	end
	sum = sum/#loss

	return loss, sum
end

local cnt = 1
for epoch = 1, epoch do
	print('==> Training epoch # ' .. epoch .. '/ Total Iter for 1 epoch : ' .. totalIter)
	for iter = 1, totalIter do
		cnt = cnt + 1

		-- forward and get output feature (u,v)
		local data = loader:get_batch(opt.batch_size)[1]
		local feature = model:forward({data[1]:cuda(), data[2]:cuda()})
		
		-- forward
		local input = {feature[1]:cuda(), feature[2]:cuda(), data[3]:cuda()}
		local predict = reg:forward(input)
		local loss = crit:forward(predict, data[4]:cuda())
		local avg_loss = loss/opt.batch_size
		local loss_history, avg_loss_f = avg_filter(loss_history, avg_loss, 5)
	
		-- backward
		reg:zeroGradParameters()
		local d_crit = crit:backward(predict, data[4]:cuda())
		local d_dummy = reg:backward(input, d_crit)

		-- update params
		rmsprop(params, gradParams, lr, 0.99, 1e-8, optim_state)
		print('['.. cnt  ..']' .. '  Epoch : ' .. epoch .. '  Iter : ' .. iter .. '  Loss : ' .. avg_loss_f)

		-- savesnapshot
		if (cnt % snapshot_every == 0 and cnt >= 50) then
			local save_path = 'repo/regressor/reg_M'.. model_num .. 'Iter' .. cnt .. '.t7'
			torch.save(save_path, reg)
		end

		if (cnt > 50000) then
			break
		end
	end
	if (cnt > 50000) then
		break
	end
end



