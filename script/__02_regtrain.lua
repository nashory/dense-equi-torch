-- training face landmark regressor.
-- Last modified : 2017.08.07, nashory


require 'torch'
require 'optim'
require 'layers.DataLoaderReg'
require 'layers.SpatialGridSrch'
require 'script.trainer'
local opts = require 'script.opts'

-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)

local loader = DataLoaderReg()


-- forward pretrained model.
local model_num = opt.pretrain_modelIter
local exp = opt.targName
local model_path = 'repo/pretrain/' .. exp  .. '/'.. exp  ..'_Iter' .. model_num .. '.t7'
local pretrain_model = torch.load(model_path)
local model = pretrain_model['model']
model:cuda()
model:evaluate()


-- network structure for regression (mlp)
local channel = 3
local units = 4			-- units per each grid section.
local grid = 5			-- total points = grid x grid x units
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


-- forward and backward
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
		local loss_history, avg_loss = avg_filter(loss_history, loss/opt.batchSize, 5)
	
		-- backward
		reg:zeroGradParameters()
		local d_crit = crit:backward(predict, data[4]:cuda())
		local d_dummy = reg:backward(input, d_crit)

		-- update params
		rmsprop(params, gradParams, lr, 0.99, 1e-8, optim_state)
        print(string.format('[%d]\tEpoch:%d\tIter:%d\tLoss:.4f', cnt, epoch, iter, avg_loss))

		-- savesnapshot
		if (cnt % snapshot_every == 0 and cnt >= 50) then
            os.execute('mkdir -p repo/regressor/' .. opt.expName)
			local save_path = 'repo/regressor/reg_P'.. model_num .. '_R' .. cnt .. '.t7'
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



