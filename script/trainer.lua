
-- For training loop and learning rate scheduling.
-- This code is based on ResNet of Facebook, Inc.
-- last modified : 2017.07.21, nashory


require 'sys'
require 'optim'
require 'script.optim_updates'
require 'image'
require 'math'
require 'tools.mcvis'
local utils = require 'tools.mcutils'
local gp = require 'gnuplot'
local opts = require 'script.opts'
local opt = opts.parse(arg)


local Trainer = torch.class('Trainer')


function Trainer:__init(model, criterion, optimState)
	self.model = model
	self.criterion = criterion
	self.optimState = optimState or {
		lr = opt.lr,
		momentum = opt.momentum,
		weight_decay = opt.weight_decay,
	}
	self.opt = opt

	-- visualizer
	self.vis = Visualizer(gp)
end


function Trainer:train(epoch, dataloader)
	-- Trains the model for specified epoch
	local timer = torch.Timer()
	local dataTimer = torch.Timer()

	local function feval()
		return self.criterion.output, self.gradParams
	end

	---- ship everything to cuda  & get parameters.-----
	self.model:cuda()
	self.criterion:cuda()
	self.params, self.gradParams = self.model:getParameters()
	

	optim_state = {}
	self.model:training()
	self.totalIter = dataloader:getTotalIter()

	local loss_history = {}
	local iter_history = {}


	self.globalIter = 0
	for epoch = 1, epoch do
		print('==> Training epoch # ' .. epoch .. '/ Total Iter for 1 epoch : '.. self.totalIter .. ' / snapshot every: ' .. math.ceil(self.totalIter*opt.snapshot_every)..' iter.' )
		for iter = 1, self.totalIter do
			self.globalIter = self.globalIter + 1
		
			-- prepare training batch -- 
			local data = dataloader:get_batch(opt.batchSize)[1]
			local inputdata = {data[1]:cuda(), data[2]:cuda()}
			local output = self.model:forward(inputdata)

			-- forward criterion --
			local inputcrit = {output[1]:cuda(), output[2]:cuda(), data[3]:cuda()}
			local loss = self.criterion:forward(inputcrit)
			local avg_loss = loss:clone():sum()/opt.batchSize/(opt.outputSize*opt.outputSize)

			-- backward criterion and model--
			self.model:zeroGradParameters()
			local d_crit = self.criterion:backward(inputcrit, loss)
			local d_model = self.model:backward(inputdata, d_crit)

			-- update parameters.
			--rmsprop(self.params, self.gradParams, self.optimState.lr, 0.99, 1e-8, optim_state)
			adam(self.params, self.gradParams, self.optimState.lr, 0.9, 0.998, 1e-8, optim_state)
			print ('Epoch  ' .. epoch .. ', Iter ' .. iter .. ', lr:' .. self.optimState.lr .. ' ==> Avg Loss : ' .. avg_loss)
			-- adjust learning rate
			self.optimState.lr = self:AdjustLR(self.optimState.lr, self.globalIter)
				
			-- snapshot --	
			local data = {params = self.params, model = self.model, optimState = self.optimState}
			self:snapshot(opt.save_pretrain_path, opt.name, data)
	

			-- visualizer (graphical plot)
			self.vis:setTitle('Train loss history')
			self.vis:arrange(5000)
			self.vis:insert(self.globalIter, avg_loss)
			self.vis:plot()			
		

			-- save .png for bckup --
			if iter%50 == 0 then
				self.vis:savepng('repo/output.png')
				--gnuplot.figure(1)
				--gnuplot.title('Loss history')
				--gnuplot.plot(torch.Tensor(iter_history), torch.Tensor(loss_history), '-')
				--gnuplot.pngfigure('repo/output.png')
				--gnuplot.plot(torch.Tensor(iter_history), torch.Tensor(loss_history), '-')
				--gnuplot.plotflush()
			end
		end
	end

end


function Trainer:snapshot(path, fname, data)
	local fname = fname .. '_Iter' .. self.globalIter .. '.t7'

	if (self.globalIter % math.ceil(self.opt.snapshot_every*self.totalIter) == 0) then
		local save_path = path .. '/' .. fname
		torch.save(save_path, data)
		print('[GO] ==> save model @ ' .. save_path)
	end
end

function Trainer:AdjustLR(lr, iter)
	if iter == 20000 then
		lr = lr * 0.5
	end

	return lr
end








