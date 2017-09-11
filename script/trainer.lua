
-- For training loop and learning rate scheduling.
-- last modified : 2017.07.21, nashory


require 'sys'
require 'optim'
require 'script.optim_updates'
require 'image'
require 'math'
require 'hdf5'


local Trainer = torch.class('Trainer')


function Trainer:__init(model, criterion, opt, optim_state)
	self.model = model:cuda()
	self.criterion = criterion:cuda()

	-- load options
	self.opt = opt
	self.batchSize = opt.batchSize
	self.sampleSize = opt.sampleSize
	self.outputSize = opt.outputSize
	
	-- optimizer setting.
	self.optim_state = optim_state or {
		method = adam,
		config = {
			lr = opt.lr,
			beta1 = 0.9,
			beta2 = 0.997,
			elipson = 1e-8
		},
		optimstate = {}
	}

	if opt.display then
		self.disp = require 'display'
		self.disp.configure({hostname=opt.display_server_ip, port=opt.display_server_port})
	end
end

function Trainer:refreshTPS()
	os.execute('python data/gen_tps.py')
	local tps = {}
	local h5_file = hdf5.open('data/save/tps.h5')
	table.insert(tps, h5_file:read('/_wfx'):all())
	table.insert(tps, h5_file:read('/_wfy'):all())
	return tps
end

function Trainer:getTPS()
	local g = torch.Tensor(self.batchSize, 2, self.sampleSize, self.sampleSize):zero()
	local g_adj = torch.Tensor(self.batchSize, 2, self.outputSize, self.outputSize):zero()
	local indices = torch.range(1, self.tps[1]:size(1))
	-- load tps matriz and squeeze into batch.
	for b = 1, self.batchSize do
		local idx = indices[math.random(1, self.tps[1]:size(1))]
		g[{{b},{1},{},{}}] = self.tps[1][{{idx},{},{}}]:clone()
		g[{{b},{2},{},{}}] = self.tps[2][{{idx},{},{}}]:clone()
		g_adj[{{b},{1},{},{}}] = image.scale(self.tps[1][{{idx},{},{}}], self.outputSize, self.outputSize):div(self.sampleSize/self.outputSize)
		g_adj[{{b},{2},{},{}}] = image.scale(self.tps[2][{{idx},{},{}}], self.outputSize, self.outputSize):div(self.sampleSize/self.outputSize)
	end
	return g, g_adj
end


function Trainer:applyWarp(_bim, _bg)
	local warpim = torch.Tensor(self.batchSize, 3, self.sampleSize, self.sampleSize):zero()
	for b = 1, self.batchSize do
		warpim[{{b},{},{},{}}] = image.warp(_bim[{{b},{},{},{}}]:squeeze(), _bg[{{b},{},{},{}}]:squeeze())
	end
	return warpim
end



Trainer['fDx'] = function(self)
	self.model:zeroGradParameters()
	-- prepare input
	self.x = self.dataset:getBatch()
	self.g, self.g_adj = self:getTPS()
	self.x_tilde = self:applyWarp(self.x:float(), self.g:float())
	-- forward and backward
	local output = self.model:forward({self.x:cuda(), self.x_tilde:cuda()})
	local loss = self.criterion:forward({output[1], output[2], self.g_adj:cuda()})
	local avgloss = loss:sum()/(self.batchSize*self.outputSize*self.outputSize)
	local d_criterion = self.criterion:backward({output[1], output[2], self.g_adj:cuda()}, loss)
	local d_dummy = self.model:backward({self.x:cuda(), self.x_tilde:cuda()}, d_criterion)

	return avgloss
end


function Trainer:train(epoch, loader)
	-- init variables.
	self.tps = self.refreshTPS()
	self.g = torch.Tensor(self.batchSize, 2, self.sampleSize, self.sampleSize):zero()
	self.g_adj = torch.Tensor(self.batchSize, 2, self.outputSize, self.outputSize):zero()


	-- load dataset and get model params.
	self.dataset = loader.new(self.opt.nthreads, self.opt)
	print(string.format('Dataset size : %d', self.dataset:size()))
	self.model:training()
	self.params, self.gradParams = self.model:getParameters()
	
	local timer = torch.Timer()			-- timer starts to count now.
	local totalIter = 0
	local loss_his = {}

	-- do training.
	for e = 1, epoch do
		local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
		for iter = 1, iter_per_epoch do
			totalIter = totalIter+1

			-- forward/backward and update weights with optimizer.
			local loss = self:fDx()

			-- weight update.
			-- self.optim_state.config.lr = adjust_lr(iter)
			self.optim_state.method(self.params, self.gradParams, self.optim_state.config.lr,
									self.optim_state.config.beta1, self.optim_state.config.beta2,
									self.optim_state.config.elipson, self.optim_state.optimstate)
			
			-- save model at every specified epoch.
			local data = {model = self.model, optim = self.optim_state}
			self:snapshot(string.format('repo/pretrain/%s', self.opt.name), self.opt.name, totalIter, data)

			-- logging.
			local log_msg = string.format('Epoch: [%d][%6d/%6d]\tLoss: %.4f\tTime elpase: %.1f(min)', e, iter, iter_per_epoch, loss, timer:time().real/60.0)
			print(log_msg)
			
			-- display(iamge, plot).
			if self.opt.display then
				-- image.
				local im = self.x:clone()
				local im_tilde = self.x_tilde:clone()
				self.disp.image(im, {win=self.opt.display_id+self.opt.gpuid, title=self.opt.server_name})
				self.disp.image(im_tilde, {win=self.opt.display_id*5+self.opt.gpuid, title=self.opt.server_name})
				-- plot.
				local range = 10000
				local start = 1
				local last = totalIter/self.opt.display_iter
				if totalIter <= range then start = 1
				elseif totalIter > range then start = totalIter-range end
				table.insert(loss_his, {totalIter, loss})
				if #loss_his > range then table.remove(loss_his, 1) end
				self.disp.plot(loss_his, {labels={'Iter', string.format('loss(gpu:%d)', self.opt.gpuid)}, win=string.format('loss(gpu:%d)', self.opt.gpuid)})
			end
		end
	end
end


function Trainer:snapshot(path, fname, iter, data)
	-- if dir not exist, create it.
	if not paths.dirp(path) then 	os.execute(string.format('mkdir -p %s', path)) end
	local fname = fname .. '_Iter' .. iter .. '.t7'
	local iter_per_epoch = math.ceil(self.dataset:size()/self.batchSize)
	if iter % math.ceil(self.opt.snapshot_every*iter_per_epoch) == 0 then
		local save_path = path .. '/' .. fname
		torch.save(save_path)
		print(string.format('[snapshot]: saved model @$s', save_path))
	end
end

return Trainer
	

