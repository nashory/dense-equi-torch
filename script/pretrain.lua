-- training code.
-- this code is based on ResNet code of Facebook, Inc.
-- last modified : 2017.07.14, nashory


require 'torch'
require 'optim'
require 'nn'
require 'script.trainer'
local opts = require 'script.opts'
local net = require 'models.simple_conv'
require 'layers.CorrErrCriterion'


-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
print(opt)
if opt.seed == 0 then opt.seed = torch.random(1,9999) end
torch.manualSeed(opt.seed)
print(string.format('Seed : %d', opt.seed))

-- set if use gpu
if opt.gpuid >=0 then
	require 'cutorch'
	require 'cunn'
	require 'cudnn'
	cutorch.manualSeedAll(opt.seed)
	cutorch.setDevice(opt.gpuid+1)
end


-- create dataloader.
local loader = paths.dofile('../data/data.lua')

-- model and criterion --
if (opt.restart) then
	snapshot = torch.load('repo/pretrain/' .. opt.start_from .. '.t7')
	model = snapshot['model']
	optim_state = snapshot['optim'] 
else
	model = net()
end
local criterion = nn.CorrErrCriterion(opt.batchSize)


-- trainer --
local trainer = Trainer(model, criterion, opt, optim_state)

-- run trainer for N epoch.
trainer:train(1000, loader)

print('Congrats! You just finisehd the training.')


