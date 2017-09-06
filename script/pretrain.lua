-- training code.
-- this code is based on ResNet code of Facebook, Inc.
-- last modified : 2017.07.14, nashory


require 'torch'
require 'cutorch'
require 'optim'
require 'nn'
require 'layers.DataLoader'
require 'script.trainer'
local opts = require 'script.opts'
local net = require 'models.simple_conv'
require 'layers.CorrErrCriterion'


-- basic settings.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
print(opt)
torch.manualSeed(opt.seed)
cutorch.manualSeedAll(opt.seed)

local loader = DataLoader()

-- model and criterion --
if (opt.restart) then
	snapshot = torch.load('repo/save/pretrain/' .. opt.start_from .. '.t7')
	model = snapshot['model']
	optimState = snapshot['optimState'] 
else
	model = net()
end


-- set criterion
local criterion = nn.CorrErrCriterion(opt.batchSize)

-- trainer --
local trainer = Trainer(model, criterion, optimState)

-- run trainer for N epoch.
trainer:train(100, loader)




