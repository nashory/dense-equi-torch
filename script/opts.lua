-- Training parameter settings.


require 'nn'
require 'torch'
require 'optim'

local M = { }

function M.parse(arg)
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Training code.')
	cmd:text()
	cmd:text('Input arguments')


	---------------- General options ---------------
	cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
	cmd:option('-seed', 123, 'random number generator seed to use')
	cmd:option('-backend', 'cudnn', 'cudnn option.')
	cmd:option('-save_pretrain_path', 'repo/pretrain', 'path to save the model.')
	cmd:option('-snapshot_every', 1, 'save model every XX epoch.')
	cmd:option('-start_from', 'EXP5/3-conv_Iter10130', 'model name to restart training.')
	cmd:option('-restart', false, 'if you want to restart from specified model, change is to true.')
	cmd:option('-visualize', true, 'true: plot training loss / false: no plot')
	
	---------------- Data path options ---------------
	cmd:option('-data_root_train', '/home1/irteam/nashory/data/domain')
	cmd:option('-nthreads', 8, '# of workers to use for data loading.')

	----------------- Generating tps matrix ----------
	cmd:option ('-tps_h5', 'prepro/save/tps.h5', 'tps file path to generate')
	cmd:option('-tpslen', 1000, '# of tps matrix to generate.')
	cmd:option('-name', 'simple-conv', 'experiment name')


	cmd:option('-sampleSize', 64, 'crop size.')
	cmd:option('-loadSize', 80, 'load size.')
	cmd:option('-outputSize', 40, 'output feature map size.')


	-------------- Training options---------------
	cmd:option('-batchSize', 40, 'batch size for training')
	cmd:option('-lr', 0.0005, 'learning rate')
	cmd:option('-momentum', 0.997, 'momentum')
	cmd:option('-weight_decay', 0.05, 'weigth decay')
	cmd:text()



	-- return opt.
	local opt = cmd:parse(arg or {})
	return opt
end

return M


