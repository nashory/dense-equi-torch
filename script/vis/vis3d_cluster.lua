-- visualize mappings of pretrained model on 3-dimensional latent space.
require 'nn'
require 'cunn'
require 'cudnn'
require 'torch'
require 'cutorch'
require 'layers.DataLoaderReg'
require 'math'

local net = require 'models.nets_uni'
local opts = require 'misc.opts'
local npy4th = require 'npy4th'
require 'image'

local num_img = 500
local trgidx = {}
for i = 1, 500 do
	local idx = math.random(1, 15000)
	table.insert(trgidx, idx)
end

print(trgidx)


-- get options
local opt = opts.parse(arg)

-- load sample data
local loader = DataLoaderReg{data_t7=opt.data_t7, data_json = opt.data_json, data_h5 = opt.data_h5, batch_size = opt.batch_size}



-- specify this.
local iter = {60780}
local exp = 'EXP1'


for k, v in pairs(iter) do
	local model_path = 'repo/pretrain/' .. exp  .. '/Model_Iter' .. v .. '.t7'
	print('Loading pretrained model... '  ..  model_path)


	-- load the pretrained weights and apply.
	local snapshot = torch.load(model_path)
	local model = snapshot['model']
	model:cuda()
	model:evaluate()

	local feat_le = torch.CudaTensor(3, #trgidx):zero()
	local feat_re = torch.CudaTensor(3, #trgidx):zero()
	local feat_no = torch.CudaTensor(3, #trgidx):zero()
	local feat_lm = torch.CudaTensor(3, #trgidx):zero()
	local feat_rm = torch.CudaTensor(3, #trgidx):zero()

	local cnt = 1
	for key, idx in pairs(trgidx) do
		print('['.. key .. ']' .. ' processing image iid : ' .. idx)
		local data = loader:get_sample(idx)

		-- prepare input data and forward.
		local inputdata = {data[1]:cuda(), data[2]:cuda()}
		local output = model:forward(inputdata)

		-- get feature vector of each ldmk.
		local anno = torch.ceil(data[4]:div(2):squeeze())
		for i = 1, 10 do
			if anno[i] > 40 then 
				anno[i] = 40
			end
			if anno[i] < 1 then
				anno[i] = 1
			end
		end
		output[1] = output[1]:squeeze()	
		feat_le[{{},key}] = output[1][{{},anno[2],anno[1]}]
		feat_re[{{},key}] = output[1][{{},anno[4],anno[3]}]
		feat_no[{{},key}] = output[1][{{},anno[6],anno[5]}]
		feat_lm[{{},key}] = output[1][{{},anno[8],anno[7]}]
		feat_rm[{{},key}] = output[1][{{},anno[10],anno[9]}]
	end

	npy4th.savenpy('repo/vis/Iter'.. v ..'_feat_le.npy', feat_le)
	npy4th.savenpy('repo/vis/Iter'.. v ..'_feat_re.npy', feat_re)
	npy4th.savenpy('repo/vis/Iter'.. v ..'_feat_no.npy', feat_no)
	npy4th.savenpy('repo/vis/Iter'.. v ..'_feat_lm.npy', feat_lm)
	npy4th.savenpy('repo/vis/Iter'.. v ..'_feat_rm.npy', feat_rm)

	print('save success.')
end











