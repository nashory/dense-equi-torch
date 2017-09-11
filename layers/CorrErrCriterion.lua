-- criterion for loss definition.

require 'sys'
require 'image'
require 'layers.DenseCorrDistLoss'

local CorrErrCriterion, parent = torch.class('nn.CorrErrCriterion', 'nn.Criterion')

function CorrErrCriterion:__init(batchsize)
	parent.__init(self)
	self.split1 = nn.SplitTable(1)
	self.split2 = nn.SplitTable(1)
	self.split3 = nn.SplitTable(1)
	
	self.batchsize = batchsize
	self.batch = {}

	self.parallel = nn.ParallelTable()
	for idx = 1, batchsize do
		self.parallel:add(nn.DenseCorrDistLoss(1))
	end
	
	self.model = nn.Sequential()
	self.model:add(self.parallel)
	self.model:add(nn.JoinTable(1))
end

-- input[1] : feat_u (batchsize x 3 x H x W)
-- input[2] : feat_v (batchsize x 3 x H x W)
-- input[3] : g_matrix (batchsize x 2 x H x W)
function CorrErrCriterion:updateOutput(input)
	-- get parameters
	self.L = input[1]:size(2)
	self.W = input[1]:size(3)
	self.feat_u = input[1]
	self.feat_v = input[2]
	self.g_matrix = input[3]

	-- split w.r.t each batch. {batchsize x C x H x W} --> batchsize x {C x H x W}
	self.u_out = self.split1:forward(self.feat_u)
	self.v_out = self.split2:forward(self.feat_v)
	self.g_out = self.split3:forward(self.g_matrix)

	self.batch = {}
	for k = 1, self.batchsize do
		table.insert(self.batch, {self.u_out[k], self.v_out[k], self.g_out[k]})
	end
	
	-- forward DensCorrDistLoss
	self.loss = self.model:forward(self.batch)
	--local avgloss = self.loss:clone():sum()/(self.batchsize*self.L*self.W)
	
	return self.loss
end

function CorrErrCriterion:updateGradInput(input, gradOutput)

	self.gradInput = {}
	self.dLoss = self.model:backward(self.batch, gradOutput)

	self.dfeat_u = {}
	self.dfeat_v = {}
	for i = 1, self.batchsize do
		self.dfeat_u[i] = self.dLoss[i][1]
		self.dfeat_v[i] = self.dLoss[i][2]
	end

	self.gradInput[1] = self.split1:backward(self.feat_u, self.dfeat_u)
	self.gradInput[2] = self.split2:backward(self.feat_v, self.dfeat_v)

	return self.gradInput
end


