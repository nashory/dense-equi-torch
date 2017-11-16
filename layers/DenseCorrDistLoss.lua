-- criterion for loss definition.


require 'sys'
require 'image'

local DenseCorrDistLoss, parent = torch.class('nn.DenseCorrDistLoss', 'nn.Module')

function DenseCorrDistLoss:__init(gamma)
	parent.__init(self)

	self.gamma = gamma
	self.softmax = nn.SoftMax()
end

function DenseCorrDistLoss:meshgrid(w)
	local mesh = torch.CudaTensor(2, w, w):zero()
	for i = 1, w do
		mesh[{{},{},i}] = torch.CudaTensor(2,w):fill(i)
	end
	mesh[{1,{},{}}] = mesh[{1,{},{}}]:clone():transpose(2,1)
	-- mesh : 2 x self.H x self.W
	return mesh
end


-- input[1] : feat_u (L x H x W)
-- input[2] : feat_v (L x H x W)
-- input[3] : g_matrix (2 x H x W)
function DenseCorrDistLoss:updateOutput(input)
	self.L = input[1]:size(1)
	self.H = input[1]:size(2)
	self.W = input[1]:size(3)
	self.P = self.H * self.W
	self.feat_u = input[1]:view(self.L, -1)		-- 3 x 1600
	self.feat_v = input[2]:view(self.L, -1)		-- 3 x 1600
	self.g_matrix = input[3]

	-- calculate innerproduct (feat_v * feat_u) and forward Spatial SoftMax.
	self.ip = torch.mm(self.feat_v:transpose(1,2), self.feat_u)		--> 1600(v) x 1600(u)
	self.soft = self.softmax:updateOutput(self.ip:transpose(1,2)):transpose(1,2):contiguous()
	self.soft = self.soft:view(self.H, self.W, -1)					--> 40(v) x 40(v) x 1600(u)


	-- calculate distance |v - gu|
	self.mesh = self:meshgrid(self.W)
	self.mesh2 = self.mesh:clone()
	self.gu = self.mesh2:csub(self.g_matrix):view(-1, self.P)			-- 2 x 1600(u)
	
	-- rescale (w/h ratio = 1.0)
	--self.mesh = self.mesh:div(self.W)
	--self.gu = self.gu:div(self.W)

	self.gu = torch.repeatTensor(self.gu, self.H, self.W, 1, 1)		--> 40(u) x 40(u) x 2 x 1600(u)
	self.mesh = torch.repeatTensor(self.mesh, self.P,1,1,1):permute(3,4,2,1)	--> 40(v) x 40(v) x 2 x 1600(u)
	self.diff = self.mesh:clone():csub(self.gu):pow(2)				--> 40 x 40 x 2 x 1600(u)
	self.dist = torch.add(self.diff[{{},{},1,{}}], self.diff[{{},{},2,{}}]):sqrt():pow(self.gamma) --> 40 x 40 x 1600(u)
	
	-- calculate total L = loss |v - gu|*P(v|u)
	-- 1600 tensor version.
	self.output = self.soft:clone():cmul(self.dist):view(self.P, -1):sum(1):squeeze()	-- 1600(u) tensor.
	return self.output
end



function DenseCorrDistLoss:updateGradInput(input, gradOutput)
	-- init gradInput tensor.
	self.gradInput = {}
	self.gradInput[1] = input[1]:clone():fill(0)
	self.gradInput[2] = input[2]:clone():fill(0)

	self.gradOutput = torch.repeatTensor(gradOutput, self.P, 1):contiguous():view(self.H, self.W, -1)
	self.dzdL = self.gradOutput:cmul(self.dist):view(self.P, -1):transpose(1,2):contiguous()
	self.dzdCorr = self.softmax:updateGradInput(self.ip:transpose(1,2), self.dzdL):transpose(1,2) --> 1600(v) x 1600(u)
	
	self.dzdP1 = torch.mm(self.feat_v, self.dzdCorr):view(-1, self.H, self.W)	--> L x H x W
	self.dzdP2 = torch.mm(self.dzdCorr, self.feat_u:transpose(1,2)):transpose(1,2):contiguous():view(-1, self.H, self.W)		--> L x H x W

	self.gradInput[1]:copy(self.dzdP1)
	self.gradInput[2]:copy(self.dzdP2)

	return self.gradInput
end

