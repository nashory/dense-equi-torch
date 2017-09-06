-- criterion for loss definition.


require 'sys'
require 'image'
require 'math'

local SpatialGridSrch, parent = torch.class('nn.SpatialGridSrch', 'nn.Module')

function SpatialGridSrch:__init(grid, units)
	parent.__init(self)

	-- params setting.
	self.units = units
	self.grid = grid
	self.total = self.units * self.grid * self.grid
	self.softmax = nn.SoftMax()
end


function SpatialGridSrch:meshgrid(w)
	local mesh = torch.CudaTensor(2,w,w):zero()
	for i = 1, w do
		mesh[{{},{},i}] = torch.CudaTensor(2,w):fill(i)
	end
	mesh[{1,{},{}}] = mesh[{1,{},{}}]:transpose(2,1)
	-- mesh : 2 x self.H x self.W
	return mesh
end


-- input[1] : feat_u (batch x L x H x W)
-- input[2] : feat_v (batch x L x H x W)
-- input[3] : g_matrix (batch x 2 x H x W)
function SpatialGridSrch:updateOutput(input)
	self.batchsize = input[1]:size(1)
	self.L = input[1]:size(2)
	self.H = input[1]:size(3)
	self.W = input[1]:size(4)
	self.P = self.H * self.W

	local keypoints = torch.CudaTensor(self.batchsize, self.grid, self.grid, self.units, self.L):zero()			-- (batch x 3 x 50)
	local interval = math.floor(self.H / self.grid)

	for nbatch = 1, self.batchsize do
		local feat_u = input[1][{nbatch, {},{},{}}]:view(self.L, -1)			-- (L x HW)
		local feat_v = input[2][{nbatch, {},{},{}}]:view(self.L, -1)			-- (L x HW)
		local g_matrix = input[3][{nbatch, {},{},{}}]							-- (2 x H x W)

		-- calculate correlation between feat_u and feat_v
		local ip = torch.mm(feat_u:transpose(1,2), feat_v)				-- 1600(u) x 1600(v)
		local soft = self.softmax(ip)									-- 1600(u) x 1600(v)
		
		-- get top-N units with largest correlation.
		local maxcorr = torch.max(soft, 2):squeeze():view(self.H, self.W)		-- 40(u) x 40(u)
		
		-- get top-N units w.r.t each grid section.
		local spatial_topidx = torch.Tensor(self.grid*self.grid, self.units):zero()
		feat_u = feat_u:view(-1, self.H, self.W)							-- 3 x 40 x 40
		for gw = 1, self.grid do
			for gh = 1, self.grid do
				local grid_soft = soft[{{1+(gh-1)*interval, (gh)*interval}, {1+(gw-1)*interval, (gw)*interval}}]
				local grid_feat = feat_u[{{},{1+(gh-1)*interval, (gh)*interval}, {1+(gw-1)*interval, (gw)*interval}}]
				grid_soft = grid_soft:contiguous():view(-1)
				grid_feat = grid_feat:contiguous():view(self.L, -1)
				topval, topidx = torch.topk(grid_soft, self.units, 1, true)
				for k = 1, self.units do
					keypoints[{nbatch,gh,gw,k,{}}]:copy(grid_feat[{{},topidx[k]}]:squeeze())
				end
			end
		end
	end

	keypoints = keypoints:view(self.batchsize, -1)		-- batch x L x units
	return keypoints

end

function SpatialGridSrch:updateGradInput(input, gradOutput)
	-- no backpropagation is needed.
	return -1
end

