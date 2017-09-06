-- network model definition
-- last modified : 2017.07.18, nashory



local nn = require 'nn'
require 'cunn'
require 'cudnn'

-- Usage--
-- module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
-- module = nn.SpatialAveragePooling(kW, kH [, dW, dH, padW, padH])
-- module = nn.SpatialMaxPooling(kW, kH [, dW, dH, padW, padH])
-- module = nn.SpatialBatchNormalization(N [,eps] [, momentum] [,affine])


local Convolution = cudnn.SpatialConvolution
local AvgPool = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local LeakyReLU = nn.LeakyReLU
local MaxPool = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local SSoftMax = nn.SpatialSoftMax
local Sigmoid = nn.Sigmoid
local Tanh = nn.Tanh
local Add = nn.Add


local function build_model(opt)
	---- build model as you want ----
	-- Note that last layer must be Conv-layer.
	-- Conv-layer should not have bias when followed by BN.
	-- Both Avgpool / Maxpool show competitive performance.
	-- By default, torch initializes the weights with uniform distribution.

	-- conv 1 (5x5)
	local convnet = nn.Sequential()
	convnet:add(Convolution(3, 20, 5, 5, 1, 1):noBias())		-- (84 x 84 x 3) ==> (80 x 80 x 20)
	convnet:add(SBatchNorm(20))						-- (80 x 80 x 20) ==> (80 x 80 x 20)
	convnet:add(ReLU(true))							-- (80 x 80 x 20) ==> (80 x 80 x 20)
	convnet:add(MaxPool(2, 2, 2, 2))				-- (80 x 80 x 3) ==> (40 x 40 x 20)	
	
	-- conv2 (3x3)
	convnet:add(Convolution(20, 48, 3, 3, 1, 1, 1, 1):noBias()) -- (40 x 40 x 20) ==> (40 x 40 x 48)
	convnet:add(SBatchNorm(48))						-- (40 x 40 x 48) ==> (40 x 40 x 48)
	convnet:add(ReLU(true))							-- (40 x 40 x 48) ==> (40 x 40 x 48)
	
	-- conv3 (3x3)
	convnet:add(Convolution(48, 64, 3, 3, 1, 1, 1, 1):noBias())	-- (40 x 40 x 48) ==> (40 x 40 x 64)
	convnet:add(SBatchNorm(64))						-- (40 x 40 x 64) ==> (40 x 40 x 64)
	convnet:add(ReLU(true))							-- (40 x 40 x 64) ==> (40 x 40 x 64)
	
	-- conv4 (3x3)
	convnet:add(Convolution(64, 80, 3, 3, 1, 1, 1, 1):noBias()) -- (40 x 40 x 64) ==> (40 x 40 x 80)
	convnet:add(SBatchNorm(80))						-- (40 x 40 x 80) ==> (40 x 40 x 80)
	convnet:add(ReLU(true))							-- (40 x 40 x 80) ==> (40 x 40 x 80)
	
	-- conv5 (1x1)
	convnet:add(Convolution(80, 256, 1, 1, 1, 1):noBias()) -- (40 x 40 x 80) ==> (40 x 40 x 256)
	convnet:add(SBatchNorm(256))						-- (40 x 40 x 256) ==> (40 x 40 x 256)
	convnet:add(ReLU(true))							-- (40 x 40 x 256) ==> (40 x 40 x 256)
	
	-- conv6 (1x1)
	convnet:add(Convolution(256,3, 1, 1, 1, 1):noBias()) -- (40 x 40 x 256) ==> (40 x 40 x 3)


	-- true flag : share memory.
	local siamese = nn.MapTable(convnet, true)
	
	local model = nn.Sequential()
	model:add(siamese)

	print(model)
	return model
end

return build_model





