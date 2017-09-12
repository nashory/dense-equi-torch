-- save images into t7 format.

require 'torch'
require 'image'
local xlua = require 'xlua'
local path = require 'paths'
local utils = require 'tools.mcutils'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('compress training/test images for regressor into .t7 format.')
cmd:text()


-- please change these when you start training.
cmd:option('-imdataroot', '/home1/work/nashory/data/CelebA/Img/img_align_celeba')
cmd:option('-imlist', '/home1/work/nashory/data/CelebA/Anno/list_landmarks_align_celeba.txt',  'CelebA annotation file path.')
cmd:option('-reg_train_t7', 'data/save/reg_train.t7', 'path to save training image for regressor')
cmd:option('-reg_test_t7', 'data/save/reg_test.t7', 'path to save test image for regressor')
cmd:option('-imlim', 200, 'you can pick up to N number of images. (mostly for making small size db for test. set -1 to pick entire images.)')
cmd:option('-ratio', 0.7, 'train image / test image')
cmd:option('-imsize', 96, 'resize image')

local opt = cmd:parse(arg)


-- read annotation .txt file.
local imlist = opt.imlist
local imsize = opt.imsize
local lines = utils.readlines_txt(imlist)
local imlim = opt.imlim
local imdataroot = opt.imdataroot


local dataset = {}
local trainset = {}
local testset = {}

for i = 1, #lines do
    idx = math.random(3, #lines)
	local cur = lines[idx]
	local info = {}
	for i in string.gmatch(cur, "%S+") do
		table.insert(info, i)
	end
		
	-- load image.
	im = image.load(imdataroot .. '/' .. info[1])
	crop = im:clone()

	-- resize image and adjust ldmk pos.
	out = utils.resize_with_padding(crop, imsize)
	
	local im = out[1]
	local offset_h = out[2]
	local offset_w = out[3]
	local len = out[4]	
	for i = 2,11,2 do
		info[i] = info[i] + offset_w
		info[i] = info[i]*imsize/len
	end
	for i = 3,11,2 do
		info[i] = info[i] + offset_h
		info[i] = info[i]*imsize/len
	end
		
	-- save info.
	local img = info[1]
	local l_eye_x = info[2]
	local l_eye_y = info[3]
	local r_eye_x = info[4]
	local r_eye_y = info[5]
	local nose_x = info[6]
	local nose_y = info[7]
	local l_mouth_x = info[8]
	local l_mouth_y = info[9]
	local r_mouth_x = info[10]
	local r_mouth_y = info[11]
	

	-- display (uncomment this part to see if the image is cropped properly.)
    -- os.execute('mkdir -p anno/')
	--local display = image.drawRect(im, l_eye_x, l_eye_y, l_eye_x+1, l_eye_y+1)
	--display = image.drawRect(display, r_eye_x, r_eye_y, r_eye_x+1, r_eye_y+1)
	--display = image.drawRect(display, nose_x, nose_y, nose_x+1, nose_y+1)
	--display = image.drawRect(display, l_mouth_x, l_mouth_y, l_mouth_x+1, l_mouth_y+1)
	--display = image.drawRect(display, r_mouth_x, r_mouth_y, r_mouth_x+1, r_mouth_y+1)
	--image.save('anno/' .. idx .. '.png', display)
		

	-- save informations in table.
	local data = {}
	data['data'] = im
	data['l_eye_x'] = l_eye_x
	data['l_eye_y'] = l_eye_y
	data['r_eye_x'] = r_eye_x
	data['r_eye_y'] = r_eye_y
	data['nose_x'] = nose_x
	data['nose_y'] = nose_y
	data['l_mouth_x'] = l_mouth_x
	data['l_mouth_y'] = l_mouth_y
	data['r_mouth_x'] = r_mouth_x
	data['r_mouth_y'] = r_mouth_y
	table.insert(dataset, data)
    if imlim == -1 then imlim = #lines end
    xlua.progress(i, imlim)


	if (imlim~=-1) and (i >= imlim) then
		break
	end
end

print('creating train/test set ...')
for i = 1, #lines do
	if i < math.ceil(opt.ratio*#lines) then	table.insert(trainset, dataset[i])
	else table.insert(testset, dataset[i]) end
end

torch.save(opt.reg_train_t7, trainset)
torch.save(opt.reg_test_t7, testset)
print('successfully compressed images into .t7')

