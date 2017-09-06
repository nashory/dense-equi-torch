-- save images into t7 format.

require 'torch'
require 'image'
local cjson = require 'cjson'
local path = require 'paths'
local utils = require 'mcutils'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('compress training images into .t7 format.')
cmd:text()

cmd:option('-imlist', 'save/imlist.txt',  'ldmk annotation file.')
cmd:option('-prepro_data', 'save/prepro_data.json',  'prepro_data.json')
cmd:option('-save_img_t7', 'save/prepro_img_anno_50K.t7', 'where to save the final database file (.t7)')
cmd:option('-imlim', 50000, 'you can pick up to N number of images. (mostly for making small size db for test. set -1 to pick entire images.)')


local opt = cmd:parse(arg)

-- load json file to restore parameters.
print('Loading json file...' .. opt.prepro_data)
local json_file = utils.read_json(opt.prepro_data)
print(json_file)

-- read annotation .txt file.
local imlist = opt.imlist
local imdataroot = json_file._data_root
local imsize = json_file._imsize
local imlim = opt.imlim
local lines = utils.readlines_txt(imlist)


local dataset = {}

for idx = 1, #lines do
	if idx > 0 then
		local cur = lines[idx]
		local info = {}
		for i in string.gmatch(cur, "%S+") do
			table.insert(info, i)
		end
		
		-- load image.
		im = image.load(imdataroot .. '/' .. info[1])
		crop = im:clone()

		-- crop and adjust ldmk pos.
		local x1 = tonumber(info[12])
		local y1 = tonumber(info[13])
		local x2 = tonumber(info[14])
		local y2 = tonumber(info[15])
		local w = im:size(3)
		local h = im:size(2)
		if ((x2-x1)<=w) and ((y2-y1)<=h) and (x1>=0) and (x2<w) and (y1>=0) and (y2<h) then
			image.crop(crop, im, x1, y1, x2, y2)
			for i = 2,11,2 do
				info[i] = info[i] - info[12]
				info[i] = info[i]*im:size(3)/(info[14]-info[12])
			end
			for i = 3,11,2 do
				info[i] = info[i] - info[13]
				info[i] = info[i]*im:size(2)/(info[15]-info[13])
			end
		end

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
		local display = image.drawRect(im, l_eye_x, l_eye_y, l_eye_x+1, l_eye_y+1)
		display = image.drawRect(display, r_eye_x, r_eye_y, r_eye_x+1, r_eye_y+1)
		display = image.drawRect(display, nose_x, nose_y, nose_x+1, nose_y+1)
		display = image.drawRect(display, l_mouth_x, l_mouth_y, l_mouth_x+1, l_mouth_y+1)
		display = image.drawRect(display, r_mouth_x, r_mouth_y, r_mouth_x+1, r_mouth_y+1)
		image.save('anno/' .. idx .. '.png', display)
		

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
		print('[' .. idx .. '] ' .. 'now processing:	' .. info[1] .. '	(' .. (#lines-idx) .. ' images left.)' )

		if (imlim~=-1) and (idx >= imlim) then
			break
		end
	end
end


torch.save(opt.save_img_t7, dataset)
print('successfully compressed images into .t7')

