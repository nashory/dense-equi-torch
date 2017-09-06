local cjson = require 'cjson'
local utils = {}
require 'nn'
require 'math'
require 'lfs'

-- for random # generation.
math.randomseed(os.time())

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end

  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  --cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

function utils.right_align(seq, lengths)
    -- right align the questions. 
    local v=seq:clone():fill(0)
    local N=seq:size(2)
    for i=1,seq:size(1) do
        v[i][{{N-lengths[i]+1,N}}]=seq[i][{{1,lengths[i]}}]
    end
    return v
end

function utils.normlize_image(imgFeat)
    local length = imgFeat:size(2)
    local nm=torch.sqrt(torch.sum(torch.cmul(imgFeat,imgFeat),2)) 
    return torch.cdiv(imgFeat,torch.repeatTensor(nm,1,length)):float()
end

function utils.count_key(t)
    local count = 1
    for i, w in pairs(t) do 
        count = count + 1 
    end
    return count
end


function utils.prepro(im, on_gpu)
  assert(on_gpu ~= nil, 'pass this in. careful here.')

  im=im*255
  local im2=im:clone()
  im2[{{},{3},{},{}}]=im[{{},{1},{},{}}]-123.68
  im2[{{},{2},{},{}}]=im[{{},{2},{},{}}]-116.779
  im2[{{},{1},{},{}}]=im[{{},{3},{},{}}]-103.939  
  
  return im2
end

-- Numbers
function utils.float(num, order)
	num = math.floor(num*math.pow(10, order))/math.pow(10,order)
	return num
end




-- File I/O
-- %%% recursive 하게 폴더 생성해 주는 기능 추가해야 됨.
function utils.mkdir(path)
	lfs.mkdir(path)
end
function utils.rmdir(path)
	lfs.rmdir(path)
end

function utils.get_file_list(path)
	print('file list')	
end


function utils.file_exists(file)
	local f = io.open(file, 'rb')
	if f then f:close() end
	return f~=nil
end

function utils.readlines_txt(file)
	if not utils.file_exists(file) then
		return {}
	end
	lines = {}
	for line in io.lines(file) do
		lines[#lines + 1] = line
	end
	return lines
end


-- Image processing
function utils.load_image(path)
	local im = image.load(path, 3, 'float')
	return im
end

function utils.save_image(path, im)
	return torch.save(path, im)
end

function utils.get_dim(im, dim)
	if dim == 'channel' then
		return im:size(1)
	elseif dim == 'width' then
		return im:size(3)
	elseif dim == 'height' then
		return im:size(2)
	else
		print('wrong dim argument!')
		return -1
	end
end

function utils.resize(im, width, height)
	im = image.scale(im, widdth, height)
	return im
end

-- returns resize X resize image with padding.
function utils.resize_with_padding(im, resize)
	w = utils.get_dim(im, 'width')
	h = utils.get_dim(im, 'height')
	
	if (w>=h) then
		buf = torch.Tensor(3,w,w):zero()
		offset = math.ceil((w-h)/2)
		mask = torch.ByteTensor(3,w,w):zero()
		mask[{ {}, {offset+1, offset+h}, {} }] = 1
		buf:maskedCopy(mask, im)
		buf = image.scale(buf, resize, resize)
		return {buf, offset, 0, w}

	elseif (h>w) then
		buf = torch.Tensor(3,h,h):zero()
		offset = math.ceil((h-w)/2)
		mask = torch.ByteTensor(3,h,h):zero()
		mask[{ {}, {}, {offset+1, offset+w} }] = 1
		buf:maskedCopy(mask, im)
		buf = image.scale(buf, resize, resize)
		return {buf, 0, offset, h}
	end
end

function utils.crop(im, size, format)
	assert(utils.get_dim(im, 'width')>size, 'crop size is larger than image size.')
	assert(utils.get_dim(im, 'height')>size, 'crop size is larger than image size.')


	local key = 0
	if format == 'center' then
		key = 0
	elseif format == 'top-left' then
		key = 1
	elseif format == 'top-right' then
		key = 2
	elseif format == 'btm-left' then
		key = 3
	elseif format == 'btm-right' then
		key = 4
	elseif format == 'random' then
		key = math.random(0,4)
	end
	
	if key == 0 then
		im = image.crop(im, 'c', size, size)
	elseif key == 1 then
		im = image.crop(im, 'tl', size, size)
	elseif key == 2 then
		im = image.crop(im, 'tr', size, size)
	elseif key == 3 then
		im = image.crop(im, 'bl', size, size)
	elseif key == 4 then
		im = image.crop(im, 'br', size, size)
	end

	return im
end


function utils.crop_with_g(im1, im2, g, size, format)
	assert(utils.get_dim(im1, 'width')>size, 'crop size is larger than image size.')
	assert(utils.get_dim(im1, 'height')>size, 'crop size is larger than image size.')
	assert(utils.get_dim(im2, 'width')>size, 'crop size is larger than image size.')
	assert(utils.get_dim(im2, 'height')>size, 'crop size is larger than image size.')


	local key = 0
	if format == 'center' then
		key = 0
	elseif format == 'top-left' then
		key = 1
	elseif format == 'top-right' then
		key = 2
	elseif format == 'btm-left' then
		key = 3
	elseif format == 'btm-right' then
		key = 4
	elseif format == 'random' then
		key = math.random(0,4)
	end
	
	if key == 0 then
		im1 = image.crop(im1, 'c', size, size)
		im2 = image.crop(im2, 'c', size, size)
		g = image.crop(g, 'c', size, size)
	elseif key == 1 then
		im1 = image.crop(im1, 'tl', size, size)
		im2 = image.crop(im2, 'tl', size, size)
		g = image.crop(g, 'tl', size, size)
	elseif key == 2 then
		im1 = image.crop(im1, 'tr', size, size)
		im2 = image.crop(im2, 'tr', size, size)
		g = image.crop(g, 'tr', size, size)
	elseif key == 3 then
		im1 = image.crop(im1, 'bl', size, size)
		im2 = image.crop(im2, 'bl', size, size)
		g = image.crop(g, 'bl', size, size)
	elseif key == 4 then
		im1 = image.crop(im1, 'br', size, size)
		im2 = image.crop(im2, 'br', size, size)
		g = image.crop(g, 'br', size, size)
	end

	return im1, im2, g
end


return utils
