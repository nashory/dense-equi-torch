-- torch visualization tool.
-- last modified : 2017.08.16, nashory


--local gp = require 'gnuplot'


local Visualizer = torch.class('Visualizer')


function Visualizer:__init(gp)
	self.gp = gp
	self.gp.figure(1)
	self.gp.title('No title.')

	self.x_data = {}
	self.y_data = {}
	print('Initialize visualizer.')
end


function Visualizer:insert(x_data, y_data)
	table.insert(self.x_data, x_data)
	table.insert(self.y_data, y_data)
end


function Visualizer:arrange(len)
	while (#self.x_data >= len) do
		table.remove(self.x_data, 1)
		table.remove(self.y_data, 1)
	end
end


function Visualizer:label(x_label, y_label)
	self.gp.xlabel(x_label)
	self.gp.ylabel(y_label)
end

function Visualizer:plot()
	self.gp.plot(torch.Tensor(self.x_data), torch.Tensor(self.y_data))
end


function Visualizer:flush()
	self.x_data = {}
	self.y_data = {}
end

function Visualizer:setTitle(title)
	self.gp.title(title)
end

function Visualizer:savepng(path)
	self.gp.savefigure(path)
end

function Visualizer:autorun(input_x, input_y)
	self:setTitle('autorun graph')
	self:arrange(5000)
	self:insert(input_x, input_y)
	self:plot()
end

return Visualizer

