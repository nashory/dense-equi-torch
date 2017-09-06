# plot 3d scatter graph. (used python plotly open-source library.)
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import math

#--------------- verify API keys (these are fakes :-)) -----------------------
# user_id : stewdlfra
# API key : 1mGJXNwkdowkYIktmE

# user_id : stewdftra555
# API key : ERqawleid9Xsdfd4S

# user_id : stedddtra5666
# API key : Ihcsdfsdfsfnc5

plotly.tools.set_credentials_file(username='stesdfftra666', api_key='IhcsdfdsdffKwKnc5')
#--------------------------------------------------------


test = [60780]


idx = 0
for i in test:
	feat_le = np.load('repo/vis/Iter' + str(i) + '_feat_le.npy')
	feat_re = np.load('repo/vis/Iter' + str(i) + '_feat_re.npy')
	feat_no = np.load('repo/vis/Iter' + str(i) + '_feat_no.npy')
	feat_lm = np.load('repo/vis/Iter' + str(i) + '_feat_lm.npy')
	feat_rm = np.load('repo/vis/Iter' + str(i) + '_feat_rm.npy')
	
	
	# draw plot.
	trace_le = go.Scatter3d(
		x = feat_le[0,:],
		y = feat_le[1,:],
		z = feat_le[2,:],
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(20,20,20,0)',
				width = 0.1
			),
			opacity = 1.0
		)
	)

	trace_re = go.Scatter3d(
		x = feat_re[0,:],
		y = feat_re[1,:],
		z = feat_re[2,:],
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(255,0,0,0)',
				width = 0.1
			),
			opacity = 1.0
		)
	)
	
	trace_no = go.Scatter3d(
		x = feat_no[0,:],
		y = feat_no[1,:],
		z = feat_no[2,:],
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(0,255,0,0)',
				width = 0.1
			),
			opacity = 1.0
		)
	)
	
	trace_lm = go.Scatter3d(
		x = feat_lm[0,:],
		y = feat_lm[1,:],
		z = feat_lm[2,:],
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(0,0,255,0)',
				width = 0.1
			),
			opacity = 1.0
		)
	)
	
	trace_rm = go.Scatter3d(
		x = feat_rm[0,:],
		y = feat_rm[1,:],
		z = feat_rm[2,:],
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(100,20,50,0)',
				width = 0.1
			),
			opacity = 1.0
		)
	)
	
	trace_origin = go.Scatter3d(
		x = np.zeros(1),
		y = np.zeros(1),
		z = np.zeros(1),
	
		mode='markers',
		marker = dict(
			size = 5,
			line = dict(
				color='rgba(255,100,70,0.14)',
				width = 0.5
			),
			opacity = 0.8
		)
	)

	data = [trace_le, trace_re, trace_no, trace_lm, trace_rm, trace_origin]
	
	layout = go.Layout(
		margin=dict(
			l=0,
			r=0,
			b=0,
			t=0
		)
	)


	fig = go.Figure(data=data, layout=layout)
	py.iplot(fig, filename='cluster-L3-Iter' + str(i))
	print 'saved cluster-L3-Iter' + str(i)
	idx = idx+1







