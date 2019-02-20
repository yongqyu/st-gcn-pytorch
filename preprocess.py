from torch.utils import data
import torch
import os
import random
import numpy as np


file_name = './dataset/Florence_3d_actions/Florence_dataset_WorldCoordinates.txt'
f = open(file_name)
lines = f.readlines()
prev_video = int(lines[0][0])
prev_categ = int(lines[0][2])
frames = []
train = []
valid = []
test  = []
train_label = []
valid_label = []
test_label  = []
for line in lines:
	line = line.split(' ')
	vid = int(line[0])
	aid = int(line[1])
	cid = int(line[2])-1

	features = list(map(float, line[3:])) 
	#norm_val = float(line[-1])

	if prev_video == vid:
		frames.append(np.reshape(np.asarray(features), (-1,3)))
	else:
		if len(frames) >= 32:
			frames = random.sample(frames, 32)
			frames = torch.from_numpy(np.stack(frames, 0))
		else:
			frames = np.stack(frames, 0)
			xloc = np.arange(frames.shape[0])
			new_xloc = np.linspace(0, frames.shape[0], 32)
			frames = np.reshape(frames, (frames.shape[0], -1)).transpose()

			new_datas = []
			for data in frames:
				new_datas.append(np.interp(new_xloc, xloc, data))
			frames = torch.from_numpy(np.stack(new_datas, 0)).t()

		frames = frames.view(32, -1, 3)
		if prev_actor < 9:
			train.append(frames)
			train_label.append(prev_categ)
		elif prev_actor < 10:
			valid.append(frames)
			valid_label.append(prev_categ)
		else:
			test.append(frames)
			test_label.append(prev_categ)
		frames = [np.reshape(np.asarray(features), (-1,3))]
	prev_actor = aid
	prev_video = vid
	prev_categ = cid
	

if len(frames) >= 32:
	frames = random.sample(frames, 32)
	frames = torch.from_numpy(np.stack(frames, 0))
else:
	frames = np.stack(frames, 0)
	xloc = np.arange(frames.shape[0])
	new_xloc = np.linspace(0, frames.shape[0], 32)
	frames = np.reshape(frames, (frames.shape[0], -1)).transpose()

	new_datas = []
	for data in frames:
		new_datas.append(np.interp(new_xloc, xloc, data))
	frames = torch.from_numpy(np.stack(new_datas, 0)).t()

	
frames = frames.view(32, -1, 3)
if aid < 9:
	train.append(frames)
	train_label.append(prev_categ)
elif aid < 10:
	valid.append(frames)
	valid_label.append(prev_categ)
else:
	test.append(frames)
	test_label.append(prev_categ)

train_label = torch.from_numpy(np.asarray(train_label))
valid_label = torch.from_numpy(np.asarray(valid_label))
test_label  = torch.from_numpy(np.asarray(test_label))

torch.save((torch.stack(train, 0), train_label), './dataset/train.pkl')
torch.save((torch.stack(valid, 0), valid_label), './dataset/valid.pkl')
torch.save((torch.stack(test, 0),  test_label),  './dataset/test.pkl')
