import os
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from model import *
from config import get_args
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

train_tensor, train_label = torch.load(args.train_path)
valid_tensor, valid_label = torch.load(args.valid_path)
test_tensor , test_label  = torch.load(args.test_path)
train_loader = data.DataLoader(data.TensorDataset(train_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
valid_loader = data.DataLoader(data.TensorDataset(valid_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
test_loader  = data.DataLoader(data.TensorDataset(test_tensor.to(device)),
							   batch_size = args.batch_size, shuffle=False)
train_label = train_label.to(device)
valid_label = valid_label.to(device)
test_label  = test_label.to(device)
A = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	 [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
	 [0,1,0,1,0,0,1,0,0,1,0,0,0,0,0],
	 [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
	 [0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
	 [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
	 [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
	 [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
	 [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
	 [0,0,1,0,0,0,0,0,0,0,1,0,1,0,0],
	 [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
	 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
	 [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
	 [0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
	 [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]]
A = torch.from_numpy(np.asarray(A)).to(device)

model = GGCN(A, train_tensor.size(3), args.num_classes, 
			 [train_tensor.size(3), train_tensor.size(3)*3], [train_tensor.size(3)*3, 16, 32, 64])
if device == 'cuda':
	model.cuda()

num_params = 0
for p in model.parameters():
	num_params += p.numel()
print(model)
print('The number of parameters: {}'.format(num_params))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

def train():
	if args.start_epoch:
		model.load_state_dict(torch.load(os.path.join(args.model_path, 
													  'model-%d.pkl'%(args.start_epoch))).state_dict())

	# Training
	for epoch in range(args.start_epoch, args.num_epochs):
		train_loss = 0
		model.train()
		for i, x in enumerate(train_loader):
			logit = model(x[0].float())
			target = train_label[i]

			loss = criterion(logit, target.view(1))

			model.zero_grad()
			loss.backward()
			optimizer.step()

		if (epoch+1) % args.val_step == 0:
			model.eval()
			val_loss = 0
			val_acc  = 0
			with torch.no_grad():
				for i, x in enumerate(valid_loader):
					logit = model(x[0].float())
					target = valid_label[i]

					val_loss += criterion(logit, target.view(1))

			print(val_loss)


if __name__ == '__main__':
	if args.mode == 'train':
		train()
	elif args.mode == 'test':
		best_epoch = args.test_epcoh
	test()

