import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
	def __init__(self, input_dim, output_dim, num_vetex, act=F.relu, dropout=0.5, bias=True):
		super(GraphConvolution, self).__init__()

		self.alpha = 0.8

		self.act = act
		self.dropout = nn.Dropout(dropout)
		self.weight = nn.Parameter(torch.randn(input_dim, output_dim)).to(device)
		if bias:
			self.bias = nn.Parameter(torch.randn(output_dim)).to(device)
		else:
			self.bias = None

		for w in [self.weight]:
			nn.init.xavier_normal_(w)

	def normalize(self, m):
		rowsum = torch.sum(m, 0)
		r_inv = torch.pow(rowsum, -0.5)
		r_mat_inv = torch.diag(r_inv).float()

		m_norm = torch.mm(r_mat_inv, m)
		m_norm = torch.mm(m_norm, r_mat_inv)

		return m_norm

	def forward(self, adj, x):

		x = self.dropout(x)

		# K-ordered Chebyshev polynomial
		adj_norm = self.normalize(adj)
		sqr_norm = self.normalize(torch.mm(adj,adj))
		m_norm = self.alpha*adj_norm + (1.-self.alpha)*sqr_norm

		x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
		x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
		if self.bias is not None:
			x_out += self.bias

		x_out = self.act(x_out)
		
		return x_out
		

class StandConvolution(nn.Module):
	def __init__(self, dims, num_classes, dropout):
		super(StandConvolution, self).__init__()

		self.dropout = nn.Dropout(dropout)
		self.conv = nn.Sequential(
								   nn.Conv2d(dims[0], dims[1], kernel_size=3),
								   nn.InstanceNorm2d(dims[1]),
								   nn.ReLU(inplace=True),
								   nn.AvgPool2d(3, stride=2),
								   nn.Conv2d(dims[1], dims[2], kernel_size=3),
								   nn.InstanceNorm2d(dims[2]),
								   nn.ReLU(inplace=True),
								   nn.AvgPool2d(3, stride=2),
								   nn.Conv2d(dims[2], dims[3], kernel_size=3),
								   nn.InstanceNorm2d(dims[3]),
								   nn.ReLU(inplace=True),
								   nn.AvgPool2d(3, stride=2)
								   ).to(device)

		self.fc = nn.Linear(dims[3]*3, num_classes).to(device)

	def forward(self, x):
		x = self.dropout(x.permute(0,3,1,2))
		x_tmp = self.conv(x)
		x_out = self.fc(x_tmp.view(x.size(0), -1))

		return x_out


class StandRecurrent(nn.Module):
	def __init__(self, dims, num_classes, dropout):
		super(StandRecurrent, self).__init__()

		self.lstm = nn.LSTM(dims[0]*45, dims[1], batch_first=True,
							dropout=0).to(device)
		self.fc = nn.Linear(dims[1], num_classes).to(device)

	def forward(self, x):
		x_tmp,_ = self.lstm(x.contiguous().view(x.size(0), x.size(1), -1))
		x_out = self.fc(x_tmp[:,-1])

		return x_out
