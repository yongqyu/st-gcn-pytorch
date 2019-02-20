import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from layer import GraphConvolution, StandConvolution

class GGCN(nn.Module):
	def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, dropout=0.5):
		super(GGCN, self).__init__()

		adj = adj + torch.eye(adj.size(0)).to(adj).detach()
		ident = torch.eye(adj.size(0)).to(adj)
		zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
		self.adj = torch.cat([torch.cat([adj, ident, zeros], 1),
							  torch.cat([ident, adj, ident], 1),
							  torch.cat([zeros, ident, adj], 1)], 0).float()
		
		self.gcl = GraphConvolution(gc_dims[0], gc_dims[1], num_v, dropout=dropout)
		self.conv= StandConvolution(sc_dims, num_classes, dropout=dropout)

	def forward(self, x):
		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # 1, 30, 45, 3
		multi_conv = self.gcl(self.adj, concat_seq)
		logit = self.conv(multi_conv)
		
		return logit
		
