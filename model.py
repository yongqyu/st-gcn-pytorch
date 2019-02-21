import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from layer import GraphConvolution, StandConvolution

class GGCN(nn.Module):
	def __init__(self, adj, num_v, num_classes, gc_dims, sc_dims, feat_dims, dropout=0.5):
		super(GGCN, self).__init__()

		terminal_cnt = 5
		actor_cnt = 1
		adj = adj + torch.eye(adj.size(0)).to(adj).detach()
		ident = torch.eye(adj.size(0)).to(adj)
		zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
		self.adj = torch.cat([torch.cat([adj, ident, zeros], 1),
							  torch.cat([ident, adj, ident], 1),
							  torch.cat([zeros, ident, adj], 1)], 0).float()
		self.terminal = nn.Parameter(torch.randn(terminal_cnt, actor_cnt, feat_dims))

		self.gcl = GraphConvolution(gc_dims[0]+feat_dims, gc_dims[1], num_v, dropout=dropout)
		self.conv= StandConvolution(sc_dims, num_classes, dropout=dropout)

		nn.init.xavier_normal_(self.terminal)

	def forward(self, x):
		head_la = F.interpolate(torch.stack([self.terminal[0],self.terminal[1]],2), 6)
		head_ra = F.interpolate(torch.stack([self.terminal[0],self.terminal[2]],2), 6)
		lw_ra = F.interpolate(torch.stack([self.terminal[3],self.terminal[4]],2), 6)
		node_features = torch.cat([
								   (head_la[:,:,:3] + head_ra[:,:,:3])/2,
								   torch.stack((lw_ra[:,:,2], lw_ra[:,:,1], lw_ra[:,:,0]), 2),
								   lw_ra[:,:,3:], head_la[:,:,3:], head_ra[:,:,3:]], 2).to(x)
		x = torch.cat((x, node_features.permute(0,2,1).unsqueeze(1).repeat(1,32,1,1)), 3)

		concat_seq = torch.cat([x[:,:-2], x[:,1:-1], x[:,2:]], 2) # 1, 30, 45, 3
		multi_conv = self.gcl(self.adj, concat_seq)
		logit = self.conv(multi_conv)
		
		return logit
		
