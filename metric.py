import torch
from sklearn.metrics import accuracy_score

def accuracy(preds, target):
	preds = torch.max(preds, 1)[1].float()
	acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())

	return acc
