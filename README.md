# Human Action Recognition

This code is implemented based on [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455).  
And this code only supports the Florence 3D action dataset.  
Please download from [here](https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/) at ```./dataset/```

## File Detail
| File Name | 가운데 정렬 |
|:--------:|:--------|
| ```preprocess.py``` | Preprocess the Florence 3D action dataset.   Each frame is unified into 32 frames, separated by train / valid / test and dropped into a file. |
| ```config.py``` | Hyperparameter and data path setting |
| ```main.py``` |  |
| ```model.py``` |  |
| ```layer.py``` |  |
| ```metric.py``` |  |
