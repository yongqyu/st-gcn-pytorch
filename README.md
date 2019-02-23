# Human Action Recognition

This code is implemented based on [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455).  
And this code only supports the Florence 3D action dataset.  
Please download from [here](https://www.micc.unifi.it/resources/datasets/florence-3d-actions-dataset/) at ```./dataset/```

## Environments
* python 3.6  
* pytorch 1.0

## Run
<pre><code>python main.py</code></pre>


## File Details
| File Name | Description |
|:--------:|:--------|
| ```preprocess.py``` | Preprocess the Florence 3D action dataset.   Each frame is unified into 32 frames, separated by train / valid / test and dropped into a file. |
| ```config.py``` | Hyperparameter and data path setting |
| ```main.py``` | Execution File. Loads data and models, and performs training and testing. |
| ```model.py``` | Defines the General Graph Convolutional Network (GGCN) class. Construct an adjacency matrix with three consecutive graphs, and call several layers. |
| ```layer.py``` | Defines the graph convolution, standard convolution and classifier layer. |
| ```metric.py``` | Defines the accuracy function |
