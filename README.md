
# Learning on 3D Meshes with Laplacian Encoding and Pooling
This is a PyTorch re-implementation to show the preprocessing of feature computation and clustering can be done on the fly. (in progress)

## Usage
Run a demo training code for coseg.

1. Download the [COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm). A [script](https://github.com/ranahanocka/MeshCNN/blob/master/scripts/coseg_seg/get_data.sh) can be used to download the dataset. 

2. Put the dataset in `../dataset` and make a working directory in `../work_dir`

3. Runing the training code
```
python train_coseg
```

