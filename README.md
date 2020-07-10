# Laplacian Cluster
This is a PyTorch re-implementation to show the preprocessing of feature computation and clustering can be done on the fly.

## Usage
Run a demo inference code for classification.
```
python script_inference.py
```

The input to the network is faces, vertices, and features (optional, if there is no feature, it can be computed in the network) of a mesh. 

`network.py` constructs the network in `__init__()`, preprocesses in `preprocess()`, and forward pass in `forward.py`. `laplacian.py` will compute the mesh Laplacian. `normal.py` computes the vertex normal. 