# Chamfer Distance (torch_nndistance)
### Tested with pytorch 1.2.0 and CUDA 10.0 / 10.1

Credits to:
(1) https://github.com/ThibaultGROUEIX/AtlasNet/

'''
    Input:
        pc1: float tensor in shape (B,N,C) the first point cloud
        pc2: float tensor in shape (B,M,C) the second point cloud
    Return:
        dist1: float tensor in shape (B,N) distance from first to second
        dist2: float tensor in shape (B,M) distance from second to first
        idx1: int32 tensor in shape (B,N) nearest neighbor from first to second
        idx2: int32 tensor in shape (B,M) nearest neighbor from second to first
'''

Note: Inputs (pc1, pc2) need to be contiguous!
See sample test for usage example. See (1) for further documentation.
Build the module with:
```bash
python build.py install
```
