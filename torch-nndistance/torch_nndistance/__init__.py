__version__ = "1.0.0"

import torch
import torch_nndistance_aten as my_lib
from torch.autograd import Function

class NNDFunction(Function):
    """
    Input:
        pc1: float TF tensor in shape (B,N,C) the first point cloud
        pc2: float TF tensor in shape (B,M,C) the second point cloud
    Return:
        dist1: float TF tensor in shape (B,N) distance from first to second
        dist2: float TF tensor in shape (B,M) distance from second to first
        idx1: int32 TF tensor in shape (B,N) nearest neighbor from first to second
        idx2: int32 TF tensor in shape (B,M) nearest neighbor from second to first
        """
    @staticmethod
    def forward(self, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)
        
        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
            
        self.save_for_backward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @ staticmethod
    def backward(self, graddist1, graddist2, idx1_, idx2_):
        xyz1, xyz2, dist1, dist2, idx1, idx2 = self.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())
        
        if not graddist1.is_cuda:
            my_lib.nnd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            my_lib.nnd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


def nnd(xyz1, xyz2):
    return NNDFunction.apply(xyz1, xyz2)
