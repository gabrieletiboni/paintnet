"""PointNet++ classification network
https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/log/classification/pointnet2_ssg_wo_normals
"""
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction


class PointNet2Regressor(nn.Module):
    def __init__(self,
                 outdim=3,
                 outdim_orient=3,
                 weight_orient=1.,
                 normal_channel=False,
                 out_vectors=1500,
                 hidden_size=(1024, 1024)
                 ):
        super(PointNet2Regressor, self).__init__()
        """
        outdim: translational dims of each output vector
        outdim_orient: orientation dims of each output vector
        out_vectors: number of output vectors
        """
        self.outdim = outdim
        self.outdim_orient = outdim_orient
        self.out_vectors = out_vectors
        self.weight_orient = weight_orient

        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], out_vectors*outdim)

        if outdim_orient > 0:
            self.fc_normals = nn.Linear(hidden_size[1], out_vectors*outdim_orient)
            self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        x = self.dropout(F.relu(self.bn1(self.fc1(global_feat))))
        final = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(final)

        if self.outdim_orient > 0:
            normals = self.tanh(self.fc_normals(final))
            normals = normals.view(B, -1, 3)
            normals = F.normalize(normals, dim=-1)
            normals *= self.weight_orient
            x = x.view(B, -1, 3)
            out = torch.cat((x, normals), dim=-1)
            out = out.view(B, self.out_vectors, -1)
        else:
            out = x.view(B, self.out_vectors, self.outdim)
        
        return out



# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss