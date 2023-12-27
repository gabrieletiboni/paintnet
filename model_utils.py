import os
import pdb

import torch

# from models.pointnet import PointNetRegressor
# from models.pointnet_deeper import PointNetRegressor as PointNetDeeperRegressor
from models.pointnet2_cls_ssg import PointNet2Regressor
from paintnet_utils import *

def get_model(backbone, config):
    outdim = get_dim_traj_points(config['extra_data'])
    orient_outdim = get_dim_orient_traj_points(config['extra_data'])

    vector_outdim_transl =  (outdim - orient_outdim) * config['lambda_points']  # Translation dimensionality of each output vector
    vector_outdim_orient = orient_outdim * config['lambda_points']  # Orientation dimensionality of each output vector

    out_vectors = (config['traj_points']-config['lambda_points'])//(config['lambda_points']-config['overlapping']) + 1   # Rounded number of overlapping sequences
    print('Number of output vectors (mini-sequences or single poses):', out_vectors)


    """
        Backbones
    """
    if backbone == 'pointnet2':  # PointNet++
        assert config['pc_points'] > 512, 'farthest point sampling set to 512'
        return PointNet2Regressor(out_vectors=out_vectors,
                                  outdim=vector_outdim_transl,
                                  outdim_orient=vector_outdim_orient,
                                  weight_orient=config['weight_orient'],
                                  hidden_size=config['model']['hidden_size'])
    else:
        raise ValueError(f'Specified backbone does not exist: {backbone}')


def init_from_pretrained(model, config, device='cpu'):
    """Initialize feature encoder with a pretrained model on
    ShapeNet or similar datasets for common tasks
    (PartSeg, Classification, etc.)
    """
    if config['backbone'] == 'pointnet2':
        state_dict = torch.load(os.path.join('pretrained_models', 'pointnet2_cls_ssg.pth'), map_location=device)['model_state_dict']
        feature_encoder_state_dict = _filter_out_dict(state_dict, ['fc1.weight', 'fc1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked', 'fc2.weight', 'fc2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var', 'bn2.num_batches_tracked', 'fc3.weight', 'fc3.bias'])
        model.load_state_dict(feature_encoder_state_dict, strict=False)
        return model

    else:
        raise ValueError(f"No pretrained model exists for this backbone: {config['backbone']}")


def _filter_out_dict(state_dict, remove_layers):
    """Filter out layers that you do not want to initialize with transfer learning"""
    pretrained_dict = {k: v for k, v in state_dict.items() if k not in remove_layers}
    return pretrained_dict