"""Load PaintNet dataset for use in a PyTorch Dataloader"""
import sys
import json
import os
import os.path
import pdb
import math

import torch
import torch.utils.data as data
import numpy as np
import csv
from scipy.spatial.transform import Rotation as Rot

from paintnet_utils import *


class PaintNetDataloader(data.Dataset):
    
    def __init__(self,
                 root='',
                 dataset=None,
                 pc_points=5120,
                 traj_points=500,
                 lambda_points=1,
                 overlapping=0,
                 split='train',
                 full_traj=True,
                 extra_data=None,
                 weight_orient=1.,
                 cache_size=2000,
                 overfitting=None,
                 augmentations=None,
                 normalization='per-mesh'):
        """
        root : str
               dataset root path
        dataset : str
                  dataset name (e.g. containers-v2, windows-v1, ...)
        pc_points : int
                  number of final subsampled points from the point-cloud (mesh)
        traj_points : int
                       number of final subsampled points from the trajectory
        split : str
                <train,test>
        full_traj : bool
                    Whether to return the trajectory with merged strokes, rather than separate strokes (not implemented yet)
        extra_data : tuple of strings
                     Whether to include velocities and/or orientations in trajectories.
                     e.g. ('vel',); ('vel', 'orientquat', 'orientrotvec', 'orientnorm');
        cache_size : int
                     number of obj-traj pairs to save in cache during training
        overfitting : int
                      if set, overfits to a single sample, whose index is <overfitting>
        lambda_points : int
                        instead of considering point-clouds (N,3), reshape data into lambda-sequences as (N/lambda, lambda)
                        Any remainder is filled up with padding values
        overlapping : int
                      number of overlapping points between subsequent lambda-sequences. overlapping > lambda_points
        normalization : str
                        normalization type. One of: None, 'per-mesh' (norm mesh to unit sphere),
                        'per-dataset' (scale based on max mesh point across dataset).
                        NOTE: all (mesh, traj) pairs are ALWAYS shifted to have mesh zero-mean.
        """
        self.dataset = dataset
        self.root = root
        self.pc_points = pc_points
        self.traj_points = traj_points
        self.lambda_points = lambda_points
        self.overlapping = overlapping
        self.normalization = normalization
        self.full_traj = full_traj
        self.cache = {}
        self.cache_size = cache_size
        self.overfitting = overfitting
        self.weight_orient = weight_orient


        """
            Sanity checks
        """
        assert len(root) > 0, "No data root specified"
        assert full_traj == True, "Strokes loading has not been implemented yet"
        assert lambda_points > overlapping, 'Overlapping can not be equal or lower than lambda'
        assert overlapping >= 0, 'overlapping value can only be positive'

        if extra_data is not None and not(set(extra_data) <= {'vel', 'orientquat', 'orientrotvec', 'orientnorm'}):
            raise ValueError('extra_data allowed entries are ("vel", "orientquat", "orientrotvec, orientnorm")')
        elif extra_data is None:
            extra_data = tuple()
        assert not ('vel' in extra_data and orient_in(extra_data)[0]), 'vel and orientations together are not yet compatible. You would need to fix the network output as well'
        self.extra_data = extra_data

        if augmentations is None:
            augmentations = []
        assert set(augmentations) <= {'rot', 'roty', 'rotx'}, f'Some augmentation is not available: {augmentations}'
        self.augmentations = augmentations

        assert normalization in ['none', 'per-mesh', 'per-dataset'], f'Normalization type {normalization} is not valid.'
        if normalization == 'per-dataset':
            # self.dataset_max_distance = 0
            self.dataset_mean_max_distance = get_dataset_downscale_factor(self.dataset)
            if self.dataset_mean_max_distance is None:
                self.compute_dataset_mean_max_distance = []

        """
            Directories loading
        """
        assert split in ['train', 'test'], f'Split value {split} is not valid'
        with open(os.path.join(self.root, f'{split}_split.json'), 'r') as f:
            dir_samples = [str(d) for d in json.load(f)]

        self.datapath = []

        for curr_dir in dir_samples:
            mesh_filename = curr_dir+'.obj'
            traj_filename = 'trajectory.txt'
            assert os.path.exists(os.path.join(self.root, curr_dir, mesh_filename)), f"mesh file {mesh_filename} does not exist in dir: {curr_dir}"
            assert os.path.exists(os.path.join(self.root, curr_dir, traj_filename)), f"traj file {traj_filename} does not exist in dir: {curr_dir}"

            if normalization == 'per-dataset' and self.dataset_mean_max_distance is None:
                # self.dataset_max_distance = max(self.dataset_max_distance, get_max_distance(os.path.join(self.root, curr_dir, mesh_filename)))
                self.compute_dataset_mean_max_distance.append(get_max_distance(os.path.join(self.root, curr_dir, mesh_filename)))

            mesh_file = os.path.join(self.root, curr_dir, mesh_filename)
            traj_file = os.path.join(self.root, curr_dir, traj_filename)
            self.datapath.append(  (mesh_file, traj_file, curr_dir)   )


        if normalization == 'per-dataset' and self.dataset_mean_max_distance is None:  # Compute dataset mean if it has not been computed yet
            self.dataset_mean_max_distance = np.mean(self.compute_dataset_mean_max_distance)
            print(f'Mean_max_distance computed on the fly for dataset {self.dataset}: {self.dataset_mean_max_distance}')


    def __getitem__(self, index):
        if self.overfitting is not None:
            index = self.overfitting
        if index in self.cache:  # Retrieve from cache
            point_cloud, traj, dirname = self.cache[index]
        else:  # Retrieve from filesystem
            mesh_file, traj_file, dirname = self.datapath[index]
            point_cloud = read_mesh_as_pointcloud(mesh_file)
            traj, stroke_ids = read_traj_file(traj_file, extra_data=self.extra_data, weight_orient=self.weight_orient)
            point_cloud, traj = center_pair(point_cloud, traj, mesh_file)  # Shift to zero mean
            if self.normalization == 'per-dataset':
                point_cloud /= self.dataset_mean_max_distance
                traj[:, :3] /= self.dataset_mean_max_distance
            elif self.normalization == 'per-mesh':
                # max_distance = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
                max_distance = get_max_distance(mesh_file)
                point_cloud /= max_distance
                traj[:, :3] /= max_distance

            assert point_cloud.shape[0] >= self.pc_points
            choice = np.random.choice(point_cloud.shape[0], self.pc_points, replace=False)  # Sub-sample point-cloud randomly
            point_cloud = point_cloud[choice, :]
            
            choice = np.round_(np.linspace(0, (traj.shape[0]-1), num=self.traj_points)).astype(int)  # Sub-sample traj at equal intervals (up to rounding) for a total of <self.traj_points> points
            traj = traj[choice, :]
            stroke_ids = stroke_ids[choice]

            if self.lambda_points > 1:
                traj, stroke_ids = get_sequences_of_lambda_points(traj, stroke_ids, self.lambda_points, dirname, overlapping=self.overlapping, extra_data=self.extra_data)  # Note: stroke_ids are not padded, traj is

            # plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
            # visualize_pc(point_cloud, plotter=plotter, index=(0,0))
            # visualize_mesh(os.path.join(self.root, dirname, dirname+'_norm.obj'), plotter=plotter, index=(0,0))
            # visualize_sequence_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_traj(traj, plotter=plotter, index=(0,0), extra_data=self.extra_data)
            # visualize_mesh_traj(os.path.join(self.root, dirname, dirname+'_norm.obj'), traj, extra_data=extra_data)
            # plotter.add_axes_at_origin()
            # plotter.show()

            # mean_knn = mean_knn_distance(traj[:, :3], k=1, render=True)
            # pdb.set_trace()

            if 'vel' in self.extra_data:  # Include velocities
                assert self.lambda_points == 1, 'The opposite needs to be thought through: does it make sense to compute velocities for sequences? ALSO. MAKE SURE PADDING IS TAKEN CARE OF OTHERWISE'
                traj_vel = get_velocities(traj, stroke_ids)
                traj = np.concatenate((traj, traj_vel), axis=-1)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_cloud, traj, dirname)


        if len(self.augmentations) > 0:
            if 'rot' in self.augmentations or 'roty' in self.augmentations or 'rotx' in self.augmentations:
                """3D Rotations have multiple representations, but can be described by a minimum of 3 parameters.
                For example, quaternions use 4 plus a constraint. In general, all rotations in 3D space can
                be broken down to a single rotation about some axis (Euler's theorem), hence be described by
                one parameterization of choice (quaternions, rotation matrix, euler angles).
                https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
                """
                

                point_cloud, traj = point_cloud.copy(), traj.copy()  # Not sure if needed, but this way I don't apply changes to the cache

                if 'roty' in self.augmentations:
                    alpha = np.random.uniform(-math.pi, math.pi)
                    rot = Rot.from_euler(seq="y", angles=alpha)
                elif 'rotx' in self.augmentations:
                    alpha = np.random.uniform(-math.pi, math.pi)
                    rot = Rot.from_euler(seq="x", angles=alpha)
                else:
                    rot = Rot.random()

                outdim = get_dim_traj_points(self.extra_data)
                if self.lambda_points > 1:
                    traj = traj.reshape(-1, outdim)
                    traj = remove_padding(traj, extra_data=self.extra_data)

                    # visualize_pc(point_cloud)
                    # visualize_traj(traj, extra_data=self.extra_data)
                    if orient_in(self.extra_data)[0]:
                        orient_indexes = get_traj_feature_index(orient_in(self.extra_data)[1], self.extra_data)
                        point_cloud, traj[:, :3], traj[:, orient_indexes] = rot.apply(point_cloud), rot.apply(traj[:, :3].copy()), rot.apply(traj[:, orient_indexes].copy())
                    else:
                        point_cloud, traj[:, :3] = rot.apply(point_cloud), rot.apply(traj[:, :3].copy())
                    # visualize_pc(point_cloud)
                    # visualize_traj(traj, extra_data=self.extra_data)

                    traj = traj.reshape(-1, outdim*self.lambda_points)
                    traj = add_padding(traj, traj_points=self.traj_points, lmbda=self.lambda_points, overlapping=self.overlapping, extra_data=self.extra_data)
                else:
                    point_cloud, traj[:,:outdim] = rot.apply(point_cloud), rot.apply(traj[:,:outdim])
                    if 'vel' in self.extra_data:
                        raise NotImplementedError('This part is now deprecated. Fix it with the correct indexes in case orient_in() is True etc.')
                        traj[:,3:6] = rot.apply(traj[:,3:6])
        

        return point_cloud, traj, dirname

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    set_seed(1)

    tr_dataset = PaintNetDataloader(root='<data root path>',
                                  pc_points=5120,
                                  traj_points=1500,
                                  normalize=True,
                                  extra_data=('vel',), # ('vel',)
                                  split='train')


    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            drop_last=True)

    it = iter(tr_loader)
    point_cloud, traj, dirname = next(it)

    print('Object:', point_cloud.shape) # (B, pc_points, 3)
    print('Trajectory:', traj.shape) # (B, traj_points, <3,6>)
    print('Dirname:', dirname) # (B,) list of strings