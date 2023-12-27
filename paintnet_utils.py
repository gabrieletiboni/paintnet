import math
import csv
import sys
import pdb
import os
import string
import json
import zlib
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
try:
    import pyvista as pv  
except ImportError:
    print('Warning! Unable to import pyvista package. visualizations won\'t be available. Run `pip install pyvista`')
    pass
try:
    import point_cloud_utils as pcu
except ImportError:
    print(f'Warning! Unable to import point_cloud_utils package.')
    pass
try:
    import networkx as nx 
except ImportError:
    print(f'Warning! Unable to import networkx package.')
    pass



def normalize_pc(pc):
    """Normalizes point-cloud such that furthest point 
    from origin is equal to 1 and mean is centered
    around zero
    
    pc : (N, 3) array

    returns (N, 3) array
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def get_max_distance(meshpath):
    """Returns max distance from mean of given mesh filename"""
    v, f = pcu.load_mesh_vf(os.path.join(meshpath))
    centroid = np.mean(v, axis=0)
    v = v - centroid
    m = np.max(np.sqrt(np.sum(v ** 2, axis=1)))
    return m

def get_mean_mesh(meshpath):
    v, f = pcu.load_mesh_vf(os.path.join(meshpath))
    centroid = np.mean(v, axis=0)
    return centroid

def get_random_string(n=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def center_pair(point_cloud, traj, meshpath):
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3
    centroid = get_mean_mesh(meshpath) # np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    traj[:, :3] -= centroid
    return point_cloud, traj

def center_traj(traj, meshpath):
    centroid = get_mean_mesh(meshpath) # np.mean(point_cloud, axis=0)
    traj[:, :3] -= centroid
    return traj

def read_mesh_as_pointcloud(filename):
    v, f = pcu.load_mesh_vf(os.path.join(filename))
    #target_radius = np.linalg.norm(v.max(0) - v.min(0)) * 0.01
    # print(target_radius)
    f_i, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=15000)  # Num of points (not guaranteed), radius for poisson sampling
    points = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
    return points

def orient_in(extra_data):
    """Whether each output pose includes orientations.
    
    Returns the specific orient representation as well.
    """
    valid = ['orientquat', 'orientrotvec', 'orientnorm']
    for v in valid:
        if v in extra_data:
            return True, v
    
    return False, None

def read_traj_file(filename, extra_data=[], weight_orient=1.):
    """Returns trajectory as nd-array (T, <3,6>)
    given traj filename in .txt"""
    points = []
    stroke_ids = []
    stroke_id_index = 6
    cols_to_read = [0, 1, 2]
    orientations, orient_repr = orient_in(extra_data)
    if orientations:
        cols_to_read += [3, 4, 5]

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)
    
        for cols in reader:
            cols_float = np.array(cols, dtype='float64')[cols_to_read]
            stroke_id = int(np.array(cols, dtype='float64')[stroke_id_index])

            if orientations:
                if orient_repr == 'orientquat':
                    quats = weight_orient*Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True).as_quat()
                    points.append(np.concatenate((cols_float[:3], quats)))
                elif orient_repr == 'orientrotvec':
                    rotvec = weight_orient*Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True).as_rotvec()
                    points.append(np.concatenate((cols_float[:3], rotvec)))
                elif orient_repr == 'orientnorm':
                    rot = Rot.from_euler('yzx', [cols_float[4], cols_float[5], cols_float[3]], degrees=True)
                    e1 = np.array([1,0,0])
                    normals = weight_orient*rot.apply(e1)
                    points.append(np.concatenate((cols_float[:3], normals)))
            else:
                points.append(cols_float)

            stroke_ids.append(stroke_id)

    return np.array(points), np.array(stroke_ids)


def get_traj_feature_index(feat, extra_data):
    if feat == None:
        return None

    if len(extra_data) == 0:
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'vel' in extra_data and len(extra_data) == 1:  # Vel only
        indexes = {
            'pos': [0, 1, 2],
            'vel': [3, 4, 5],
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'orientquat' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': [3, 4, 5, 6],
            'orientrotvec': None,
            'orientnorm': None
        }
    elif 'orientrotvec' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': [3, 4, 5],
            'orientnorm': None
        }
    elif 'orientnorm' in extra_data and len(extra_data) == 1:  # Orient only
        indexes = {
            'pos': [0, 1, 2],
            'vel': None,
            'orientquat': None,
            'orientrotvec': None,
            'orientnorm': [3, 4, 5]
        }
    else:
        raise ValueError('Other combinations of extra_data are not supported yet.')

    return indexes[feat]

def get_dim_traj_points(extra_data):
    """Returns dimensionality of each output pose"""
    if len(extra_data) == 0:
        return 3
    elif 'vel' in extra_data and len(extra_data) == 1:  # Vel only
        return 6
    elif 'orientquat' in extra_data and len(extra_data) == 1:  # Orient only
        return 7
    elif 'orientrotvec' in extra_data and len(extra_data) == 1: # Orient only
        return 6
    elif 'orientnorm' in extra_data and len(extra_data) == 1: # Orient only
        return 6
    else:
        raise ValueError('Other combinations of extra_data are not supported yet.')

def get_dim_orient_traj_points(extra_data):
    """Returns dimensionality of current orientation representation"""
    if not orient_in(extra_data)[0]:
        return 0

    dims = {
        'orientquat': 4,
        'orientnorm': 3,
        'orientrotvec': 3
    }
    for k, v in dims.items():
        if k in extra_data:
            return dims[k]
    raise ValueError(f'Unexpected error: code flow should not get here. Inspect it otherwise. extra_data: {extra_data}')


def get_velocities(traj, stroke_ids):
    """Returns per-point translational velocities.
    The last point of each stroke has zero velocity"""
    vels = np.zeros((traj.shape))

    vels[:-1,:] = (traj[1:, :] - traj[:-1, :]) # / 0.004*100 

    n_strokes = stroke_ids[-1]+1
    for stroke_id in range(1, n_strokes):  # Set to zero velocities at stroke changes
        ending_index = np.argmax(stroke_ids == stroke_id) - 1   # index of last point in current stroke
        vels[ending_index] = 0
    return vels

def get_sequences_of_lambda_points(traj, stroke_ids, lmbda, dirname, overlapping=0, extra_data=[]):
    """Merge consecutive points in traj in a single sequence of lmbda points

    Input:
        traj: (N, 3)
        stroke_ids (N,)
        lmbda: int
        overlapping : int
    Output:
        new_traj: (~(N//lmbda), lmbda)
                  first size is not constant, due to multiple strokes in the same traj.
                  consecutive points are per-stroke, so not all consecutive sequences can be made.
                  In general, new_traj.shape[0] <= N//lmbda if overlapping = 0. Otherwise,
                  the maximum number of sub-sequences is (N-lmbda)/(lmbda-overlapping) + 1 
    """
    outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == outdim
    
    N, D = traj.shape

    n_strokes = stroke_ids[-1]+1
    count = 0
    new_stroke_count = 0
    first_time = True
    warning_skipped_strokes = 0

    start_idx = 0
    for stroke_id in range(n_strokes):
        if stroke_id == n_strokes-1:  # if last one
            end_idx = N - 1
        else:
            end_idx = np.argmax(stroke_ids == (stroke_id+1)) - 1   # index of last point in current stroke (search for next stroke's starting point)

        stroke_length = end_idx+1 - start_idx
        curr_stroke = traj[start_idx:start_idx+stroke_length]

        if stroke_length >= lmbda:
            """Index of starting sequences (last idx is used as :last_idx to get final point of sequence)
            (stroke_length+1 because <stop> is never included, this way we can potentially include the index of point in the next stroke to use as :ar[-1])
            """

            if overlapping == 0:
                ar = np.arange(0, stroke_length+1, step=lmbda)
                
                remainder = (stroke_length%lmbda)
                centered_stroke = curr_stroke[(remainder//2) : ar[-1] + (remainder//2)]  # Pad this stroke
                new_traj_piece = centered_stroke.reshape((-1, lmbda*outdim))  # Join lambda points in the same dimension


            else:  # Handle mini-sequence that overlap
                overlapped_repetitions = int(  (stroke_length - lmbda) / (lmbda - overlapping)  )
                assert  int(  (stroke_length - lmbda) / (lmbda - overlapping)  ) ==  (stroke_length - lmbda) // (lmbda - overlapping)
                
                eff_length = overlapped_repetitions*(lmbda - overlapping) + lmbda
                remainder = stroke_length%eff_length

                ol_length = lmbda - overlapping  # overlapped_length

                new_traj_piece = np.array([ curr_stroke[(i*ol_length):(i*ol_length)+lmbda] for i in range(overlapped_repetitions+1) ])  # +1 for the starting non-overlapped sequence on the left, (overlapped_repetitions+1, lambda, outdim)
                assert new_traj_piece.ndim == 3
                new_traj_piece = new_traj_piece.reshape((overlapped_repetitions+1), lmbda*outdim)  # (overlapped_repetitions+1, lmbda*outdim)


            if first_time:
                new_traj = new_traj_piece.copy()
                new_stroke_ids = np.ones((new_traj_piece.shape[0]))*(new_stroke_count)
                first_time = False
            else:
                new_traj = np.append(new_traj, new_traj_piece, axis=-2)
                new_stroke_ids = np.append(new_stroke_ids, np.ones((new_traj_piece.shape[0]))*(new_stroke_count))

            new_stroke_count += 1

        else:
            # Cannot make sequences of points longer than the points present in this stroke
            # print(f'Warning! A stroke has been ignored and discarded due to a <lambda_points> value' \
            #       f'being higher than this stroke length. Lambda:{lmbda} | Stroke_length:{stroke_length}' \
            #       f'| Stroke_id:{stroke_id} | Dirname:{dirname}')
            warning_skipped_strokes += 1

        start_idx = end_idx+1
        count += 1

    if overlapping == 0:
        assert new_traj.shape[0] <= N//lmbda
    else:
        assert new_traj.shape[0] <= (N-lmbda)//(lmbda-overlapping) + 1 
    assert count == n_strokes
    assert new_traj.shape[-1] == (lmbda*outdim)

    # Pad last values with -100
    new_traj = add_padding(new_traj, N, lmbda, overlapping, extra_data=extra_data)
    # NOTE: new_stroke_ids are not padded, hence their sizes differ when padding is considered

    if warning_skipped_strokes > 0:
        print(f'Warning! Skipped {warning_skipped_strokes} strokes in {dirname} as having length < {lmbda}')

    return new_traj, new_stroke_ids


def save_traj_file(traj, filepath):
    """Save trajectory as k-dim sequence of vectors,
    with k=3 (x,y,z), or k=6 (+ orientations), or k=7 (+ strokeId)

    traj : (N, k) array, 
    """
    assert traj.ndim == 2 and (traj.shape[-1]==3 or traj.shape[-1]==6 or traj.shape[-1]==7), f"Trajectory is not formatted correctly: {traj.shape} - {traj}"

    if torch.is_tensor(traj):
        traj = traj.cpu().detach().numpy()
    
    k = traj.shape[-1]

    header = ['X','Y','Z','A','B','C','strokeId']
    header = header[:k]
    with open(os.path.join(filepath), 'w', encoding='utf-8') as file:
        print(";".join(header), file=file)
        for cols in traj:
            print(";".join(map(str, cols)), file=file)
    return

def visualize_mesh(meshfile, plotter=None, index=None, text=None, camera=None):
    """Visualize mesh, given filename.obj"""
    show_plot = True if plotter is None else False
    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh_obj = pv.read(meshfile)
    plotter.add_mesh(mesh_obj)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return

def is_padded(traj):
    return np.any(np.where(((traj[:,0] == -100) & (traj[:,1] == -100) & (traj[:,2] == -100))))

def add_padding(traj, traj_points, lmbda, overlapping=0, extra_data=[]):
    assert traj.shape[-1] == get_dim_traj_points(extra_data)*lmbda, f'traj shape: {traj.shape} vs. expected shape: N,{get_dim_traj_points(extra_data)*lmbda}'
    if overlapping == 0:
        num_fake_points = ((traj_points//lmbda) - traj.shape[0])
    else:
        max_subsequences = (traj_points-lmbda)//(lmbda-overlapping) + 1
        num_fake_points = (max_subsequences - traj.shape[0])
    return np.pad(traj, pad_width=((0, num_fake_points),(0,0)), constant_values=-100)  # Pad last values with -100

def remove_padding(traj, extra_data=[]):
    assert traj.ndim == 2 and traj.shape[-1] == get_dim_traj_points(extra_data), f'Make sure to reshape the traj correctly before removing padding. ndim:{traj.ndim} | shape:{traj.shape}'
    if is_padded(traj):  # Check if it's actually padded
        first_padding_index = np.where( ((traj[:,0] == -100) & (traj[:,1] == -100) & (traj[:,2] == -100)) )[0][0]
        traj = traj[:first_padding_index, :].copy()
    return traj

def from_seq_to_pc(traj, extra_data):
    """From lambda-sequences to point-cloud"""
    assert traj.ndim == 2
    expected_outdim = get_dim_traj_points(extra_data)
    if traj.shape[-1] == expected_outdim:
        return traj

    traj = traj.reshape(-1, expected_outdim)
    traj = remove_padding(traj, extra_data)
    return traj

def from_pc_to_seq(traj, traj_points, lambda_points, overlapping, extra_data, padding=True):
    """From point-cloud to lambda-sequences"""
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == expected_outdim

    traj = traj.reshape(-1, expected_outdim*lambda_points)
    if padding:
        traj = add_padding(traj, traj_points=traj_points, lmbda=lambda_points, overlapping=overlapping, extra_data=extra_data)
    return traj

def handle_cmap_input(cmap):
    if cmap == 'gt':
        cmap = ['#D84315', '#F57C00', '#FFB300', '#FFEB3B']
    elif cmap == 'pred':
        cmap = ['#2C3E50', '#16A085', '#27AE60', '#2980B9', '#8E44AD']

    return cmap

def visualize_mesh_traj(meshfile,
                        traj,
                        plotter=None,
                        index=None,
                        text=None,
                        trajc='lightblue',
                        trajvel=False,
                        lambda_points=1,
                        camera=None,
                        extra_data=[],
                        stroke_ids=None,
                        cmap=None,
                        arrow_color=None):
    """Visualize mesh-traj pair

    meshfile: str
              mesh filename.obj
    traj : (N,k) array
    
    lambda_points: traj is set of sequences of lambda_points
    """
    curr_traj = traj.copy()

    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    mesh_obj = pv.read(meshfile)
    plotter.add_mesh(mesh_obj)

    if torch.is_tensor(curr_traj):
        curr_traj = curr_traj.cpu().detach().numpy()

    if lambda_points > 1:
        outdim = get_dim_traj_points(extra_data)
        assert curr_traj.shape[-1]%outdim == 0
        curr_traj = curr_traj.reshape(-1, outdim)
        curr_traj = remove_padding(curr_traj, extra_data)  # makes sense only if it's GT traj, but doesn't hurt

    traj_pc = pv.PolyData(curr_traj[:, :3])

    cmap = handle_cmap_input(cmap)

    plotter.add_mesh(traj_pc,
                     color=trajc,
                     # scalars=(None if stroke_ids is None else 60*np.sin(2*math.pi*(1/12)*stroke_ids)),
                     scalars=stroke_ids,
                     point_size=14.0,
                     opacity=1.0,
                     render_points_as_spheres=True,
                     cmap=cmap)

    if trajvel:
        assert 'vel' in extra_data, 'Cannot display traj velocity: trajectory does not contain velocities'
        plotter.add_arrows(curr_traj[:, :3], curr_traj[:, 3:6], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, curr_traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor

        if arrow_color is None:
            arrow_color = 'red'
            
        plotter.add_arrows(curr_traj[:, :3]-e1_rots, e1_rots, mag=1, color=arrow_color, opacity=0.8)

    if camera is not None:
        plotter.set_position(camera)
    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()
    return

def rot_from_representation(orient_repr, arr):
    if orient_repr == 'orientquat':
        return Rot.from_quat(arr)
    elif orient_repr == 'orientrotvec':
        return Rot.from_rotvec(arr)
    elif orient_repr == 'orientnorm':
        return FakeRot(arr)

class FakeRot():
    """Mimics Rot object from scipy, to apply rotations in terms of normals
    (2D pose representation for points)"""
    def __init__(self, normals):
        self.normals = normals

    def apply(self, *args, **kwargs):
        return self.normals
        

def visualize_traj(traj, plotter=None, index=None, text=None, trajc='lightblue', extra_data=[]):
    expected_outdim = get_dim_traj_points(extra_data)
    if traj.shape[-1] != expected_outdim:
        # traj = from_seq_to_pc(traj, extra_data=extra_data)
        raise ValueError('Use visualize_sequence_traj to view lambda-sequences')

    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    if torch.is_tensor(traj):
        traj = traj.cpu().detach().numpy()
    traj_pc = pv.PolyData(traj[:, :3])
    plotter.add_mesh(traj_pc, color=trajc, point_size=10.0, opacity=1.0, render_points_as_spheres=True)

    if get_traj_feature_index('vel', extra_data) is not None:
        indexes = get_traj_feature_index('vel', extra_data)
        plotter.add_arrows(traj[:, :3], traj[:, indexes], mag=1, color='green', opacity=0.8)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)

        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor
        plotter.add_arrows(traj[:, :3]-e1_rots, e1_rots, mag=1, color='red', opacity=0.8)

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show_axes()
        plotter.show()
    return


def visualize_sequence_traj(traj, **args):
    """Visualize traj as groups of sequences (lmbda > 1)

    traj: (N/lmbda, 3*lmbda) array
    """
    expected_outdim = get_dim_traj_points(args['extra_data'])
    assert traj.ndim == 2 and traj.shape[-1] > expected_outdim, f'Make sure to reshape the traj as a group of sequences before visualizing it. ndim:{traj.ndim} | shape:{traj.shape}'

    lmbda = int(traj.shape[-1]/expected_outdim)

    traj = traj.reshape(-1, expected_outdim)
    traj = remove_padding(traj, args['extra_data'])
    traj = traj.reshape(-1, expected_outdim*lmbda)

    n_sequences = traj.shape[0]

    sequence_ids = np.repeat(np.arange(n_sequences), lmbda)  # (N,) new stroke_ids, one for each sequence

    traj = traj.reshape(-1, expected_outdim)
    visualize_complete_traj(traj, sequence_ids, **args)

    return

def visualize_complete_traj(traj, stroke_ids, plotter=None, index=None, text=None, extra_data=[]):
    """Plot trajectory with strokes as different colours
    Input:
        traj: (N, 3) array with x-y-z
        stroke_ids: (N,) with stroke_ids
    """
    expected_outdim = get_dim_traj_points(extra_data)
    assert traj.ndim == 2 and traj.shape[-1] == expected_outdim

    show_plot = True if plotter is None else False

    if plotter is not None:
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    pc = pv.PolyData(traj[:, :3])
    plotter.add_mesh(pc,
                     # scalars=np.multiply(5*stroke_ids, np.sin(30*stroke_ids)),
                     # scalars=list(map(lambda x: zlib.crc32(bytes(x)) % 400, stroke_ids)),  # (https://stackoverflow.com/questions/66055968/python-map-integer-to-another-random-integer-without-random)
                     scalars=np.sin(60*stroke_ids),
                     # scalars=60*np.sin(2*math.pi*(1/12)*stroke_ids),
                     cmap=['#D84315', '#F57C00', '#FFB300', '#FFEB3B'],
                     show_scalar_bar=False,
                     point_size=20.0,
                     opacity=1.0,
                     render_points_as_spheres=True)

    if orient_in(extra_data)[0]:
        orient_repr = orient_in(extra_data)[1]
        indexes = get_traj_feature_index(orient_repr, extra_data)
        e1 = np.array([1, 0, 0])
        rots = rot_from_representation(orient_repr, traj[:, indexes])
        e1_rots = rots.apply(e1)
        down_scale_factor = 10
        e1_rots /= down_scale_factor
        plotter.add_arrows(traj[:, :3]-e1_rots, e1_rots, mag=1, color='red', opacity=0.8)

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        plotter.show()
    return


def visualize_pc(pc, plotter=None, index=None, text=None):
    """Visualize point-cloud"""

    show_plot = True if plotter is None else False

    if plotter is not None:
        assert index is not None, 'index is None but plotter is not None'
        plotter.subplot(*index)
    else:
        plotter = pv.Plotter(shape=(1, 1), window_size=(1920,1080))
        plotter.subplot(0,0)

    if torch.is_tensor(pc):
        pc = pc.cpu().detach().numpy()
    pc = pv.PolyData(pc[:, :3])
    plotter.add_mesh(pc, point_size=6.0, opacity=1.0, render_points_as_spheres=True)

    if text is not None:
        plotter.add_text(text)
    if show_plot:
        # plotter.set_position([0, -5, 0])
        plotter.show_axes()
        plotter.show()

    return


def mean_knn_distance(point_cloud, k=2, render=False, y_lengths=None):
    """Visualization function for computing k-NNs of point-cloud"""
    # assert point_cloud.ndim == 3, 'point-cloud does not contain batch dimension'
    if point_cloud.ndim == 2:
        point_cloud = point_cloud[np.newaxis, :, :]

    if not torch.is_tensor(point_cloud):
        point_cloud = torch.tensor(point_cloud)

    B, _, _ = point_cloud.size()

    """
        mean k-nn distance histogram
    """
    distances = torch.cdist(point_cloud, point_cloud, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

    top_dists = top_dists[:, :, 1:]  # Remove self-distance
    top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))

    top_dists = torch.mean(top_dists, dim=-1) # B, traj_points

    if y_lengths is not None:
        mask = torch.arange(point_cloud.shape[1], device=point_cloud.device)[None] >= y_lengths[:, None]
        top_dists[mask] = 0.0 
        top_dists_per_batch = top_dists.sum(1) / y_lengths
    else:
        top_dists_per_batch = torch.mean(top_dists, dim=-1)  # B, 

    if render:
        for b in range(B):
            top_dists_b = top_dists[b, :]
            top_dists_b = top_dists_b.flatten()

            top_dists_b = top_dists_b.detach().cpu().numpy()
            sns.histplot(top_dists_b)  # binwidth=0.0001
            plt.show()

    return top_dists_per_batch


def get_train_config(config_file):
    if config_file is None:
        config_file = 'default.json'
    with open(os.path.join('configs', config_file)) as json_file:
        data = json.load(json_file)
    return data

def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
        # pprint(vars(config), stream=file)
        json.dump(config, file)
    return

def load_config(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def set_seed(seed):
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

def create_dir(path):
    try:
        os.mkdir(os.path.join(path))
    except OSError as error:
        pass

def create_dirs(path):
    try:
        os.makedirs(os.path.join(path))
    except OSError as error:
        pass


def get_dataset_downscale_factor(dataset_name):
    mean_max_distance = {
            'containers-v2': 884.1423249856435,
            'cuboids-v1': 888.7967305471634,
            'shelves-v1': 905.4091900499023,
            'windows-v1': 1157.9744613449216
    }
    if dataset_name not in mean_max_distance:
        return None
    else:
        return mean_max_distance[dataset_name]


def get_dataset_path(category):
    """Returns dir path where files are stored.

    category : str
               e.g. cuboids-v1, windows-v1, shelves-v1, containers-v2
    """
    print(category)
    print(os.environ.get("PAINTNET_ROOT"))
    assert os.environ.get("PAINTNET_ROOT") is not None, "Set PAINTNET_ROOT environment variable to localize the paintnet dataset root."
    assert os.path.isdir(os.environ.get("PAINTNET_ROOT")), f'Dataset root path was set but does not exist on current system. Path: {s.environ.get("PAINTNET_ROOT")}'
    assert os.path.isdir(os.path.join(os.environ.get("PAINTNET_ROOT"), category)), 'current dataset category {category} does not exist on your system.'
    
    return os.path.join(os.environ.get("PAINTNET_ROOT"), category)