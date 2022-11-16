# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import pdb

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    velocities=False,
    min_centroids=False,
    padded=False,
    avoid_in_sequence_collapsing=False,
    soft_attraction=False,
    asymmetric=False
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    if not soft_attraction:
        _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    if padded:  # Overwrite y_lengths to handle custom padding of [-100, -100, -100] points
        ridx, cidx = torch.where(y[:,:,0] == -100)
        new_y_lengths = []
        for b in range(N):
            new_y_lengths.append(cidx[torch.argmax((ridx == b).type(torch.IntTensor))].item())
        new_y_lengths = torch.tensor(new_y_lengths, device=x.device)
        y_lengths[:] = new_y_lengths

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    if velocities:  # Custom chamfer that does not take into account velocities when computing min
        assert D == 6, 'Velocities is True but traj does not contain velocities'
        x_nn_pos = knn_points(x[:,:,:3], y[:,:,:3], lengths1=x_lengths, lengths2=y_lengths, K=1)
        y_nn_pos = knn_points(y[:,:,:3], x[:,:,:3], lengths1=y_lengths, lengths2=x_lengths, K=1)

        x_indx = x_nn_pos.idx
        x_dists = x_nn_pos.dists

        y_indx = y_nn_pos.idx
        y_dists = y_nn_pos.dists

        cham_x = x.new_zeros((N, P1))
        cham_y = y.new_zeros((N, P1))
        for b, batch in enumerate(x_indx):
            x_knn_dist_b = y[b, x_indx[b, :, 0], :]
            cham_x[b, :] = torch.linalg.norm(x[b,:] - x_knn_dist_b, axis=-1).square()

            y_knn_dist_b = x[b, y_indx[b, :, 0], :]
            cham_y[b, :] = torch.linalg.norm(y[b,:] - y_knn_dist_b, axis=-1).square()

    elif avoid_in_sequence_collapsing:
        assert P1 == P2
        seq_ids = torch.arange(P1).cuda()

        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=2)
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=2)


        if not soft_attraction:
            cham_x = torch.empty((N)).cuda()
            cham_y = torch.empty((N)).cuda()
            for b in range(N):
                cham_x_sum = x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids != 0, 0].sum()
                cham_x_sum += x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids == 0, 1].sum()
                cham_x[b] = cham_x_sum

                cham_y_sum = y_nn.dists[b, y_nn.idx[b, :, 0]-seq_ids != 0, 0].sum()
                cham_y_sum += y_nn.dists[b, y_nn.idx[b, :, 0]-seq_ids == 0, 1].sum()
                cham_y[b] = cham_y_sum

                assert x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids != 0, 0].shape[0] + \
                       x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids == 0, 1].shape[0] == P1
        else:
            assert point_reduction is None and batch_reduction is None

            cham_x = torch.empty((0,)).cuda()
            cham_y = torch.empty((0,)).cuda()
            for b in range(N):
                cham_x_mean = x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids != 0, 0].mean()
                cham_x = torch.cat((cham_x, cham_x_mean.view(1)))

                cham_y_mean = y_nn.dists[b, y_nn.idx[b, :, 0]-seq_ids != 0, 0].mean()
                cham_y = torch.cat((cham_y, cham_y_mean.view(1)))

                assert x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids != 0, 0].shape[0] + \
                       x_nn.dists[b, x_nn.idx[b, :, 0]-seq_ids == 0, 1].shape[0] == P1

            cham_x = cham_x.mean()
            cham_y = cham_y.mean()

    else:

        if min_centroids:  # Compute chamfer on centroids only
            assert P1 == P2
            assert D%3 == 0
            lmbda = int(D/3)

            y = y.unsqueeze(-1)
            y = y.view(N, P1, lmbda, 3)
            y = y.mean(axis=-2)

            x = x.unsqueeze(-1)
            x = x.view(N, P1, lmbda, 3)
            x = x.mean(axis=-2)

        x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
        y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

        cham_x = x_nn.dists[..., 0]  # (N, P1)
        cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    if not avoid_in_sequence_collapsing:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    if not asymmetric:
        cham_dist = cham_x + cham_y
    else:
        cham_dist = cham_x
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals