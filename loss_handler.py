"""Handler class for loss function terms

To add a loss term:
    - insert its name and its method name in the constructor
    - add the method implementation itself
    - add a --weight_<lossname> arg parameter
"""

import pdb

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
try:
    import torch_nndistance as NND   
except ImportError:
    print('Warning! Unable to import torch_nndistance package. Chamfer distance won\'t be available.')
    pass
try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass


from paintnet_utils import *
# from models.dgcnn import DGCNNDiscriminator
from models.gradient_penalty import GradientPenalty


class LossHandler():
    def __init__(self, loss, config=None):
        """
        loss : list of str
                   list of loss terms, each weighted by the
                   corresponding specified weight as command argument
        config : dict with loss term weights
        """
        self.loss_names =   ['chamfer',
                             'repulsion',
                             'mse',
                             'align',
                             'velcosine',
                             'intra_align',
                             'discriminator',
                             'wdiscriminator',
                             'attraction_chamfer',
                             'rich_attraction_chamfer']
        self.loss_methods = [self.get_chamfer,
                             self.get_repulsion,
                             self.get_mse,
                             self.get_align_loss,
                             self.get_vel_cosine,
                             self.get_intra_align,
                             self.get_discr_loss,
                             self.get_wdiscr_loss,
                             self.get_attraction_chamfer,
                             self.get_rich_attraction_chamfer]

        self.loss_index = {loss_name: i for i, loss_name in enumerate(self.loss_names)}
        assert (set(loss) <= set(self.loss_names)), f'Specified loss list {loss} contains non-valid names ({self.loss_names})'

        self.loss = list(loss)
        self.config = config


        """
            Loss initializations
        """
        # if 'discriminator' in self.loss:  # Initialize discriminator
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.D = DGCNNDiscriminator(inputdim=3, k=self.config['knn_gcn']).to(self.device)
            
        #     self.minimax_loss = nn.BCEWithLogitsLoss().cuda()
        #     self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.9, 0.999))

        # if 'wdiscriminator' in self.loss:  # Initialize wasserstein discriminator
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.D = DGCNNDiscriminator(inputdim=3, k=self.config['knn_gcn']).to(self.device)
            
        #     self.GradPenalty = GradientPenalty(self.config['discr_lambdaGP'], gamma=1, device=self.device)
        #     self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.9, 0.999))

        """
            Asserts for loss compatibility
        """
        for l in self.loss:
            assert 'weight_'+str(l) in self.config.keys(), f'weight parameter does not exist in the current config' \
            f' for loss {l}. Make sure to include a --weight_<loss_name> arg par for each loss you use.'

        assert not ('chamfer' in self.loss and 'mse' in self.loss), f'Incompatible losses: chamfer with mse'

        if self.config['lambda_points'] > 1:
            assert set(loss) <=  {'chamfer', 'intra_align', 'attraction_chamfer', 'rich_attraction_chamfer', 'repulsion'}

        assert not ('discriminator' in self.loss and 'wdiscriminator' in self.loss), 'Choose between Minimax discriminator and Wasserstein Discriminator'

        if 'intra_align' in self.loss:
            assert self.config['lambda_points'] > 3, 'Fitting a plane to 3 points in 3D would always have degenerate covariance matrix.'
        if 'align' in self.loss:
            assert 'mse' not in self.loss, 'Align loss is not meant to be used with MSE'
            assert self.config['knn_repulsion'] > 1, 'Using Align loss with 1 NN -> unexplained variance would always be zero.'
        if 'attraction_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1
        if 'rich_attraction_chamfer' in self.loss:
            assert self.config['lambda_points'] > 1
            assert orient_in(self.config['extra_data'])[0]
            assert 'vel' not in self.config['extra_data']
        return


    def compute(self, y_pred, y, return_list=True, **loss_args):
        """Return loss function

        return_list: bool
                     if True, additionally return seperate loss terms as list
        """
        loss_val = 0
        loss_val_list = []

        for l in self.loss:  # Compute each loss term
            l_ind = self.loss_index[l]
            l_value = self.loss_methods[l_ind](  **{"y_pred": y_pred, "y": y, **loss_args}  )  # (y_pred, y, **loss_args) as input parameters

            loss_val += self.config['weight_'+str(l)]*l_value  # Weight * loss_term
            loss_val_list.append(l_value.detach().cpu().numpy())

        if return_list:
            return loss_val, np.array(loss_val_list)
        else:
            return loss_val


    def log_on_wandb(self, loss_list, epoch, wandb, suffix='_train_loss'):
        """Log loss list on wandb"""
        loss_list_names = self.loss.copy()

        if 'discriminator' in self.loss or 'wdiscriminator' in self.loss:
            loss_list_names.append('discr_internal')
            loss_list = np.append(loss_list, self.last_discr_internal_loss.detach().cpu().numpy())

        for loss_term, train_loss_term in zip(loss_list_names, loss_list):
                wandb.log({str(loss_term)+str(suffix): train_loss_term, "epoch": (epoch+1)})



    """
        
        Loss list

    """
    def get_discr_loss(self, y_pred, y, **args):
        """A discriminator is used to learn a loss
        function adversarially (mesh-agnostic).

        """
        y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1) # B, 3, pc_points

        ###### DISCRIMINATOR TRAINING ######
        if 'train' not in args or args['train'] == True:
            self.D.train()
            self.D.zero_grad()

            real_out = self.D(y)
            real_loss = self.minimax_loss(real_out, Variable(torch.ones(real_out.size()).to(self.device))) # -log(D(traj_real))

            fake_out = self.D(y_pred.detach())
            fake_loss = self.minimax_loss(fake_out, Variable(torch.zeros(fake_out.size()).to(self.device)))  # -log(1-D(traj_predicted))

            d_loss = self.config['weight_discr_training']*(real_loss + fake_loss)

            
            d_loss.backward()
            self.D_optimizer.step()

            self.last_discr_internal_loss = d_loss
        else:
            self.D.train(False)
            self.last_discr_internal_loss = torch.zeros(1)
        ####################################


        ###### Learned loss term #########
        D_out = self.D(y_pred)

        learned_loss = self.minimax_loss(D_out, Variable(torch.ones(D_out.size()).to(self.device)))  # -log(D(traj_predicted))
        ####################################

        return learned_loss


    def get_wdiscr_loss(self, y_pred, y, **args):
        """Wasserstein-loss discriminator
        https://github.com/jtpils/TreeGAN
        """
        y, y_pred = y.permute(0, 2, 1), y_pred.permute(0, 2, 1) # B, 3, pc_points

        # -------------------- Discriminator -------------------- #
        if 'train' not in args or args['train'] == True:
            self.D.train()
            for d_iter in range(self.config['discr_train_iter']):
                self.D.zero_grad()
                
                # z = torch.randn(self.args.batch_size, 1, 96).to(args.device)
                # tree = [z]
                
                # with torch.no_grad():
                #     fake_point = self.G(tree)         
                    
                D_real = self.D(y)
                D_realm = D_real.mean()

                D_fake = self.D(y_pred.detach())
                D_fakem = D_fake.mean()

                gp_loss = self.GradPenalty(self.D, y.data, y_pred.data)
                
                d_loss = -D_realm + D_fakem
                d_loss_gp = d_loss + gp_loss

                d_loss_gp.backward()
                self.D_optimizer.step()

                self.last_discr_internal_loss = d_loss_gp
        else:
            self.D.train(False)
            self.last_discr_internal_loss = torch.zeros(1)

        
        # ---------------------- Generator ---------------------- #
        G_fake = self.D(y_pred)
        G_fakem = G_fake.mean()
        g_loss = -G_fakem

        return g_loss

    def get_rich_attraction_chamfer(self, y_pred, **args):
        """first and last points are enriched with orientation and inferred velocity.

        See attraction_loss for the standard version.
        """
        outdim = get_dim_traj_points(self.config['extra_data'])

        starting_points = y_pred[:, :, :outdim]
        ending_points = y_pred[:, :, -outdim:]      

        inferred_vel_starting = y_pred[:, :, outdim:outdim+3] - y_pred[:, :, :3]
        inferred_vel_ending = y_pred[:, :, -outdim:-(outdim-3)] - y_pred[:, :, -(outdim*2):-(outdim*2-3)]

        starting_points = torch.cat((starting_points, inferred_vel_starting), dim=-1)
        ending_points = torch.cat((ending_points, inferred_vel_starting), dim=-1)

        if not self.config['soft_attraction']:
            # Full version (all points get attracted, a different 2nd-nn sequence is taken into account in case same sequence is 1st-nn)
            chamfer = 100*chamfer_distance(starting_points, ending_points, padded=False, avoid_in_sequence_collapsing=True)[0]
        else:
            # Soft version (only a few points are attracted, those whose 1-nn is not in-sequence)
            chamfer = 100*chamfer_distance(starting_points,
                                           ending_points,
                                           padded=False,
                                           avoid_in_sequence_collapsing=True,
                                           soft_attraction=True,
                                           point_reduction=None,
                                           batch_reduction=None)[0]

        return chamfer

    def get_attraction_chamfer(self, y_pred, **args):
        """Chamfer loss between ending points (1st point-cloud) and starting points (2nd point-cloud).

        It encourages predicted mini-sequences to be contiguous.
        """
        starting_points = y_pred[:, :, :3]
        ending_points = y_pred[:, :, -3:]

        chamfer = 100*chamfer_distance(starting_points, ending_points, padded=False)[0]
        return chamfer

    def get_chamfer(self, y_pred, y, **args):
        if 'vel' in self.config['extra_data']:  # Fallback to custom chamfer distance for velocities
            chamfer = 100*chamfer_distance(y_pred, y, velocities=True)[0]
        
        elif self.config['lambda_points'] > 1:
            chamfer = 100*chamfer_distance(y_pred, y, padded=True, min_centroids=self.config['min_centroids'])[0]  # Handle padded GT trajs for dataloader

        else:
            dist1, dist2, _, _ = NND.nnd(y_pred, y)  # Chamfer loss
            chamfer = (100 * (torch.mean(dist1) + torch.mean(dist2))) # Chamfer is weighted by 100

        return chamfer


    def get_repulsion(self, y_pred, y, **args):
        if 'mse' in self.loss:  # Ordered repulsion if MSE is used
            return self.get_ordered_repulsion(y_pred, y, **args)
        elif 'chamfer' in self.loss:
            return self.get_unordered_repulsion(y_pred, y, **args)
        else:  # Fallback to unordered repulsion
            return self.get_unordered_repulsion(y_pred, y, **args)


    def get_unordered_repulsion(self, y_pred, y, **args):
        outdim = get_dim_traj_points(self.config['extra_data'])

        B = y_pred.shape[0]  # Batch size

        if self.config['lambda_points'] > 1:
            # traj_pred = from_seq_to_pc(y_pred.clone(), extra_data=self.config['extra_data'])
            traj_pc = y_pred.view(B, -1, outdim)
        else:
            traj_pc = y_pred

        traj_pc = traj_pc[:, :, :3]

        if self.config['rep_target'] is not None:
            target_dist = self.config['rep_target']
        else:
            y_lengths = None
            if self.config['lambda_points'] > 1:
                ridx, cidx = torch.where(y[:,:,0] == -100)
                y_lengths = []
                for b in range(B):
                    y_lengths.append(cidx[torch.argmax((ridx == b).type(torch.IntTensor))].item())
                y_lengths = torch.tensor(y_lengths, device=y.device)

            target_dist = mean_knn_distance(y[:, :, :3], k=self.config['knn_repulsion'], y_lengths=y_lengths)

        k = self.config['knn_repulsion']
        h = target_dist*np.sqrt(2)
        distances = torch.cdist(traj_pc, traj_pc, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

        top_dists = top_dists[:, :, 1:]  # Remove self-distance
        top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))  # Regularization

        if torch.is_tensor(h) and h.ndim == 1:
            h = h.view(B,1,1)  # For broadcasting

        weight = torch.exp(-(top_dists.square())/(h**2))

        rep = 100*torch.mean(-top_dists*weight)  # Repulsion loss is weighted by 100

        return rep


    def get_ordered_repulsion(self, y_pred, y, **args):
        raise NotImplementedError('If you want to use MSE with repulsion, change the get_repulsion method temporarily.')
        return



    def get_align_loss(self, y_pred, **args):
        # Generate some data that lies along a line

        # x = np.mgrid[-2:5:120j]
        # y = np.mgrid[1:9:120j]
        # z = np.mgrid[-5:3:120j]

        # data = np.concatenate((x[:, np.newaxis], 
        #                        y[:, np.newaxis], 
        #                        z[:, np.newaxis]), 
        #                       axis=1)

        # # Perturb with some Gaussian noise
        # data += np.random.normal(size=data.shape) * 0.4

        # # Calculate the mean of the points, i.e. the 'center' of the cloud
        # datamean = data.mean(axis=0)

        # y_pred_mean = y_pred.mean(axis=1)
        # y_pred_mean = y_pred_mean[:, np.newaxis, :]

        # pdb.set_trace()

        # Do an SVD on the mean-centered data.
        # S = torch.linalg.svdvals(y_pred - y_pred_mean)  # Returns singular values of input matrix
        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.

        # y = y[:, :, :3]
        y_pred = y_pred[:, :, :3] 

        B = y_pred.shape[0]  # Batch size
        traj_points = y_pred.shape[1]  # Traj_points

        k = self.config['knn_repulsion']
        distances = torch.cdist(y_pred, y_pred, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)
        
        # top_dists = top_dists[:, :, 1:]  # Remove self-distance
        # top_dists = torch.maximum(top_dists, torch.tensor([1e-12]).to(top_dists.device))
        # ind = ind[:, :, 1:]

        # tot_unexp_var = 0
        # for b, batch in enumerate(ind): # per batch in indices of top distances
        #     unexplained_variance = 0
        #     for indices in batch:  # per point, consider its k-NNs
        #         # current_point = indices[0]
        #         # nns = indices[1:] # k-NNs indices
        #         data = y_pred[b, indices, :] # considering itself and its k-NNs
        #         datamean = data.mean(axis=0)
        #         S = torch.linalg.svdvals(data - datamean) # singuar values of itself and its k-NNs
        #         unexplained_variance += S[1:].sum()
        #     tot_unexp_var += (unexplained_variance / traj_points)
        # tot_unexp_var /= B

        tot_unexp_var2 = 0
        for b, batch in enumerate(ind):
            unexplained_variance2 = 0

            data = y_pred[b, ind[b, :, :], :]
            datamean = data.mean(axis=-2)
            datamean = datamean[:, None, :]

            S = torch.linalg.svdvals(data - datamean)

            unexplained_variance2 = S[:, 1:].sum(axis=-1)

            tot_unexp_var2 += unexplained_variance2.mean()

        tot_unexp_var2 /= B

        # assert tot_unexp_var == tot_unexp_var2, f'NON ERA UGUALE QUI 1: {tot_unexp_var} ||| 2: {tot_unexp_var2}'
        return tot_unexp_var2



    def get_intra_align(self, y_pred, **args):
        """Encourage sub-sequences to lay on planes

        Fit a plane to points in each sequence,
        and penalizes least-squares to plane.
        """
        B, N_seq, outdim = y_pred.size()
        lmbda = outdim//3

        # tot_unexp_variance = 0
        # for b in range(B):
        #     flatten_data = y_pred[b, :, :].view(-1, 3)  # (traj_points, 3)
        #     slices = torch.arange(0, flatten_data.shape[0]).view((flatten_data.shape[0]//lmbda), lmbda)
        #     data = flatten_data[slices, :]  # (N_seq, lmbda, 3)
        #     datamean = data.mean(axis=-2)
        #     zeromean = (data-datamean[:, None, :])
        #     S = torch.linalg.svdvals(zeromean)
        #     unexplained_variance = S[:, 2]  # Last singular value per sequence
        #     tot_unexp_variance += unexplained_variance.mean()

        flatten_data = y_pred.view(B, -1, 3)  # (B, traj_points, 3)
        slices = torch.arange(0, flatten_data.shape[1]).view((flatten_data.shape[1]//lmbda), lmbda)

        data = flatten_data[:, slices, :]  # (B, N_seq, lmbda, 3)

        datamean = data.mean(axis=-2)
        zeromean = (data-datamean[:, :, None, :])

        S = torch.linalg.svdvals(zeromean)

        unexplained_variance = S[:, :, 2]  # Last singular value

        return unexplained_variance.mean()



    def get_vel_cosine(self, y_pred, **args):
        """Encourage each point's velocity to be close to
        the mean of velocities of k-NNs, in terms of
        cosine similarity"""


        # todo: direct to unordered and ordered, depending on whether chamfer and lambda_points are considered.
        assert 'vel' in self.config['extra_data'], 'Velocity cosine loss cannot be used if velocities are not learned.'

        # input1 = torch.randn(100, 128)
        # input2 = torch.randn(100, 128)
        # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # output = cos(input1, input2)

        y_pred_vel = y_pred[:, :, 3:]
        y_pred_pos = y_pred[:, :, :3]

        B = y_pred.shape[0]  # Batch size
        traj_points = y_pred.shape[1]  # Traj_points

        k = self.config['knn_repulsion']
        distances = torch.cdist(y_pred_pos, y_pred_pos, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        top_dists, ind = distances.topk(k+1, largest=False, sorted=True, dim=-1)

        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        tot_cos = 0
        for b, batch in enumerate(ind):
            # current_point = indices[0]
            # nns = indices[1:] # k-NNs indices

            curr_points = ind[b, :, 0] # Indices of curr points
            nns = ind[b, :, 1:]  # Indices of k-NNs

            curr_vels = y_pred_vel[b, curr_points, :] # Vel of curr points
            vel_nns = y_pred_vel[b, nns, :] # Velocities of k-NNs
            mean_vel_nns = vel_nns.mean(axis=-2)  # Mean vel of k-NNs

            tot_cos += cos(curr_vels, mean_vel_nns).mean()
            
        tot_cos /= B

        return -tot_cos

    def get_mse(self, y_pred, y, **args):
        return F.mse_loss(y_pred, y)
