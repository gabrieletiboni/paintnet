"""Class for implementing and computing evaluation metrics

E.g. pose-wise chamfer distance between predicted mini-sequences
and ground-truth on the test set.
"""
import numpy as np
import torch

from paintnet_utils import *
try:
    from pytorch3d_chamfer import chamfer_distance 
except ImportError:
    print(f'Warning! Unable to import pytorch3d package.'\
          f'Chamfer distance with velocities won\'t be available.'\
          f'(Check troubleshooting.txt for info on how to install pytorch3d)')
    pass

class MetricsHandler():
    """Handle computation of evaluation metrics.

    E.g. compute pose-wise chamfer distance between
    predicted mini-sequences and ground-truth
    on the test set.
    """

    def __init__(self,
                 config,
                 metrics=[],
                ):
        """
        Parameters:
            metrics : list of str
                      metrics to be computed
        """
        super(MetricsHandler, self).__init__()
        self.metrics = metrics
        self.metrics_names = [
                    'pcd',
                    'chamfer_original',  # Unbalanced: comparison is done with entire original GT point-cloud (untrimmed due to lambda-sequences)
                    'stroke_chamfer'
                  ]
        self.metric_functions = [
                    self.get_pcd,
                    self.get_chamfer_original,
                    self.get_stroke_chamfer
                  ]

        self.metric_index = {metric: i for i, metric in enumerate(self.metrics_names)}
        self.config = config
        
    def get_eval_metric(self, y_pred, y, metric, **args):
        """Compute single metric"""
        assert metric in self.metrics_names, f"metric {metric} is not valid"
        metric = self.metric_functions[self.metric_index[metric]](**{"y_pred": y_pred, "y": y, **args})
        return metric

    def compute(self, y_pred, y, **args):
        """Compute all metrics in self.metrics
        and returns them in a list"""
        assert len(self.metrics) > 0
        metrics = [0 for _ in self.metrics]

        for m, metric in enumerate(self.metrics):
            metrics[m] = self.get_eval_metric(y_pred, y, metric=metric, **args).detach().cpu().numpy()
        return np.array(metrics)


    # def summary_on_wandb(self, metric_values, wandb, suffix=''):
    #     """Log metrics on wandb as a summary"""
    #     assert len(metric_values) == len(self.metrics)

    #     for name, value in zip(self.metrics, metric_values):
    #             # wandb.log({str(name)+str(suffix): value})
    #             wandb.run.summary[f"{name}{suffix}"] = value

    def log_on_wandb(self, metric_values, wandb, suffix=''):
        """Log metrics on wandb"""
        assert len(metric_values) == len(self.metrics)
        
        for name, value in zip(self.metrics, metric_values):
                wandb.log({str(name)+str(suffix): value})

    def pprint(self, metric_values):
        """Pretty print metric values"""
        for name, value in zip(self.metrics, metric_values):
            print(f"{name}:\t{round(value, 3)}")
        print('------------')




    """
    
        EVALUATION METRICS

    """
    def get_pcd(self, y_pred, y, **args):
        """Pose-wise Chamfer Distance between predictions and ground-truth poses"""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)
            y = y.reshape(B, -1, outdim)

        traj_pred_pc = torch.tensor(y_pred)
        traj_pc = torch.tensor(y)

        # print('effective points pred:', traj_pred_pc.shape[1])
        # print('effective points GT:', traj_pc.shape[1])

        if self.config['lambda_points'] > 1:
            chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_pc, padded=True)[0]
        else:
            chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_pc)[0]

        return chamfer


    def get_chamfer_original(self, y_pred, y, traj_pc, **args):
        """Chamfer between predictions and full, untrimmed ground truth traj_pc.

        trimming may happen because of splitting into lambda-sequences,
        but nevertheless it generally just skips a few poses."""
        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        if self.config['lambda_points'] > 1:
            y_pred = y_pred.reshape(B, -1, outdim)

        traj_pred_pc = torch.tensor(y_pred)

        print('effective points pred:', traj_pred_pc.shape[1])
        print('effective points GT original:', traj_pc.shape[1])

        chamfer = (10**4)*chamfer_distance(traj_pred_pc, traj_pc)[0]
        return chamfer

    def get_stroke_chamfer(self, y_pred, y, traj_pc, stroke_ids, **args):
        """Debug: chamfer between predicted vectors and original strokes,
        with inner distance metric as an additional chamfer distance."""
        asymmetric = True
        print(f'---\nCAREFUL! Stroke-wise chamfer is with ASYMMETRIC={asymmetric}\n---')

        B = y_pred.shape[0]
        outdim = get_dim_traj_points(self.config['extra_data'])

        traj_pred = torch.tensor(y_pred)
        
        ##### 1° version
        chamfers = [0 for b in range(B)]
        for b in range(B):
            chamfer = 0

            n_pred_strokes = y_pred.shape[1]
            n_gt_strokes = stroke_ids[b, -1]+1
            unique, counts = np.unique(stroke_ids[b], return_counts=True)
            assert len(unique) == n_gt_strokes
            for i in range(n_pred_strokes):
                min_chamfer = 10000000
                
                pred_pc = traj_pred[b, i].view(-1, outdim)[None, :, :]
                for i_gt in range(n_gt_strokes):
                    curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
                    curr_chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=asymmetric)[0]
                    # dist1, dist2, _, _ = NND.nnd(pred_pc, curr_gt_pc)  # Chamfer loss
                    # chamfer = (10**4)*(torch.mean(dist1))

                    min_chamfer = min(min_chamfer, curr_chamfer.item())

                chamfer += min_chamfer

            chamfers[b] = chamfer/n_pred_strokes

        chamfers = np.array(chamfers).mean()
        ##############################

        ##### 2° version (would require stroke-padding, so it currently does not work)
        # batch_stroke_chamfer = torch.empty((B, 0))

        # n_pred_strokes = y_pred.shape[1]
        # min_chamfer = torch.ones((B,))*10000000
        # for i in range(n_pred_strokes):

        #     pred_pc = traj_pred[:, i, :].view(B, -1, outdim)
        #     for i_gt in range(n_gt_strokes):
        #         curr_gt_pc = traj_pc[b, stroke_ids[b, :] == i_gt, :][None, :, :]
        #         chamfer = (10**4)*chamfer_distance(pred_pc, curr_gt_pc, asymmetric=True)[0]
        ##############################
        return chamfers