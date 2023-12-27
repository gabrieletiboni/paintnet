import os
import argparse
import numpy as np
import torch
import time
from paintnet_utils import *
from paintnet_loader import PaintNetDataloader
from model_utils import get_model, init_from_pretrained
from loss_handler import LossHandler
os.environ["PAINTNET_ROOT"] = "./dataset"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', default=None)
    parser.add_argument('--dataset',        default='cuboids-v1', type=str, help='Dataset name [containers-v2, windows-v1, shelves-v1, cuboids-v1]')
    parser.add_argument('--name',           default=None, type=str, help='Run name suffix')
    parser.add_argument('--group',          default=None, type=str, help='Wandb group name')
    parser.add_argument('--backbone',       default='pointnet', type=str, help='Backbone [pointnet2]')
    parser.add_argument('--pretrained',     default=False, action='store_true', help='If exists, loads a pretrained model as starting backbone for global features encoder.')
    parser.add_argument('--lambda_points',  default=1, type=int, help='Traj is considered as point-cloud made of vectors of <lambda> ordered points (Default=1, meaning that'\
                                                                     'chamfer distance would be computed normally on each traj point)')
    parser.add_argument('--min_centroids',  default=False, action='store_true', help='Whether to compute chamfer distance on mini-sequences with centroids only')
    parser.add_argument('--overlapping',    default=0, type=int, help='Number of overlapping points between subsequent mini-sequences (only valid when lambda_points > 1)')
    parser.add_argument('--pc_points',      default=5120, type=int, help='Number of points to sub-sample for each point-cloud')
    parser.add_argument('--traj_points',    default=500, type=int, help='Number of points to sub-sample for each trajectory')
    parser.add_argument('--augmentations',  default=[], type=str, nargs='+', help='List of str [rot, roty, rotx]')
    parser.add_argument('--normalization',  default='per-dataset', type=str, help='Normalization for mesh, traj pairs. (per-mesh, per-dataset, none)')
    parser.add_argument('--extra_data',     default=[], type=str, nargs='+', help="list of str [vel, orientquat, orientrotvec, orientnorm]")
    parser.add_argument('--config',         default=None, type=str, help='name of .json file in configs/ dir')
    parser.add_argument('--epochs',         default=200, type=int, help='Training epochs')
    parser.add_argument('--steplr',         default=100, type=int, help='Step learning rate every <steplr> epocgs')
    parser.add_argument('--batch_size', '-bs', default=8, type=int)
    parser.add_argument('--lr',             default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--workers',        default=0, type=int, help='Number of workers for datasetloader')
    parser.add_argument('--loss',           default=['chamfer'], type=str, nargs='+', help='List of str with Loss name (chamfer, repulsion, mse)')
    parser.add_argument('--eval_metrics',   default=['pcd'], type=str, nargs='+', help='Eval metrics [pcd: pose-wise chamfer distance, ...]')
    parser.add_argument('--weight_orient',  default=1.0, type=float, help='Weight for L2-norm between orientation w.r.t. positional L2-norm')
    parser.add_argument('--eval_freq',      default=100, type=int, help='Evaluate model on test set and save it every <eval_freq> epochs')
    parser.add_argument('--output_dir',     default='runs', type=str, help='Dir for saving models and results')
    parser.add_argument('--notes',          default=None, type=str, help='wandb notes')
    parser.add_argument('--debug',          default=False, action='store_true', help='debug mode: no wandb')
    parser.add_argument('--overfitting',    default=False, action='store_true', help='Overfit on a single sample -> index=--seed')
    parser.add_argument('--no_save',        default=False, action='store_true', help='If set, avoids saving .npy of some final results')
    parser.add_argument('--eval_ckpt',      default='best', type=str, help='Checkpoint for evaluating final results (best, last)')
    parser.add_argument('--seed',           default=0, type=int, help='Random seed (not set when equal to zero)')

    # Loss weights
    parser.add_argument('--weight_chamfer',         default=1., type=float, help='Weight for chamfer distance')
    parser.add_argument('--weight_attraction_chamfer', default=1., type=float, help='Weight for attraction chamfer loss')
    parser.add_argument('--weight_rich_attraction_chamfer', default=1., type=float, help='Weight for rich attraction chamfer loss')
    parser.add_argument('--soft_attraction',        default=False, action='store_true', help='Soft version of attraction loss')
    parser.add_argument('--weight_repulsion',       default=1., type=float, help='Weight for repulsion loss')
    parser.add_argument('--weight_mse',             default=1., type=float, help='Weight for mse loss')
    parser.add_argument('--weight_align',           default=1., type=float, help='Weight for align loss')
    parser.add_argument('--weight_velcosine',       default=1., type=float, help='Weight for velocity-cosine attraction loss')
    parser.add_argument('--weight_intra_align',     default=1., type=float, help='Weight for intra-align loss')
    parser.add_argument('--weight_discriminator',   default=1., type=float, help='Weight for learned discriminator loss')
    parser.add_argument('--weight_discr_training',  default=1., type=float, help='Weight for the discriminator training loss')
    parser.add_argument('--weight_wdiscriminator',  default=1., type=float, help='Weight for learned discriminator loss')
    parser.add_argument('--discr_train_iter',       default=1, type=int, help='Iterations of discr training on a single batch')
    parser.add_argument('--discr_lambdaGP',         default=10, type=int, help='Lambda for GP term.')
    
    # Debug
    parser.add_argument('--rep_target',             default=None, type=float, help='DEBUG: target repulsion distance')
    parser.add_argument('--knn_repulsion',          default=1, type=int, help='Number of nearest neighbors to consider when using repulsion loss')
    parser.add_argument('--knn_gcn',                default=20, type=int, help='K value for adj matrix during GCN computation')
    return parser.parse_args()


args = parse_args()
config = get_train_config(args.config)
config = {**args.__dict__, **config}

def test(model, loader, loss_handler, metrics_handler=None, save=False, **save_args):
    """Test model on dataloader"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tot_loss = 0
    tot_loss_list = np.zeros(len(loss_handler.loss))
    data_count = 0
    if metrics_handler is not None:
        metrics = [0 for _ in metrics_handler.metrics]

    for i, data in enumerate(loader):
        point_cloud, traj, dirnames = data
        # plot
        plotter = pv.Plotter(shape=(1,1), window_size=(1920,1080), off_screen=True)
        B, N, dim = point_cloud.size()
        data_count += B
        point_cloud = point_cloud.permute(0, 2, 1) # B, 3, pc_points
        point_cloud, traj = point_cloud.to(device, dtype=torch.float), traj.to(device, dtype=torch.float)
        start = time.time()
        traj_pred = model(point_cloud)
        print("Time:", time.time()- start)
        # print("output:", traj_pred)
        # print("output shape:", traj_pred.shape)

        # print(traj.shape)
        # visualize_sequence_traj(traj[0].cpu().numpy(), plotter=plotter, index=(0,0), extra_data=('orientnorm',))
        visualize_sequence_traj(traj_pred[0].cpu().detach().numpy(), plotter=plotter, index=(0,0), extra_data=('orientnorm',))

        # visualize_traj(traj[0].cpu().numpy(), plotter=plotter, index=(0,0), extra_data=('orientnorm',))
        # visualize_mesh_traj('./dataset/cuboids-v2/22_cube_1000_1295_801/22_cube_1000_1295_801.obj', traj[0].cpu().numpy(), extra_data=('orientnorm',))
        plotter.add_axes_at_origin()
        case = os.path.basename(os.path.dirname(args.weight_path))
        print(case)
        output_path = os.path.join(f"./output/{case}/{dirnames[0]}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plotter.show(screenshot=output_path)
        loss, loss_list = loss_handler.compute(traj_pred, traj, train=False)
        print(dirnames[0])
        print("Chamfer loss:", loss_list[0])
        tot_loss += loss.item() * B
        tot_loss_list += loss_list * B

        if metrics_handler is not None:  # Compute evaluation metrics
            metrics += B * metrics_handler.compute(traj_pred, traj)

        if save and (save_args['split'] != 'train' or i > 0):  # Save first training batch only for training set
            data = {'dirnames': dirnames, 'traj': traj.detach().cpu().numpy(), 'traj_pred': traj_pred.detach().cpu().numpy(), 'batch': 0, 'suffix': str(save_args['split'])}
            np.save(os.path.join(save_args['save_dir'], 'results_'+str(save_args['split'])+'_batch'+str(i)+'.npy'), data)
    
    
    if metrics_handler is not None:
        metrics /= data_count
    else:
        metrics = None

    return (tot_loss * 1.0 / data_count,  # total loss
            tot_loss_list * 1.0 / data_count,   # list of each loss component
            metrics)   # list of evaluation metrics 

if __name__ == '__main__':
    dataset_path = get_dataset_path(config['dataset'])

    te_dataset = PaintNetDataloader(root=dataset_path,
                                  dataset=config['dataset'],
                                  pc_points=config['pc_points'],
                                  traj_points=config['traj_points'],
                                  lambda_points=config['lambda_points'],
                                  normalization=config['normalization'],
                                  extra_data=tuple(config['extra_data']), # ('vel',)
                                  weight_orient=config['weight_orient'],
                                  split='test')
    
    te_loader = torch.utils.data.DataLoader(te_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config['backbone'], config=config)
    if config['pretrained']:
        model = init_from_pretrained(model, config=config, device=device)
    checkpoint = torch.load(args.weight_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    loss_handler = LossHandler(config['loss'], config=config)
    eval_loss, eval_loss_list, _ = test(model, te_loader, loss_handler=loss_handler)

    
    print('Tot test loss: %.5f' % (eval_loss))