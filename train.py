"""Training a model for predicting spray painting
trajectory, given object point-cloud in input
    
    Examples:
        - Quick: python train.py --epochs 200 --pc_points 20 --traj_points 20 -bs 4 --loss chamfer --seed 3 --debug
        - Complete (cuboids): python train.py --epochs 1250 --pc_points 5120 --traj_points 2000 -bs 32 --loss chamfer rich_attraction_chamfer --seed 3 --backbone pointnet2 --pretrained --lambda_points 4 --extra_data orientnorm --weight_orient 0.25 --weight_rich_attraction_chamfer 0.5
        - Reproduce paper results: 
            - python train.py --config cuboids_stable_v1.json --seed 42
            - python train.py --config cuboids_lambda1_v1.json --seed 42
            - python train.py --config windows_stable_v1.json --seed 42
            - python train.py --config shelves_stable_v1.json --seed 42
            - python train.py --config containers_stable_v1.json --seed 42
"""
import pdb
import os
import sys
import argparse
from pprint import pprint
import time
import socket
import shutil
from tqdm import tqdm
import numpy as np
import torch
import wandb

from paintnet_utils import *
from paintnet_loader import PaintNetDataloader
from model_utils import get_model, init_from_pretrained
from loss_handler import LossHandler
from metrics_handler import MetricsHandler

os.environ["PAINTNET_ROOT"] = "./dataset"

def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--eval_freq',      default=50, type=int, help='Evaluate model on test set and save it every <eval_freq> epochs')
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

def main():
    random_str = get_random_string(5)
    set_seed(args.seed)

    run_name = random_str+('_'+args.name if args.name is not None else '')
    save_dir = os.path.join((args.output_dir if not args.debug else 'debug_runs'), run_name)
    create_dirs(save_dir)
    save_config(config, save_dir)

    print('\n ===== RUN NAME:', run_name, f' ({save_dir}) ===== \n')
    pprint(vars(args))

    dataset_path = get_dataset_path(args.dataset)

    wandb.init(config=config,
               name=run_name,
               group=args.group,
               save_code=True,
               notes=args.notes,
               mode=('online' if not args.debug else 'disabled'))
    
    wandb.config.path = save_dir
    wandb.config.hostname = socket.gethostname()

    tr_dataset = PaintNetDataloader(root=dataset_path,
                                  dataset=config['dataset'],
                                  pc_points=config['pc_points'],
                                  traj_points=config['traj_points'],
                                  lambda_points=config['lambda_points'],
                                  overlapping=config['overlapping'],
                                  normalization=config['normalization'],
                                  extra_data=tuple(config['extra_data']), # ('vel',)
                                  weight_orient=config['weight_orient'],
                                  split='train',
                                  overfitting=(None if args.overfitting is False else args.seed),
                                  augmentations=config['augmentations'])

    te_dataset = PaintNetDataloader(root=dataset_path,
                                  dataset=config['dataset'],
                                  pc_points=config['pc_points'],
                                  traj_points=config['traj_points'],
                                  lambda_points=config['lambda_points'],
                                  normalization=config['normalization'],
                                  extra_data=tuple(config['extra_data']), # ('vel',)
                                  weight_orient=config['weight_orient'],
                                  split='test')

    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=config['batch_size'],
                                            shuffle=(True if args.overfitting is False else False),
                                            num_workers=args.workers,
                                            drop_last=True)

    te_loader = torch.utils.data.DataLoader(te_dataset,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Run on ", device)
    model = get_model(config['backbone'], config=config)
    if config['pretrained']:
        model = init_from_pretrained(model, config=config, device=device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=config['steplr'], gamma=0.5)

    loss_handler = LossHandler(config['loss'], config=config)

    single_sample = None
    best_eval_loss = sys.float_info.max
    print("Start train")
    eval_loss = 0.0
    best_epoch = 0
    for epoch in tqdm(range(args.epochs), desc="Epoch:"):
        start_ep_time = time.time()
        tot_loss = 0.0

        tot_loss_list = np.zeros(len(loss_handler.loss))
        data_count = 0
        epoch_count = 0
        model.train()
        for i, data in enumerate(tr_loader):
            model.zero_grad()

            point_cloud, traj, dirname = data

            if args.overfitting and single_sample is None:
                single_sample = dirname
                # assert args.batch_size == 1, '--overfitting needs a batch_size=1'

            B, N, dim = point_cloud.size()
            data_count += B
            point_cloud = point_cloud.permute(0, 2, 1) # B, 3, pc_points
            point_cloud, traj = point_cloud.to(device, dtype=torch.float), traj.to(device, dtype=torch.float)
            
            # for b in range(B):
            #     visualize_mesh_traj(os.path.join(dataset_path, dirname[b], dirname[b]+'_norm.obj'), traj[b], lambda_points=config['lambda_points'], check_padding=True)
            # pdb.set_trace()

            traj_pred = model(point_cloud)

            loss, loss_list = loss_handler.compute(traj_pred, traj)

            loss.backward()
            opt.step()

            tot_loss += loss.item() * B
            tot_loss_list += loss_list * B

        sched.step()
        print({"TOT_epoch_train_loss": (tot_loss * 1.0 / data_count), "epoch": (epoch+1)})
        wandb.log({"TOT_epoch_train_loss": (tot_loss * 1.0 / data_count), "epoch": (epoch+1)})
        tot_loss_list = tot_loss_list * 1.0 / data_count
        loss_handler.log_on_wandb(tot_loss_list, epoch, wandb, suffix='_train_loss')
        print('[%d/%d] Epoch time: %s' % (
            epoch+1, args.epochs, time.strftime("%M:%S", time.gmtime(time.time() - start_ep_time))), '| Epoch train loss: %.5f' % (tot_loss * 1.0 / data_count))

        if (epoch+1) % args.eval_freq == 0:
            torch.save(
                {'epoch': epoch + 1,
                 'epoch_train_loss': tot_loss * 1.0 / data_count,
                 'model': model.state_dict(),
                 'optimizer': opt.state_dict(),
                 'scheduler': sched.state_dict(),
                },
                os.path.join(save_dir, 'last_checkpoint.pth')
            )

            eval_loss, eval_loss_list, _ = test(model, te_loader, loss_handler=loss_handler)
            print('Tot test loss: %.5f' % (eval_loss))

            wandb.log({"TOT_test_loss": eval_loss, "epoch": (epoch+1)})
            loss_handler.log_on_wandb(eval_loss_list, epoch, wandb, suffix='_test_loss')

            is_best = eval_loss < best_eval_loss
            best_eval_loss = min(eval_loss, best_eval_loss)
            if is_best:
                best_epoch = epoch+1
                shutil.copyfile(
                    src=os.path.join(save_dir, 'last_checkpoint.pth'),
                    dst=os.path.join(save_dir, 'best_model.pth'))


    wandb.run.summary["best_epoch"] = best_epoch
    if args.overfitting:
        wandb.run.summary["single_sample"] = single_sample
        print('Overfitting on:', single_sample)

    print('\n\n============== TRAINING FINISHED ==============')
    print('Best epoch:', best_epoch)
    print('Best test loss:', best_eval_loss)
    print('Last test loss:', eval_loss)


    """
        Test best model and render results
    """
    if config['eval_ckpt'] == 'best':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))
    elif config['eval_ckpt'] == 'last':
        eval_checkpoint = torch.load(os.path.join(save_dir, 'last_checkpoint.pth'), map_location=torch.device(device))
    else:  # default
        print('\n\nWARNING! Falling back to best_model.pth as eval_ckpt has invalid name.\n\n')
        eval_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=torch.device(device))

    model = get_model(config['backbone'], config=config)
    model.load_state_dict(eval_checkpoint['model'], strict=True)
    model.to(device)
    model.eval()

    tr_dataset = PaintNetDataloader(root=dataset_path,
                                  dataset=config['dataset'],
                                  pc_points=config['pc_points'],
                                  traj_points=config['traj_points'],
                                  lambda_points=config['lambda_points'],
                                  normalization=config['normalization'],
                                  extra_data=tuple(config['extra_data']),
                                  weight_orient=config['weight_orient'],
                                  split='train',
                                  overfitting=(None if args.overfitting is False else args.seed),
                                  augmentations=None)  # Data augm is False for testing

    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=config['batch_size'],
                                            shuffle=False,  # Shuffle False for testing
                                            num_workers=args.workers,
                                            drop_last=True)

    metrics_handler = MetricsHandler(config=config, metrics=config['eval_metrics'])
    save_args = {'save_dir': save_dir}
    eval_loss, eval_loss_list, _ = test(model, tr_loader, loss_handler=loss_handler, metrics_handler=None, save=(not args.no_save), **{'split': 'train', **save_args})
    if not args.overfitting:
        eval_loss, eval_loss_list, eval_metrics = test(model, te_loader, loss_handler=loss_handler, metrics_handler=metrics_handler, save=(not args.no_save), **{'split': 'test',  **save_args})
        print(f'Eval metrics on test set:')
        metrics_handler.pprint(eval_metrics)
        metrics_handler.log_on_wandb(eval_metrics, wandb, suffix='_TEST_EVAL_METRIC')

    print('Results saved successfully')
    wandb.finish()


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
        B, N, dim = point_cloud.size()
        data_count += B
        point_cloud = point_cloud.permute(0, 2, 1) # B, 3, pc_points
        point_cloud, traj = point_cloud.to(device, dtype=torch.float), traj.to(device, dtype=torch.float)

        traj_pred = model(point_cloud)

        loss, loss_list = loss_handler.compute(traj_pred, traj, train=False)

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
    main()