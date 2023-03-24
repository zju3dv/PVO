import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"

import sys
sys.path.append('VO_Module/droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.graph_utils import build_frame_graph

# Network
from droid_net import DroidNet
from logger import Logger

# DDP training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_ddp(index, gpu, args):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=index)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()


def train(index, args):
    """ Test to make sure project transform correctly maps points """
    # torch.autograd.set_detect_anomaly(True)
    gpu = int(args.gpus.split(',')[index])

    # coordinate multiple GPUs
    setup_ddp(index, gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet(args.use_aff_bri)
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    ssim = losses.SSIM().to('cuda') if args.ssim is True else None

    if args.ckpt is not None:
        model.load_state_dict(torch.load(
            args.ckpt, map_location='cuda:'+str(gpu)))

    # fetch dataloader
    # vkitti2
    # sup 240, 808
    # semisup 216, 720(ssim in ph_cam), 240, 808(no ssim)
    # unsup 216, 720
    # tartan
    # ori 384, 512
    # semisup 344, 464
    print('loading datasets -----------------------')
    db = dataset_factory(['vkitti2'], datapath=args.datapath,
                         n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax,
                        #  crop_size=[216, 720], flow_label=args.flow_label,
                         crop_size=[200, 400], flow_label=args.flow_label,
                         aug_graph=args.aug_graph, need_inv=args.need_inv,
                         mode=args.mode)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=index)

    train_loader = DataLoader(
        db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    # args.lr = args.lr / args.gpu_num
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0

    while should_keep_training:
        for i_batch, item in enumerate(train_loader):
            optimizer.zero_grad()

            if args.flow_label:
                images, poses, disps, intrinsics, \
                    fo_flows, ba_flows, \
                    fo_masks, ba_masks = [x.to('cuda') for x in item]
            elif args.mode == 'sup':
                images, poses, disps, intrinsics,\
                    gt_masks, gt_vals = [x.to('cuda') for x in item]
            elif args.mode == 'semisup':
                images, poses, disps, intrinsics,\
                    gt_masks, gt_vals, segments = [x.to('cuda') for x in item]
            elif args.mode == 'unsup':
                images, poses, disps, intrinsics = [x.to('cuda') for x in item]

            # convert poses w2c -> c2w
            # use w2c for vkitti2, which don't have to inv
            if args.need_inv:
                Ps = SE3(poses).inv()
            else: # False
                Ps = SE3(poses)
            Gs = SE3.IdentityLike(Ps)

            if args.aug_graph: # True
                if np.random.rand() < 0.5:
                    graph = build_frame_graph(
                        poses, disps, intrinsics, num=args.edges,
                        need_inv=args.need_inv)
                else:
                    graph = OrderedDict()
                    for i in range(N):
                        graph[i] = [j for j in range(
                            N) if i != j and abs(i-j) <= 2]
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if abs(i-j) == 1]

            # fix first to camera poses
            Gs.data[:, 0] = Ps.data[:, 0].clone()
            Gs.data[:, 1:] = Ps.data[:, [1]].clone()
            disp0 = torch.ones_like(disps[:, :, 3::8, 3::8])

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()

                intrinsics0 = intrinsics / 8.0
                if args.flow_label or args.ph_loss: # True
                    if args.use_aff_bri:
                        poses_est, disps_est, residuals, full_flows, masks, aff_params = \
                            model(Gs, images, disp0, intrinsics0,
                                  graph, num_steps=args.iters, fixedp=2,
                                  ret_flow=True, downsample=args.downsample)
                    else: # False
                        poses_est, disps_est, residuals, full_flows, masks = \
                            model(Gs, images, disp0, intrinsics0,
                                  graph, num_steps=args.iters, fixedp=2,
                                  ret_flow=True, downsample=args.downsample, segments=segments)
                        aff_params = None
                else:
                    poses_est, disps_est, residuals, masks = \
                        model(Gs, images, disp0, intrinsics0,
                              graph, num_steps=args.iters, fixedp=2, ret_flow=False)

                geo_loss, cam_f_loss = 0, 0
                gt_l_loss = 0
                flo_loss = 0
                cam_ph_loss = 0
                al_loss = 0

                if args.mode == 'sup':
                    geo_loss, geo_metrics = losses.geodesic_loss(
                        Ps, poses_est, graph, do_scale=False)
                    cam_f_loss, cam_f_metrics = losses.cam_flow_loss(
                        Ps, disps, poses_est, disps_est, intrinsics, graph)
                    gt_l_loss, gt_l_metrics = losses.gt_label_loss(
                        gt_masks, gt_vals, masks, graph)
                    if args.flow_label:
                        flo_loss, flo_metrics = losses.flow_loss(
                            fo_flows, ba_flows, full_flows, graph)

                elif args.mode == 'semisup':
                    cam_ph_loss, cam_ph_metrics = losses.photo_loss_cam(
                        images, poses_est, disps_est, intrinsics, graph, args.mode, gt_masks, ssim=ssim)
                    gt_l_loss, gt_l_metrics = losses.gt_label_loss(
                        gt_masks, gt_vals, masks, graph)

                elif args.mode == 'unsup':
                    art_masks = losses.unsup_art_label(
                        poses_est, disps_est, intrinsics, full_flows, graph, downsample=args.downsample)

                    use_one = False if i_batch > args.steps * \
                        (3/4) and args.occ_ph else True
                    ph_vals = losses.unsup_occ_vals(
                        poses_est, disps_est, intrinsics, args.downsample, graph,
                        'ph_loss', use_one=use_one)
                    cam_ph_vals = losses.unsup_occ_vals(
                        poses_est, disps_est, intrinsics, False, None,
                        'cam_ph_loss', use_one=use_one)

                    cam_ph_loss, cam_ph_metrics = losses.photo_loss_cam(
                        images, poses_est, disps_est, intrinsics, graph, args.mode, cam_ph_vals, ssim=ssim)
                    al_loss, al_metric = losses.art_label_loss(
                        art_masks, masks, downsample=args.downsample)

                else:
                    raise Exception('ERROR: Unknown mode!')

                ph_loss = 0
                if args.ph_loss: # True
                    if args.mode == 'sup':
                        ph_loss, ph_metrics = losses.photo_loss(
                            images, full_flows, gt_vals, graph, args.mode, ssim=None,
                            aff_params=aff_params, downsample=args.downsample)
                    elif args.mode == 'semisup':
                        if args.occ_ph and i_batch > args.steps*(3/4):
                            ph_vals = losses.unsup_occ_vals(
                                poses_est, disps_est, intrinsics, args.downsample, graph, 'ph_loss')
                            ph_vals = losses.unsup_dy_vals(
                                ph_vals, gt_masks, graph)
                            ph_loss, ph_metrics = losses.photo_loss(
                                images, full_flows, ph_vals, graph, 'unsup', ssim=None,
                                aff_params=aff_params, downsample=args.downsample)
                        else:
                            ph_loss, ph_metrics = losses.photo_loss(
                                images, full_flows, gt_vals, graph, args.mode, ssim=None,
                                aff_params=aff_params, downsample=args.downsample)
                    else:
                        if use_one:
                            ph_loss, ph_metrics = losses.photo_loss(
                                images, full_flows, ph_vals, graph, args.mode, ssim=None,
                                aff_params=aff_params, downsample=args.downsample)
                        else:
                            ph_vals = losses.unsup_dy_vals(
                                ph_vals, art_masks, graph)
                            ph_loss, ph_metrics = losses.photo_loss(
                                images, full_flows, ph_vals, graph, args.mode, ssim=None,
                                aff_params=aff_params, downsample=args.downsample)

                res_loss, res_metrics = losses.residual_loss(residuals)

                ce_loss = 0
                if args.ce_reg: # False
                    ce_loss, ce_metric = losses.ce_reg_loss(masks)
                con_loss = 0
                if args.con_loss: # Falses
                    con_loss, con_metric = losses.consistency_loss(
                        masks, args.n_frames, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + \
                    args.w3 * cam_f_loss + args.w4 * ph_loss + \
                    args.w5 * ce_loss + args.w6 * al_loss + \
                    args.w10 * cam_ph_loss + args.w7 * con_loss + \
                    args.w8 * flo_loss + args.w9 * gt_l_loss
                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:, :, 3::8, 3::8].detach()

            metrics = {}
            metrics.update(res_metrics)
            if args.ph_loss: # True
                metrics.update(ph_metrics)

            if args.mode == 'sup':
                metrics.update(geo_metrics)
                metrics.update(cam_f_metrics)
                metrics.update(gt_l_metrics)
                if args.flow_label:
                    metrics.update(flo_metrics)

            elif args.mode == 'semisup':
                metrics.update(cam_ph_metrics)
                metrics.update(gt_l_metrics)

            elif args.mode == 'unsup':
                metrics.update(cam_ph_metrics)
                metrics.update(al_metric)

            if args.ce_reg:
                metrics.update(ce_metric)
            if args.con_loss:
                metrics.update(con_metric)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            if gpu == int(args.gpus.split(',')[0]):
                logger.push(metrics)

            if total_steps % 2000 == 0 and gpu == int(args.gpus.split(',')[0]):
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    if gpu == int(args.gpus.split(',')[0]):
        final_path = 'checkpoints/%s_final.pth' % (args.name)
        torch.save(model.state_dict(), final_path)

    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='vkitti2_dy_train',
                        help='name your experiment')
    parser.add_argument('--datapath', default='datasets/Virtual_KITTI2',
                        help="path to dataset directory")
    parser.add_argument('--need_inv', type=bool, default=False,
                        help='vkitti2 doesn\'t need inv poses, tartanair needs it')
    parser.add_argument('--gpus', type=str, default='0,1')
    parser.add_argument('--mode', type=str, default='semisup',
                        help='mode in sup,semisup,unsup')
    # 0.00025 when 4 gpus, total 0.001
    # also depend on steps
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--occ_ph', type=bool, default=False,
                        help='use occ detection in semisup ph_loss')

    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+',
                        help='lists of datasets for training')
    parser.add_argument('--flow_label', type=bool, default=False,
                        help='use gt flow data when training')
    parser.add_argument('--aug_graph', type=bool, default=True,
                        help='use droid method to get training graph')
    parser.add_argument('--use_aff_bri', type=bool, default=False,
                        help='brightness transformation parameters')
    parser.add_argument('--downsample', type=bool, default=True,
                        help='use not upsampled flows in ph_loss')
    parser.add_argument('--ssim', type=bool, default=True,
                        help='use ssim metric in ph_loss')

    parser.add_argument('--ce_reg', type=bool, default=False,
                        help='use ce regularization in masks')
    parser.add_argument('--con_loss', type=bool, default=False,
                        help='use consistency loss for dynamic masks')
    parser.add_argument('--ph_loss', type=bool, default=True )
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--clip', type=float, default=2.5)
    # 7
    parser.add_argument('--n_frames', type=int, default=6)

    # sup w
    # 10
    parser.add_argument('--w1', type=float, default=40.0, help='geo_loss')
    # 0.05
    parser.add_argument('--w3', type=float, default=0.20, help='cam_f_loss')
    # TODO: 0.01->0.05
    parser.add_argument('--w9', type=float, default=0.01, help='gt_l_loss')
    # semisup w
    # TODO: 10->5
    parser.add_argument('--w10', type=float, default=100.0, help='cam_ph_loss')
    # unsup w
    parser.add_argument('--w6', type=float, default=0.05, help='al_loss')

    parser.add_argument('--w4', type=float, default=5.0, help='ph_loss')
    parser.add_argument('--w2', type=float, default=0.01, help='res_loss')

    parser.add_argument('--w5', type=float, default=0.001, help='ce_loss')
    parser.add_argument('--w7', type=float, default=0.01, help='con_loss')
    parser.add_argument('--w8', type=float, default=0.05, help='flo_loss')

    parser.add_argument('--fmin', type=float, default=8.0)
    # 96
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    # 24
    parser.add_argument('--edges', type=int, default=20)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = len(args.gpus.split(','))
    args.gpu_num = len(args.gpus.split(','))
    print(args)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpu_num, args=(args,))
