from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform, coords_valid, coords_grid, projective_transform_unsup


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[..., :3].detach().reshape(b, -1)
    t2 = Gs.data[..., :3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:, jj] * Ps[:, ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:, jj] * Gs[i][:, ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:, None])

        # pose error
        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3, 3], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() +
                phi.norm(dim=-1).mean())

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3, 3, 1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() +
                phi.norm(dim=-1).mean() +
                0.05 * sig.norm(dim=-1).mean())

        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
    }

    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9):
    """ loss on system residuals """
    residual_loss = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()

    return residual_loss, {'residual': residual_loss.item()}


def cam_flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, gamma=0.9):
    """ optical flow loss """

    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j) == 1]

    ii, jj, kk = graph_to_edge_list(graph)
    coords0, val0 = projective_transform(Ps, disps, intrinsics, ii, jj)
    val0 = val0 * (disps[:, ii] > 0).float().unsqueeze(dim=-1)

    n = len(poses_est)
    cam_flow_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        coords1, val1 = projective_transform(
            poses_est[i], disps_est[i], intrinsics, ii, jj)

        v = (val0 * val1).squeeze(dim=-1)
        epe = v * (coords1 - coords0).norm(dim=-1)
        cam_flow_loss += w * epe.mean()

    epe = epe.reshape(-1)[v.reshape(-1) > 0.5]
    metrics = {
        'f_error': epe.mean().item(),
        '1px': (epe < 1.0).float().mean().item(),
    }

    return cam_flow_loss, metrics


def flow_loss(fo_flows, ba_flows, full_flows, graph, gamma=0.9):
    """flow loss"""

    fo_vals = fo_flows[..., 2]
    ba_vals = ba_flows[..., 2]

    n = len(full_flows)
    flow_loss = 0

    for i in range(n):
        w = gamma ** (n - i - 1)

        fo_e = ((full_flows[i][:, 0::2, ...] -
                fo_flows[..., 0:2]).norm(dim=-1) * fo_vals).mean()
        ba_e = ((full_flows[i][:, 1::2, ...] -
                ba_flows[..., 0:2]).norm(dim=-1) * ba_vals).mean()

        f_e = (fo_e + ba_e) / 2
        flow_loss += w * f_e

    metrics = {
        'pure_f_error': f_e.item()
    }

    return flow_loss, metrics


def photo_loss(images, full_flows, vals, graph, mode, gamma=0.9, ssim=None,
               mean_mask=False, aff_params=None, downsample=False):
    """direct photometric loss"""

    N, C, L = images.shape[1], images.shape[2], full_flows[0].shape[1]
    ii, jj, kk = graph_to_edge_list(graph)

    if downsample:
        images = images[..., 3::8, 3::8]
    ht, wd = images.shape[-2:]

    if mode != 'unsup':
        vals = vals[..., 3::8, 3::8, :]
        # vals = torch.ones_like(images)[:, :, 0, ..., None]
        vals_all = vals[:, ii].view(-1, ht, wd)

    images0 = images[:, ii].reshape(-1, C, ht, wd) / 255.0
    images1 = images[:, jj].reshape(-1, C, ht, wd) / 255.0
    coords0 = coords_grid(ht, wd, device=images.device)

    n = len(full_flows)
    ph_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)

        coords_flow = coords0 + full_flows[i]

        grid_x = coords_flow[..., 0]/(wd-1)
        grid_y = coords_flow[..., 1]/(ht-1)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, ht, wd, 2)
        grid = grid * 2 - 1

        # valid
        if mode == 'unsup':
            vals_all = vals[i].cuda().view(-1, ht, wd)
        val_pix = (grid.abs().max(-1)[0] <= 1).float() * vals_all
        # val_pix = (grid.abs().max(-1)[0] <= 1).float()

        warped_image0 = F.grid_sample(
            images1, grid, padding_mode="border", align_corners=True)
        # warped_image0 = F.grid_sample(
        #     images1, grid, padding_mode="zeros", align_corners=False)
        if aff_params is not None:
            aff_a = aff_params[i][..., 0].view(-1, 1, 1, 1)
            aff_b = (aff_params[i][..., 1] - 0.5).view(-1, 1, 1, 1)
            warped_image0 = warped_image0*aff_a + aff_b

        diff = compute_reprojection_loss(images0, warped_image0, ssim)
        if mean_mask:
            p_e = mean_on_mask(diff, val_pix)
        else:
            p_e = (diff*val_pix).mean()
        ph_loss += w * p_e

    metrics = {
        'ph_error': p_e.item(),
        '0.01color': mean_on_mask((diff < 0.01).float(), val_pix).item(),
    }

    return ph_loss, metrics


def photo_loss_cam(images, poses_est, disps_est, intrinsics,
                   graph, mode, masks, gamma=0.9, ssim=None):
    """supervise cam flow in ph_loss"""

    N, C = images.shape[1], images.shape[2]
    ht, wd = images.shape[-2:]

    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j) == 1]
    ii, jj, kk = graph_to_edge_list(graph)

    images0 = images[:, ii].reshape(-1, C, ht, wd)/255.0
    images1 = images[:, jj].reshape(-1, C, ht, wd)/255.0

    if mode != "unsup":
        masks_all = masks[:, ii].view(-1, ht, wd)

    n = len(poses_est)
    ph_loss = 0

    for i in range(n):
        w = gamma ** (n - i - 1)

        coords_cam, val0 = projective_transform(
            poses_est[i], disps_est[i], intrinsics, ii, jj)

        grid_x = coords_cam[..., 0]/(wd-1)
        grid_y = coords_cam[..., 1]/(ht-1)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, ht, wd, 2)
        grid = grid * 2 - 1

        val_pix = (grid.abs().max(-1)[0] <= 1).float()
        val_pix = val_pix * val0.view(-1, ht, wd)

        if mode == 'unsup':
            masks_all = masks[i].cuda().view(-1, ht, wd)
        val_pix = val_pix * masks_all

        warped_image0 = F.grid_sample(
            images1, grid, padding_mode="border", align_corners=True)

        diff = compute_reprojection_loss(images0, warped_image0, ssim)
        p_e = (diff*val_pix).mean()
        ph_loss += w * p_e

    metrics = {
        'ph_cam_error': p_e.item(),
        '0.01color_cam': mean_on_mask((diff < 0.01).float(), val_pix).item(),
    }

    return ph_loss, metrics


def unsup_occ_vals(poses_est, disps_est, intrinsics,
                   downsample, graph, loss, use_one=False):
    """occlusion and dynamic obj valid masks in unsup"""

    N = disps_est[0].shape[1]
    n = len(poses_est)

    if graph == None:
        graph = OrderedDict()
        for i in range(N):
            graph[i] = [j for j in range(N) if abs(i-j) == 1]
    ii, jj, kk = graph_to_edge_list(graph)

    intrinsics = intrinsics.cpu()
    if downsample:
        intrinsics /= 8

    val_list = []

    for i in range(n):
        disp_est = disps_est[i].detach().cpu()
        pose_est = poses_est[i].detach().cpu()

        if downsample:
            disp_est = disp_est[:, :, 3::8, 3::8]
        ht, wd = disp_est.shape[2], disp_est.shape[3]

        if use_one:
            val = torch.ones_like(disp_est[:, jj].view(-1, 1, ht, wd))
            val_list.append(val)
            continue

        coords_cam, disp0, _ = projective_transform_unsup(
            pose_est, disp_est, intrinsics, ii, jj)
        disp0 = disp0.view(-1, 1, ht, wd)
        disp1 = disp_est[:, jj].view(-1, 1, ht, wd)

        grid_x = coords_cam[..., 0]/(wd-1)
        grid_y = coords_cam[..., 1]/(ht-1)
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, ht, wd, 2)
        grid = grid * 2 - 1

        warped_disp0 = F.grid_sample(
            disp1, grid, padding_mode="border", align_corners=True)

        if loss == 'ph_loss':
            val = ((1/warped_disp0 - 1/disp0) > -0.005).float()
        else:
            val = ((1/disp0 - 1/warped_disp0).abs() <= 0.005).float()
        
        val_list.append(val)

    return val_list


def unsup_dy_vals(vals, dy_masks, graph):
    ii, jj, kk = graph_to_edge_list(graph)

    if not isinstance(dy_masks, list):
        dy_masks = dy_masks.detach().cpu()
        dy_masks = dy_masks[:, :, 3::8, 3::8]
        ht, wd = dy_masks.shape[2], dy_masks.shape[3]
        dy_val = dy_masks[:, ii].view(-1, 1, ht, wd)

    n = len(vals)
    val_list = []

    for i in range(n):
        if isinstance(dy_masks, list):
            dy_val = dy_masks[i]
            ht, wd = dy_val.shape[2], dy_val.shape[3]
            dy_val = dy_val.view(-1, 1, ht, wd)

        dy_val = 1 - dy_val
        val = torch.clamp(vals[i]+dy_val, min=0, max=1)
        val_list.append(val)

    return val_list


def compute_reprojection_loss(pred, target, ssim):
    """
    From many-depth
    Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1)

    if ssim is None:
        reprojection_loss = l1_loss
    else:
        ssim_loss = ssim(pred, target).mean(1)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def mean_on_mask(diff, val_pix):
    mask = val_pix.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        print('warning - most pixels are masked.')
        mean_value = torch.tensor(0).float().type_as(mask)
    return mean_value


def ce_reg_loss(preds, gamma=0.9):
    n = len(preds)
    entry_loss = 0

    for i in range(n):
        w = gamma ** (n - i - 1)

        e_e = -preds[i] * torch.log(preds[i] + 1e-10)
        e_e = e_e.sum(-1).mean()

        entry_loss += w*e_e

    metrics = {
        'mask_entro_error': e_e.item(),
    }

    return entry_loss, metrics


def unsup_art_label(poses_est, disps_est, intrinsics, full_flows, graph, thresh=0.5,
                    downsample=True):

    ht, wd = full_flows[0].shape[2], full_flows[0].shape[3]
    ii, jj, kk = graph_to_edge_list(graph)

    intrinsics = intrinsics.cpu()
    if downsample:
        intrinsics /= 8

    coords0 = coords_grid(ht, wd)
    n = len(full_flows)
    art_list = []

    for i in range(n):

        full_flow = full_flows[i].detach().cpu()
        pose_est = poses_est[i].detach().cpu()
        disp_est = disps_est[i].detach().cpu()
        if downsample:
            disp_est = disp_est[:, :, 3::8, 3::8]

        coords_flow = coords0 + full_flow
        coords_cam, _ = projective_transform(
            pose_est, disp_est, intrinsics, ii, jj)

        delta = (coords_flow - coords_cam).norm(dim=-1)
        art_mask = (delta <= thresh).float().unsqueeze(-1)

        art_list.append(art_mask)

    return art_list


def upsample_inter(mask):
    batch, num, ht, wd, dim = mask.shape
    mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, scale_factor=8, mode='bilinear',
                         align_corners=True, recompute_scale_factor=True)
    mask = mask.permute(0, 2, 3, 1).contiguous()
    return mask.view(batch, num, 8*ht, 8*wd, dim)


def art_label_loss(art_masks, masks, gamma=0.9, downsample=True):
    """Artificial Labels Loss"""

    # ht, wd = masks[0].shape[2], masks[0].shape[3]
    # ii, jj, kk = graph_to_edge_list(graph)
    # gt_vals_all = gt_vals[:, ii]

    n = len(masks)
    al_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)

        if downsample:
            art_mask = upsample_inter(art_masks[i]).cuda()
        else:
            art_mask = art_masks[i].cuda()

        diff = ce_func(art_mask, masks[i])
        # al_e = (diff*gt_vals_all).mean()
        al_e = diff.mean()
        al_loss += w * al_e

    metrics = {
        'art_mask_error': al_e.item(),
        'static_px_rate': art_mask.mean().item(),
        'dynamic_px_rate': (1 - art_mask).mean().item()
    }

    return al_loss, metrics


def gt_label_loss(gt_masks, gt_vals, masks, graph, gamma=0.9, mean_mask=False):
    """gt static/dynamic mask loss"""

    ii, jj, kk = graph_to_edge_list(graph)
    gt_masks_all = gt_masks[:, ii]
    gt_vals_all = gt_vals[:, ii]

    n = len(masks)
    gt_l_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)

        diff = ce_func(gt_masks_all, masks[i])
        if mean_mask:
            gt_l_e = mean_on_mask(diff, gt_vals_all)
        else:
            gt_l_e = (diff*gt_vals_all).mean()

        gt_l_loss += w * gt_l_e

    metrics = {
        'gt_mask_error': gt_l_e.item(),
        'static_px_rate': (gt_masks_all*gt_vals_all).mean().item(),
        'dynamic_px_rate': ((1-gt_masks_all)*gt_vals_all).mean().item(),
    }

    return gt_l_loss, metrics


def ce_func(labels, inputs):
    pos = labels * torch.log(inputs+1e-10)
    neg = (1-labels) * torch.log(1-inputs+1e-10)
    return -(pos + neg)


def consistency_loss(masks, n_frames, graph, gamma=0.9):
    """consistency loss to help mask be the same"""

    ii, jj, kk = graph_to_edge_list(graph)
    edge_cnt = [0]*(n_frames+1)
    for i in ii:
        edge_cnt[i+1] += 1
    for i in ii:
        edge_cnt[i+1] += edge_cnt[i]

    n = len(masks)
    con_loss = 0

    for i in range(n):
        w = gamma ** (n - i - 1)
        con_e = 0
        for j in range(n_frames):
            tmp_mask = masks[i][:, edge_cnt[j]:edge_cnt[j+1]]
            tmp_mask_m = tmp_mask.mean(1, keepdim=True)
            con_e += (tmp_mask-tmp_mask_m).mean()
        con_e /= n_frames
        con_loss += con_e*w

    metrics = {
        'con_error': con_e.item()
    }

    return con_loss, metrics
