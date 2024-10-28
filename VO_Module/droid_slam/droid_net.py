import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.modules.activation import Sigmoid

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3, 3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data


def upsample_dim_1(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


def upsample_dim_x(flow, mask):
    batch, num, ht, wd, dim = flow.shape
    flow = flow.view(batch*num, ht, wd, dim)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(flow, mask).view(batch, num, 8*ht, 8*wd, dim)


def upsample_inter(mask):
    batch, num, ht, wd, dim = mask.shape
    mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, scale_factor=8, mode='bilinear',
                         align_corners=True, recompute_scale_factor=True)
    mask = mask.permute(0, 2, 3, 1).contiguous()
    return mask.view(batch, num, 8*ht, 8*wd, dim)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask_disp = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net_less = scatter_mean(net, ix, dim=1)
        net_less = net_less.view(-1, 128, ht, wd)

        net_less = self.relu(self.conv2(net_less))

        eta = self.eta(net_less).view(batch, -1, ht, wd)
        upmask_disp = self.upmask_disp(net_less).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask_disp, None, None


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DynamicUpdateModule(nn.Module):
    def __init__(self, use_aff_bri=False):
        super(DynamicUpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2
        self.mask_num = 2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4+self.mask_num+2, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.delta_dy = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        # 动静mask: 1->static, 0->dynamic
        # this is a delta mask
        self.delta_mask = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.mask_num, 3, padding=1),
            GradientClip(),
        )

        if use_aff_bri:
            self.global_avg_pool = nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                GradientClip(),
            )
            self.param_linear = nn.Sequential(
                nn.Linear(128, 2),
                nn.Sigmoid()
            )

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def do_filter(self, lay, weight, delta_dy, delta_m, segments):
        
        segments = lay * 1e4 + segments
        encode = segments * 1e4 + ((delta_dy[...,0]  + delta_dy[...,1]) > 0.5).int()
        ky, cnt = torch.unique(encode, return_counts=True)
        cat = ky // 1e4
        sig = ky % 1e4
        cat0 = cat[sig==0].detach().cpu().numpy.tolist()
        cat1 = cat[sig==1].detach().cpu().numpy.tolist()
        cnt0 = cnt[sig==0]
        cnt1 = cnt[sig==1]
        ori_cls = torch.unique(cat) 
        for i in ori_cls:
            if i not in cat0:
                prob = 0
            elif i not in cat1: 
                prob = 1
            else:
                n0 = cnt0[cat0.index(i)]
                n1 = cnt1[cat1.index(i)]
                prob = n0 / (n0 + n1)
            if prob > 0.5:     
                fil = segments[0, i//1e4,...] == i
                lay[0, i//1e4 ,...] = lay[0, i//1e4 ,...] * (1-fil*1) 
        weight[...,0] = weight[...,0] * lay
        weight[...,1] = weight[...,1] * lay

        return weight
    
    def forward(self, net, inp, corr, flow=None,
                ii=None, jj=None, use_aff_bri=False, raw_mask=None, segments=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            # add mask channel
            flow = torch.zeros(batch, num, 4+self.mask_num+2,
                               ht, wd, device=net.device)
        if raw_mask is None:
            raw_mask = torch.zeros(
                batch, num, self.mask_num, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd) 
        net = net.view(batch*num, -1, ht, wd)               
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)
        
        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)          
        delta_dy = self.delta_dy(net).view(*output_dim)     
        weight = self.weight(net).view(*output_dim)         
        delta_m = self.delta_mask(net).view(*output_dim)    

        if use_aff_bri:
            tmp = self.global_avg_pool(net).view(batch*num, -1)
            aff_params = self.param_linear(tmp).view(batch, num, -1)

        delta = delta.permute(0, 1, 3, 4, 2).contiguous() 
        delta_dy = delta_dy.permute(0, 1, 3, 4, 2).contiguous() 
        weight = weight.permute(0, 1, 3, 4, 2).contiguous()
        delta_m = delta_m.permute(0, 1, 3, 4, 2).contiguous()

        lay = torch.as_tensor(np.arange(1, num+1).repeat(ht*wd).reshape(num,ht,wd)).unsqueeze(0) 
        # weight = self.do_filter(lay, weight, delta_dy, delta_m, segments)
        
        net = net.view(*output_dim)
        delta = torch.cat([delta, delta_dy], dim=-1)            

        if ii is not None:
            eta, upmask_disp, \
                upmask_flow, upmask_dymask = self.agg(net, ii.to(net.device))
            upmask = {
                'disp': upmask_disp,
                'flow': upmask_flow,
                'dy_mask': upmask_dymask,
            }
            if use_aff_bri:
                return net, delta, weight, eta, upmask, delta_m, aff_params
            else:
                return net, delta, weight, eta, upmask, delta_m
        else:
            return net, delta, weight, delta_m


class DroidNet(nn.Module):
    def __init__(self, use_aff_bri=False):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = DynamicUpdateModule(use_aff_bri)
        self.use_aff_bri = use_aff_bri

    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2, 1, 0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)

        net, inp = net.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp

    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12,
                fixedp=2, ret_flow=False, downsample=False, segments=None): 
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)
        dy_thresh = 0.5
        mask_num = 2

        fmaps, net, inp = self.extract_features(images) 
        segm_all = None
        if segments != None:
            segm_all = segments[:,ii]
        net, inp = net[:, ii], inp[:, ii] 
      
        corr_fn = CorrBlock(fmaps[:, ii], fmaps[:, jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:] 
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)

        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target_cam = coords1.clone()
        delta_dy = torch.zeros_like(coords1)
        raw_mask = torch.zeros_like(coords1)[..., 0:mask_num]

        Gs_list, disp_list, residual_list = [], [], []
        full_flow_list, mask_list = [], []
        if self.use_aff_bri:
            aff_params_list = []

        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target_cam = target_cam.detach()
            delta_dy = delta_dy.detach()
            raw_mask = raw_mask.detach()

            corr = corr_fn(coords1)
            resd = (target_cam - coords1)
            cam_flow = coords1 - coords0
            flow = cam_flow + delta_dy

            motion = torch.cat([cam_flow, flow, resd, raw_mask], dim=-1)
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            if self.use_aff_bri:
                net, delta, weight, eta, upmask, \
                    delta_m, aff_params = \
                    self.update(net, inp, corr, motion,
                                ii, jj, self.use_aff_bri)
            else:
                net, delta, weight, eta, upmask, delta_m = \
                    self.update(net, inp, corr, motion, ii, jj)  

            # (1:static 0:dynamic)
            raw_mask = raw_mask + delta_m
            mask = torch.sigmoid(raw_mask)
            bin_mask = (mask >= dy_thresh).float() 

            target_cam = coords1 + delta[..., 0:2]
            weight = torch.sigmoid(weight + (1-bin_mask)*10) 

            for i in range(2):
                Gs, disps = BA(target_cam, weight, eta, Gs, disps,
                               intrinsics, ii, jj, fixedp=2)
           
            coords1, valid_mask = pops.projective_transform(
                Gs, disps, intrinsics, ii, jj)
            residual = (target_cam - coords1) * valid_mask

            delta_dy = delta[..., 2:4] * (1-bin_mask)
            target_all = coords1 + delta_dy

            Gs_list.append(Gs)
            disp_list.append(upsample_dim_1(disps, upmask['disp']))
            residual_list.append(residual)
            mask_list.append(upsample_inter(mask))

            if ret_flow: 
                if downsample: 
                    full_flow_list.append(target_all - coords0)
                else:
                    full_flow_list.append(
                        upsample_inter((target_all - coords0)*8))

            if self.use_aff_bri: 
                aff_params_list.append(aff_params)

        if ret_flow: 
            if self.use_aff_bri: 
                return Gs_list, disp_list, residual_list, full_flow_list, mask_list, aff_params_list
            else: 
                return Gs_list, disp_list, residual_list, full_flow_list, mask_list
        else:
            return Gs_list, disp_list, residual_list, mask_list
