import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.dy_thresh = 0.5
        self.mask_num = 2

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.segm = None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target_cam = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.raw_mask = torch.zeros(
            [1, 0, ht, wd, self.mask_num], device=device, dtype=torch.float)
        self.delta_dy = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.full_flow = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.ii_bad = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_bad = torch.as_tensor([], dtype=torch.long, device=device)

        self.target_cam_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.raw_mask_inac = torch.zeros(
            [1, 0, ht, wd, self.mask_num], device=device, dtype=torch.float)
        self.delta_dy_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.full_flow_inac = torch.zeros(
            [1, 0, ht, wd, 2], device=device, dtype=torch.float)
        

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0, 2, 3, 4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def filter_edges(self):
        """ remove bad edges """
        conf = torch.mean(self.weight, dim=[0, 2, 3, 4])
        mask = (torch.abs(self.ii-self.jj) > 2) & (conf < 0.001)

        self.ii_bad = torch.cat([self.ii_bad, self.ii[mask]])
        self.jj_bad = torch.cat([self.jj_bad, self.jj[mask]])
        self.rm_factors(mask, store=False)

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)

        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:

            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            fmap1 = self.video.fmaps[ii].to(self.device).unsqueeze(0)
            fmap2 = self.video.fmaps[jj].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj)
            weight = torch.zeros_like(target)
            raw_mask = torch.zeros_like(target)[..., 0:self.mask_num]
            delta_dy = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target_cam = torch.cat([self.target_cam, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)
        self.raw_mask = torch.cat([self.raw_mask, raw_mask], 1)
        self.delta_dy = torch.cat([self.delta_dy, delta_dy], 1)

        # segmentations
        segm = self.video.segms[ii].to(self.device).unsqueeze(0)
        self.segm = segm if self.segm is None else torch.cat([self.segm, segm], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_cam_inac = torch.cat(
                [self.target_cam_inac, self.target_cam[:, mask]], 1)
            self.weight_inac = torch.cat(
                [self.weight_inac, self.weight[:, mask]], 1)
            self.raw_mask_inac = torch.cat(
                [self.raw_mask_inac, self.raw_mask[:, mask]], 1)
            self.delta_dy_inac = torch.cat(
                [self.delta_dy_inac, self.delta_dy[:, mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]

        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:, ~mask]

        if self.inp is not None:
            self.inp = self.inp[:, ~mask]
        
        if self.segm is not None:
            self.segm = self.segm[:, ~mask]

        self.target_cam = self.target_cam[:, ~mask]
        self.weight = self.weight[:, ~mask]
        self.raw_mask = self.raw_mask[:, ~mask]
        self.delta_dy = self.delta_dy[:, ~mask]

    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        with self.video.get_lock():
            self.video.poses[ix] = self.video.poses[ix+1]
            self.video.disps[ix] = self.video.disps[ix+1]
            self.video.intrinsics[ix] = self.video.intrinsics[ix+1]

            self.video.nets[ix] = self.video.nets[ix+1]
            self.video.inps[ix] = self.video.inps[ix+1]
            self.video.fmaps[ix] = self.video.fmaps[ix+1]
            if self.video.segm_filter:
                self.video.segms[ix] = self.video.segms[ix+1]
            

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1

        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        self.rm_factors(m, store=False)

    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, motion_only=False):
        """ run update operator on factor graph """

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, mask = self.video.reproject(self.ii, self.jj)
            motn = torch.cat(
                [self.target_cam - self.coords0, self.target_cam - self.coords0 + self.delta_dy,
                 self.target_cam - coords1, self.raw_mask*torch.ones_like(coords1)[..., 0:self.mask_num]], dim=-1)
            motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        # correlation features
        corr = self.corr(coords1)

        self.net, delta, weight, damping, upmask, delta_m = \
            self.update_op(self.net, self.inp, corr,
                           motn, self.ii, self.jj, False)

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target_cam = coords1 + delta[..., 0:2].to(dtype=torch.float)
            self.weight = weight.to(dtype=torch.float) 
            self.raw_mask = self.raw_mask + delta_m
            bin_mask = ( torch.sigmoid(self.raw_mask) >= self.dy_thresh) 
            
            # filter with segm
            if self.video.segm_filter:
                ht = weight.shape[2] 
                wd = weight.shape[3]
                lay = np.arange(1, weight.shape[1]+1).repeat(ht*wd).reshape(1,-1,ht,wd) 
                segments = (lay * 1e6 + self.segm.squeeze(2).detach().cpu().numpy()).astype(np.int32) 
                dynamic_m = ( (bin_mask[...,0] == 0).__or__(bin_mask[...,1] == 0) ).detach().cpu().numpy() 
                ori_ky, ori_cnt = np.unique(segments, return_counts=True)
                ori_dic = dict(zip(ori_ky, ori_cnt))

                dy_fields = segments * dynamic_m
                dy_ky, dy_cnt = np.unique(dy_fields, return_counts=True)
                for (label, dy_n) in zip(dy_ky, dy_cnt):
                    if label % 1e6 == 0:
                        continue
                    if (dy_n / ori_dic[label]) > self.video.thresh: # 0.8
                        dim = int(label // 1e6) -1
                        fil = segments[0,dim,...] == label
                        lay[0,dim,...] =  lay[0,dim,...] * (1-fil*1)
                lay = torch.as_tensor(lay).to(self.device)
                bin_mask[...,0] = bin_mask[...,0] * (lay > 0)
                bin_mask[...,1] = bin_mask[...,1] * (lay > 0)
            
            bin_mask = bin_mask.float()
            self.delta_dy = delta[..., 2:4] * (1-bin_mask)
            self.weight = torch.sigmoid(self.weight + (1-bin_mask)*10)

            ht, wd = self.coords0.shape[0:2]
            self.damping[torch.unique(self.ii)] = damping

            if use_inactive:
                m = (self.ii_inac >= t0 - 3) & (self.jj_inac >= t0 - 3)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target_cam = torch.cat(
                    [self.target_cam_inac[:, m], self.target_cam], 1)
                weight = torch.cat([self.weight_inac[:, m], self.weight], 1)
            else:
                ii, jj, target_cam, weight = self.ii, self.jj, self.target_cam, self.weight

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP

            target_cam = target_cam.view(-1, ht, wd, 2).permute(0,
                                                                3, 1, 2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,
                                                        3, 1, 2).contiguous()
            # dense bundle adjustment
            self.video.ba(target_cam, weight, damping, ii, jj, t0, t1,
                          itrs=itrs, lm=1e-4, ep=0.1, motion_only=motion_only)
            target_all = coords1 + self.delta_dy  
            self.full_flow = target_all - self.coords0

        self.age += 1

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, use_inactive=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        # alternate corr implementation
        t = self.video.counter.value
        corr_op = AltCorrBlock(self.video.fmaps[None, :t])

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat(
                    [self.target_cam - self.coords0, self.target_cam - self.coords0 + self.delta_dy,
                     self.target_cam - coords1, self.raw_mask*torch.ones_like(coords1)[..., 0:self.mask_num]], dim=-1)
                motn = motn.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            s = 8
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:, v], iis, jjs)

                with torch.cuda.amp.autocast(enabled=True):
                    net, delta, weight, damping, _, delta_m = self.update_op(
                        self.net[:, v], self.video.inps[None, iis], corr1, motn[:, v], iis, jjs, False)

                self.net[:, v] = net
                self.target_cam[:, v] = coords1[:, v] + delta[..., 0:2].float()
                self.weight[:, v] = weight.float()
                self.damping[torch.unique(iis)] = damping
                self.raw_mask[:, v] = self.raw_mask[:, v] + delta_m.float()
                bin_mask = (torch.sigmoid(
                    self.raw_mask[:, v]) >= self.dy_thresh).float()

                self.delta_dy[:, v] = delta[..., 2:4] * (1-bin_mask)
                self.weight[:, v] = torch.sigmoid(
                    self.weight[:, v] + (1-bin_mask)*10)

            damping = self.damping[torch.unique(self.ii)].contiguous() + EP
            target_cam = self.target_cam.view(-1, ht, wd,
                                              2).permute(0, 3, 1, 2).contiguous()
            weight = self.weight.view(-1, ht, wd,
                                      2).permute(0, 3, 1, 2).contiguous()
            # dense bundle adjustment
            self.video.ba(target_cam, weight, damping, self.ii, self.jj, 1, t,
                          itrs=itrs, lm=1e-5, ep=1e-2, motion_only=False)

            self.video.dirty[:t] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0, t1), torch.arange(t0, t1))
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        keep = ((ii - jj).abs() > 0) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx)
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_bad, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_bad, self.jj_inac], 0)
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            if abs(i - j) <= 2:
                continue
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        es = []
        for i in range(t0, t):
            for j in range(i+1, min(i+rad+1, t)):
                es.append((i, j))
                es.append((j, i))

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            i = ii[k]
            j = jj[k]

            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

def upsample_inter(mask):
    batch, num, ht, wd, dim = mask.shape
    mask = mask.permute(0, 1, 4, 2, 3).contiguous()
    mask = mask.view(batch*num, dim, ht, wd)
    mask = F.interpolate(mask, scale_factor=8, mode='bilinear',
                         align_corners=True, recompute_scale_factor=True)
    mask = mask.permute(0, 2, 3, 1).contiguous()
    return mask.view(batch, num, 8*ht, 8*wd, dim)
