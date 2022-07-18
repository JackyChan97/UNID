import torch
from torch import nn
from utils import comfft as cf
from utils.wavelet import generate_wavelet, wv_norm, Fwv
import torch.nn.functional as F
from model.network_unet import UNetRes
import numpy as np


class UNID(nn.Module):
    def __init__(self, args):
        super(UNID, self).__init__()
        self.args = args

        self.dec2d, _ = generate_wavelet(1, dim=2)
        norm = torch.from_numpy(wv_norm(self.dec2d))
        lmd = []
        for i in range(len(args.beta)):
            lmd.append(torch.ones(len(self.dec2d)) * args.beta[i] / norm)

        self.net = nn.ModuleList()
        self.net = self.net.append(Db_Inv(lmd=lmd[0]))
        for i in range(args.layers):
            self.net = self.net.append(UNetRes(in_nc=i + 1))
            self.net = self.net.append(Db_Inv(lmd=lmd[i + 1]))
        self.FDN_boundary = True

    def forward(self, y, Fker, KS, ker_resize):

        B, C, H, W = y.shape

        xhat = [None] * (self.args.layers + 1)
        z = [None] * (self.args.layers)

        PH = 0 if H % 8 == 0 else 8 - H % 8
        PW = 0 if W % 8 == 0 else 8 - W % 8

        xhat[0] = self.net[0](y, Fker, None, None)

        net_input = torch.cat([xhat[0]], dim=1)
        net_input = F.pad(net_input, (0, PW, 0, PH))

        output = self.net[1](net_input)
        z[0] = output[:, :, :H, :W]

        for i in range(self.args.layers - 1):
            if self.FDN_boundary:
                y_adj = boundary_adj(y, xhat[i], Fker, KS)
                xhat[i + 1] = self.net[2 * i + 2](y_adj, Fker, z[i])
            else:
                xhat[i + 1] = self.net[2 * i + 2](y, Fker, z[i])
            input = torch.cat([(xhat[j]) for j in range(0, i + 2)], dim=1)

            net_input = torch.cat([input], dim=1)
            net_input = F.pad(net_input, (0, PW, 0, PH))

            output = self.net[2 * i + 3](net_input)
            z[i + 1] = output[:, :, :H, :W]

        i = self.args.layers - 1
        if self.FDN_boundary:
            y_adj = boundary_adj(y, xhat[i], Fker, KS)
            xhat[i + 1] = self.net[2 * i + 2](y_adj, Fker, z[i])  # ,u[i]
        else:
            xhat[i + 1] = self.net[2 * i + 2](y, Fker, z[i])
        return xhat  # ,u




class Db_Inv(nn.Module):
    def __init__(self, lmd):
        super(Db_Inv, self).__init__()
        self.dec2d, _ = generate_wavelet(frame=1)
        self.chn_num = len(self.dec2d)
        self.lmd = lmd.view(self.chn_num, 1, 1, 1).cuda()

    def forward(self, y, Fker, z=None, u=None):
        if z is None: z = torch.zeros_like(y)
        if u is None: u = torch.zeros_like(y)

        im_num = y.shape[0]
        xhat = torch.zeros_like(y)

        # for i in range(im_num):
        shape = y[0, 0,].size()[-2:]
        Fw = Fwv(self.dec2d, shape=shape).unsqueeze(0).repeat(xhat.shape[0], 1, 1, 1, 1).cuda()

        Fker_conj = cf.conj(Fker).cuda()
        Fw_conj = cf.conj(Fw).cuda()

        Fy = torch.rfft(y + u, signal_ndim=2, onesided=False)  # add w to incorporate the prior approximation of noise
        Fz = torch.rfft(z, signal_ndim=2, onesided=False).cuda()

        Fx_num = cf.mul(Fker_conj, Fy) + torch.sum(self.lmd * cf.mul(Fw_conj, cf.mul(Fw, Fz)), dim=1, keepdim=True)
        Fx_den = cf.abs_square(Fker, keepdim=True) + torch.sum(self.lmd * cf.mul(Fw_conj, Fw), dim=1, keepdim=True)
        Fx = cf.div(Fx_num, Fx_den)
        xhat = torch.irfft(Fx, signal_ndim=2, onesided=False)
        return xhat




def boundary_adj(bl, x, Fker, kerSize):
    bl = bl.squeeze(1)
    x = x.squeeze(1)
    Fker = Fker.squeeze(1)

    mask_int = torch.zeros_like(bl).cuda()
    for j in range(mask_int.shape[0]):
        padSize = np.floor_divide(kerSize[j], 2)
        mask_int[j, padSize[0]:-padSize[0], padSize[1]:-padSize[1]] = 1

    kx_fft = cf.mul(torch.rfft(x, signal_ndim=2, onesided=False), Fker)
    kx = torch.irfft(kx_fft, signal_ndim=2, onesided=False)
    kx_out = (1 - mask_int) * kx
    y_inner = mask_int * bl
    bl_adj = kx_out + y_inner
    bl_adj = bl_adj.unsqueeze(1)

    return bl_adj
