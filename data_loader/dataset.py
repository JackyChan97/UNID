import os
import re
from glob import glob
from matplotlib.image import imread
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
import cv2

from utils.imtools import for_fft, center_of_mass
from utils import comfft as cf
from utils.imtools import image_register, get_kernel_size

'''For the test of Lai_Real Dataset'''


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class Test_Lai_NoiseKernel(Dataset):
    def __init__(self, test_sp_dir, test_bl_dir, test_ker_dir, taper='same'):

        self.bl_dir = test_bl_dir
        self.ker_dir = test_ker_dir
        self.sp_dir = test_sp_dir
        self.sp_file = sorted(glob(self.sp_dir + '*.png'))
        self.ker_file = sorted(glob(self.ker_dir + '*.png'))
        self.taper = taper
        self.ker_num = 4

    def __len__(self):
        img_num = len(self.ker_file)
        return img_num

    def __getitem__(self, item):
        ##reverse
        item = self.__len__() - item - 1
        '''load test item one by one'''
        ker_name = self.ker_file[item]
        sp_name = re.findall(r'psf_([\s\S]*)_kernel', ker_name)[0]
        tr_ker_name = re.findall(r'_kernel_([\s\S]*)_1', ker_name)[0]

        sp = imread(self.sp_dir + sp_name + '.png')[:, :, :3]

        bl = imread(self.bl_dir + sp_name + '_kernel_' + tr_ker_name + '.png')

        ker = rgb2gray(imread(ker_name))
        ker = ker / np.sum(ker)
        ker = np.rot90(ker, 2)

        tr_ker = rgb2gray(imread('./data/Lai_NK/kernels/kernel_' + tr_ker_name + '.png'))
        tr_ker = tr_ker / np.sum(tr_ker)

        # TODO add center to avoid circular problem
        ker = self.center_ker(ker, tr_ker)

        tr_ker_pad = np.full([110, 110], np.nan)
        tr_ker_pad[:tr_ker.shape[0], :tr_ker.shape[1]] = tr_ker

        if self.taper == 'valid':
            from utils.imtools import pad_for_kernel, edgetaper
            bl_pad = np.zeros_like(sp)
            for chn in range(3):
                bl_pad[:, :, chn] = edgetaper(pad_for_kernel(bl[:, :, chn], tr_ker, 'edge'), ker).astype(np.float32)
            bl = bl_pad

        ker_mat = torch.FloatTensor(for_fft(tr_ker, shape=np.shape(sp[:, :, 0])))
        tr_Fker = cf.fft(ker_mat).unsqueeze(0)

        ker_pad = np.full([110, 110], np.nan)
        ker_pad[:ker.shape[0], :ker.shape[1]] = ker

        ker_mat = torch.FloatTensor(for_fft(ker, shape=np.shape(sp[:, :, 0])))
        Fker = cf.fft(ker_mat).unsqueeze(0)

        hy = (ker.shape[0] - 1) // 2
        hx = (ker.shape[0] - 1) - hy
        wy = (ker.shape[1] - 1) // 2
        wx = (ker.shape[1] - 1) - wy
        padding = np.array((hx, hy, wx, wy), dtype=np.int64)

        sp = torch.from_numpy(sp).unsqueeze(0)
        bl = torch.from_numpy(bl).unsqueeze(0)

        kh, kw = get_kernel_size(ker, ker.shape, 0.95)
        ks = max(kh, kw)
        if ks < ker.shape[0]:
            ker_resize_pad = ker[(ker.shape[0] - ks) // 2:-((ker.shape[0] - ks) // 2),
                             (ker.shape[1] - ks) // 2:-((ker.shape[1] - ks) // 2)]
        else:
            ker_resize_pad = ker

        ker_resize = cv2.resize(ker_resize_pad, (15, 15))
        ker_resize = ker_resize / np.sum(ker_resize)
        # ker_resize = ker_resize * (max(ker.shape) / 15)

        KS = torch.FloatTensor([kh, kw])

        dic = {'bl': bl, 'sp': sp, 'Fker': Fker, 'padding': padding.copy(), 'ker': ker_pad.copy(),
               'tr_ker': tr_ker_pad.copy(), 'tr_Fker': tr_Fker, 'name': sp_name + '_' + tr_ker_name,
               'ker_resize': ker_resize, 'KS': KS,'idx':item}

        return dic

    @staticmethod
    def center_ker(ker, tr_ker=None):
        from scipy.ndimage.measurements import center_of_mass
        if tr_ker is None:
            tkrh = 0
            tkrv = 0
        else:
            ctkh, ctkv = center_of_mass(tr_ker)
            tkh, tkv = np.shape(tr_ker)
            tkh2 = tkh / 2
            tkv2 = tkv / 2
            tkrh = tkh2 - ctkh - 1
            tkrv = tkv2 - ctkv - 1

        ckh, ckv = center_of_mass(ker)
        kh, kv = np.shape(ker)
        kh2 = kh / 2
        kv2 = kv / 2

        rh = int(round((kh2 - ckh - 1) - tkrh))
        rv = int(round((kv2 - ckv - 1) - tkrv))

        ker_roll = np.roll(ker, (rh, rv), (0, 1))
        return ker_roll





class Test_NoiseKernel(Dataset):
    def __init__(self, test_sp_dir, test_bl_dir, test_ker_dir, tr_ker_dir, taper='same'):
        self.bl_dir = test_bl_dir
        self.ker_dir = test_ker_dir
        self.sp_dir = test_sp_dir
        self.sp_file = sorted(glob(self.sp_dir + '*.png'))
        self.taper = taper
        self.ker_num = 8

        ker_mat = loadmat(tr_ker_dir)
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]

    def __len__(self):
        img_num = len(self.sp_file) * self.ker_num
        return img_num

    def __getitem__(self, item):
        '''load test item one by one'''
        i = item // self.ker_num
        j = item % self.ker_num

        sp = imread(os.path.join(self.sp_dir, 'im_%d.png' % (i + 1)))
        bl_path = glob(os.path.join(self.bl_dir, 'im_%d_ker_%d*.png' % (i + 1, j + 1)))
        bl = imread(bl_path[0])

        ker_name = glob(os.path.join(self.ker_dir, 'k_%d_im_%d_*' % (j + 1, i + 1)))
        ker = imread(ker_name[0])
        ker = ker / np.sum(ker)

        tr_ker = self.get_ker(j)
        tr_ker_pad = np.full([51, 51], np.nan)
        tr_ker_pad[:tr_ker.shape[0], :tr_ker.shape[1]] = tr_ker

        tr_ker_mat = torch.FloatTensor(for_fft(tr_ker, shape=np.shape(sp)))
        tr_Fker = cf.fft(tr_ker_mat).unsqueeze(0)

        if self.taper == 'valid':
            from utils.imtools import pad_for_kernel, edgetaper
            bl = edgetaper(pad_for_kernel(bl, tr_ker, 'edge'), ker)
            bl = bl.astype(np.float32)

        ker_pad = np.full([51, 51], np.nan)
        ker_pad[:ker.shape[0], :ker.shape[1]] = ker

        ker_mat = torch.FloatTensor(for_fft(ker, shape=np.shape(sp)))
        ker_mat_padding = torch.FloatTensor(for_fft(ker, shape=(256, 256)))
        Fker = cf.fft(ker_mat).unsqueeze(0)
        Fker_padding = cf.fft(ker_mat_padding).unsqueeze(0)

        # ker_align = image_register(bl, sp, ker, Fker.squeeze(0))

        center_y, center_x = center_of_mass(ker)
        center_y_int = int(np.round(center_y))
        center_x_int = int(np.round(center_x))
        kh, kw = ker.shape
        ker_align = np.roll(ker, (kh // 2 - center_y_int, kw // 2 - center_x_int), (-2, -1))

        kh, kw = get_kernel_size(ker_align, ker_align.shape, 0.95)
        ks = max(kh, kw)
        if ks < ker.shape[0]:
            ker_resize_pad = ker_align[(ker_align.shape[0] - ks) // 2:-((ker_align.shape[0] - ks) // 2),
                             (ker_align.shape[1] - ks) // 2:-((ker_align.shape[1] - ks) // 2)]
        else:
            ker_resize_pad = ker

        hy = (ker.shape[0] - 1) // 2
        hx = (ker.shape[0] - 1) - hy
        wy = (ker.shape[1] - 1) // 2
        wx = (ker.shape[1] - 1) - wy
        padding = np.array((hx, hy, wx, wy), dtype=np.int64)

        sp = torch.from_numpy(sp).unsqueeze(0)
        bl = torch.from_numpy(bl).unsqueeze(0)

        ker_resize = cv2.resize(ker_resize_pad, (15, 15))
        ker_resize = ker_resize / np.sum(ker_resize)
        # ker_resize = ker_resize * (max(ker.shape) / 15)

        KS = torch.FloatTensor([kh, kw])

        dic = {'bl': bl, 'sp': sp, 'Fker': Fker, 'padding': padding.copy(), 'ker': ker_pad.copy(),
               'tr_ker': tr_ker_pad.copy(),
               'tr_Fker': tr_Fker, 'name': 'im_%d_ker_%d' % (i + 1, j + 1), 'ker_resize': ker_resize, 'KS': KS,
               'Fker_padding': Fker_padding,'idx':item}

        return dic


class Test_Dataset(Dataset):
    def __init__(self, test_sp_dir, test_bl_dir, test_ker_dir):

        self.bl_dir = test_bl_dir
        self.sp_dir = test_sp_dir
        self.ker_dir = test_ker_dir
        self.sp_file = sorted(glob(self.sp_dir + '*.png'))

        ker_mat = loadmat(test_ker_dir)
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]
        self.ker_num = len(ker_mat[0])

    def __len__(self):
        img_num = len(self.sp_file) * self.ker_num
        return img_num

    def __getitem__(self, item):
        i = item // self.ker_num
        j = item % self.ker_num
        sp = imread(os.path.join(self.sp_dir, 'im_%d.png' % (i + 1)))
        bl = imread(os.path.join(self.bl_dir, 'im_%d_ker_%d.png' % (i + 1, j + 1)))

        ker = self.get_ker(j)
        ker_pad = np.full([51, 51], np.nan)
        ker_pad[:ker.shape[0], :ker.shape[1]] = ker

        ker_mat = torch.FloatTensor(for_fft(ker, shape=np.shape(sp)))
        Fker = cf.fft(ker_mat)
        Fker = Fker.unsqueeze(0)

        sp = torch.from_numpy(sp).unsqueeze(0)
        bl = torch.from_numpy(bl).unsqueeze(0)

        k_center = ker

        if k_center.shape[0] != k_center.shape[1]:
            ker_pad = np.zeros([np.max(k_center.shape), np.max(k_center.shape)])
            ker_pad[:k_center.shape[0], :k_center.shape[1]] = k_center
            k_center = np.roll(ker_pad, [(ker_pad.shape[0] - k_center.shape[0]) // 2,
                                         (ker_pad.shape[1] - k_center.shape[1]) // 2],
                               [0, 1])
            ker_resize = cv2.resize(k_center, (15, 15))
        else:
            ker_resize = cv2.resize(k_center, (15, 15))
        ker_resize = ker_resize / np.sum(ker_resize)

        dic = {'bl': bl, 'sp': sp, 'Fker': Fker, 'ker': ker_pad.copy(), 'ker_resize': ker_resize.astype(np.float32)}
        return dic


from utils.noise_estimation import noise_estimate


class Test_Dataset_possion(Dataset):
    def __init__(self, test_sp_dir, test_bl_dir, test_ker_dir):

        self.bl_dir = test_bl_dir
        self.sp_dir = test_sp_dir
        self.ker_dir = test_ker_dir
        self.sp_file = sorted(glob(self.sp_dir + '*.png'))

        ker_mat = loadmat(test_ker_dir)
        ker_mat = ker_mat['kernels']
        self.get_ker = lambda idx: ker_mat[0, idx]
        self.ker_num = len(ker_mat[0])

    def __len__(self):
        img_num = len(self.sp_file) * self.ker_num
        return img_num

    def __getitem__(self, item):
        i = item // self.ker_num
        j = item % self.ker_num
        sp = imread(os.path.join(self.sp_dir, 'im_%d.png' % (i + 1)))
        bl = imread(os.path.join(self.bl_dir, 'im_%d_ker_%d.png' % (i + 1, j + 1)))

        std = noise_estimate(bl, 8)

        ker = self.get_ker(j)
        ker_pad = np.full([50, 50], np.nan)
        ker_pad[:ker.shape[0], :ker.shape[1]] = ker

        ker_mat = torch.FloatTensor(for_fft(ker, shape=np.shape(sp)))
        Fker = cf.fft(ker_mat)
        Fker = Fker.unsqueeze(0)

        sp = torch.from_numpy(sp).unsqueeze(0)
        bl = torch.from_numpy(bl).unsqueeze(0)

        k_center = ker

        if k_center.shape[0] != k_center.shape[1]:
            ker_pad = np.zeros([np.max(k_center.shape), np.max(k_center.shape)])
            ker_pad[:k_center.shape[0], :k_center.shape[1]] = k_center
            k_center = np.roll(ker_pad, [(ker_pad.shape[0] - k_center.shape[0]) // 2,
                                         (ker_pad.shape[1] - k_center.shape[1]) // 2],
                               [0, 1])
            ker_resize = cv2.resize(k_center, (15, 15))
        else:
            ker_resize = cv2.resize(k_center, (15, 15))
        ker_resize = ker_resize / np.sum(ker_resize)

        dic = {'bl': bl, 'sp': sp, 'Fker': Fker, 'ker': ker_pad.copy(), 'ker_resize': ker_resize.astype(np.float32),
               'std': std}
        return dic











