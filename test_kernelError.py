from torch.utils.data.dataloader import DataLoader
import torch
import argparse
from data_loader.dataset import Test_Dataset, Test_Dataset_possion, Test_NoiseKernel, Test_Lai_NoiseKernel

from utils.imtools import imshow, for_fft, for_fft_BCHW, inverse_for_fft, inverse_for_fft_BCHW
import os
from time import time
from utils.imtools import Fker_ker_for_input
import utils.comfft as cf
import numpy as np
from tqdm import tqdm
from utils.metrics_kernelError import aver_bmp_psnr_ssim_par, aver_bmp_psnr
from utils.noise_estimation import noise_estimate
from model.UNID import UNID

from shutil import copyfile
from utils.pytools import copytree


class Tester():
    def __init__(self, args, net, test_dset):
        self.args = args
        self.net = net
        self.test_DLoader = {}

        for name in test_dset.keys():
            self.test_DLoader[name] = DataLoader(test_dset[name], batch_size=1, shuffle=args.random,
                                                 num_workers=0, pin_memory=True)

        self.load_model()

        if self.args.save_img:
            for name in self.test_DLoader.keys():
                os.makedirs(self.args.test_save_dir + self.args.dataset_name + '_' + name + '/', exist_ok=True)

    def __call__(self, ensemble):
        self.net.eval()
        t1 = time()
        with torch.no_grad():
            for sigma_count, name in enumerate(self.test_DLoader.keys()):
                bat_x = []
                bat_y = []
                bat_opt = []
                with tqdm(total=len(self.test_DLoader[name]), ncols=100, position=0, leave=True) as t:
                    for i, bat in enumerate(self.test_DLoader[name]):

                        i = bat['idx'][0]
                        if i % 1 == 0:
                            if os.path.exists(
                                    os.path.join(self.args.test_save_dir + self.args.dataset_name + '_' + name,
                                                 bat['name'][0] + '.png')):
                                if self.args.ct:
                                    t.set_postfix({'idx': i})
                                    t.update()
                                    continue

                            bat_x.append(bat['bl'])
                            bat_y.append(bat['sp'])
                            bl = bat['bl'].cuda()

                            result = torch.zeros_like(bl)
                            B, C, H, W = bl.shape

                            ker, Fker, KS = Fker_ker_for_input(ker=bat['ker'], Fker=bat['Fker'])

                            if self.args.sigma == '0':
                                est_noise = 0.01
                            else:
                                est_noise = noise_estimate(bl.squeeze().cpu().numpy(), 8)

                            parallel = 1
                            for ni in range(ensemble // parallel):
                                aug = ni % 8

                                rottime = aug % 4
                                flip = aug // 4

                                bl_r = self.flip(torch.rot90(bl, rottime, dims=[-2, -1]), flip)
                                B, C, H, W = bl_r.shape
                                x, y = np.where(~np.isnan(bat['ker'][0].numpy()))
                                x_max = np.max(x) + 1
                                y_max = np.max(y) + 1
                                k_center = bat['ker'][0, :x_max, :y_max]

                                # kernel permutation
                                k_aug = self.flip(torch.rot90(k_center, rottime, dims=[-2, -1]), flip).numpy()
                                ker_mat = torch.FloatTensor(for_fft(k_aug, shape=[H, W]))
                                Fker_aug = cf.fft(ker_mat).unsqueeze(0).unsqueeze(0).repeat([parallel, 1, 1, 1, 1])

                                knoise = np.random.randn(parallel, 1, k_aug.shape[0],
                                                         k_aug.shape[1]) * 0.001

                                Fknoise = torch.FloatTensor(for_fft_BCHW(knoise, shape=[parallel, 1, H, W]))
                                Fknoise = torch.rfft(Fknoise, signal_ndim=2, onesided=False)
                                Fker_a_p = cf.toAmptitudeAndPhase(Fker_aug)
                                Fknoise_a_p = cf.toAmptitudeAndPhase(Fknoise)

                                Fknoise_shift = torch.stack([Fknoise_a_p[..., 0] * torch.cos(Fker_a_p[..., 1])
                                                                , Fknoise_a_p[..., 0] * torch.sin(Fker_a_p[..., 1])],
                                                            dim=-1)
                                Fker_noise = Fker_aug + Fknoise_shift
                                Fker_noise = cf.div(Fker_noise, Fker_noise[:, :, :1, :1, :])

                                Fker_aug = Fker_noise.cuda()

                                noise = (torch.randn([parallel, C, H, W]) * 1 * (est_noise)).cuda()
                                net_input = bl_r + noise
                                KS_rp = np.repeat(KS, parallel, axis=0)

                                ker_resize = self.flip(torch.rot90(bat['ker_resize'], rottime, dims=[-2, -1]),
                                                       flip).cuda()

                                ker_resize = ker_resize.repeat([parallel, 1, 1, 1])

                                opt_db = self.net(net_input, Fker_aug.cuda(), KS_rp, ker_resize)

                                result += torch.rot90(self.flip(torch.sum(opt_db[-1], dim=0, keepdim=True), flip),
                                                      -rottime, dims=[-2, -1])
                            result /= ensemble
                            bat_opt.append(result.cpu())
                            if self.args.save_img:
                                m = i // 8 + 1
                                n = i % 8 + 1
                                imshow(result, str='im_%d_ker_%d' % (m, n),
                                       dir=self.args.test_save_dir + self.args.dataset_name + '_' + name + '/')

                            t.set_postfix({'idx': i})
                            t.update()
                    print(ensemble)
                    if self.args.calc:
                        PSNR, SSIM = aver_bmp_psnr_ssim_par(bat_opt, bat_y, bd_cut=self.args.bd_cut,
                                                            to_int=True,
                                                            ssim_compute=True)

                        print('-------%s-------' % (name))
                        print('OUT_PSNR', '%2.2f' % PSNR)
                        print('OUT_SSIM', '%2.3f' % SSIM)
                        t2 = time()
                        print('TestTime:%d' % (t2 - t1))
                    else:
                        print('-------%s-------' % (name))
                        t2 = time()
                        print('TestTime:%d' % (t2 - t1))

    def test_lai(self, ensemble):
        self.net.eval()
        t1 = time()
        with torch.no_grad():
            for sigma_count, name in enumerate(self.test_DLoader.keys()):
                bat_x = []
                bat_y = []
                bat_opt = []
                # bat_opt2 = []
                with tqdm(total=len(self.test_DLoader[name]), ncols=100, position=0, leave=True) as t:
                    for i, bat in enumerate(self.test_DLoader[name]):
                        i = bat['idx'][0]
                        if os.path.exists(os.path.join(self.args.test_save_dir + self.args.dataset_name + '_' + name,
                                                       bat['name'][0] + '.png')):
                            if self.args.ct:
                                t.update()
                                continue
                        bat_x.append(bat['bl'])
                        bat_y.append(bat['sp'])
                        bl = bat['bl'].cuda()
                        bl = torch.cat([bl[:, :, :, :, 0], bl[:, :, :, :, 1], bl[:, :, :, :, 2]], dim=0)

                        result = torch.zeros_like(bl)

                        ker, Fker, KS = Fker_ker_for_input(ker=bat['ker'], Fker=bat['Fker'])
                        if self.args.sigma == '0':
                            est_noise = 0.01
                        else:
                            est_noise = noise_estimate(bl.squeeze().permute([1, 2, 0]).cpu().numpy(), 8)
                        parallel = 1
                        for ni in range(ensemble // parallel):
                            aug = ni % 8
                            rottime = aug % 4
                            flip = aug // 4

                            bl_r = self.flip(torch.rot90(bl, rottime, dims=[-2, -1]), flip)
                            B, C, H, W = bl_r.shape

                            x, y = np.where(~np.isnan(bat['ker'][0].numpy()))
                            x_max = np.max(x) + 1
                            y_max = np.max(y) + 1
                            k_center = bat['ker'][0, :x_max, :y_max]
                            # kernel permutation
                            k_aug = self.flip(torch.rot90(k_center, rottime, dims=[-2, -1]), flip).numpy()
                            ker_mat = torch.FloatTensor(for_fft(k_aug, shape=[H, W]))
                            Fker_aug = cf.fft(ker_mat).unsqueeze(0).unsqueeze(0).repeat([parallel * 3, 1, 1, 1, 1])

                            knoise = np.random.randn(parallel * 3, 1, k_aug.shape[0],
                                                     k_aug.shape[1]) * 0.001
                            Fknoise = torch.FloatTensor(for_fft_BCHW(knoise, shape=[parallel * 3, 1, H, W]))
                            Fknoise = torch.rfft(Fknoise, signal_ndim=2, onesided=False)
                            Fker_a_p = cf.toAmptitudeAndPhase(Fker_aug)
                            Fknoise_a_p = cf.toAmptitudeAndPhase(Fknoise)
                            Fknoise_shift = torch.stack([Fknoise_a_p[..., 0] * torch.cos(Fker_a_p[..., 1]),
                                                         Fknoise_a_p[..., 0] * torch.sin(Fker_a_p[..., 1])], dim=-1)
                            Fker_noise = Fker_aug + Fknoise_shift
                            Fker_noise = cf.div(Fker_noise, Fker_noise[:, :, :1, :1, :])

                            Fker_aug = Fker_noise.cuda()
                            noise = (torch.randn([parallel * 3, C, H, W]) * 1 * est_noise).cuda()

                            bl_r = bl_r.repeat([parallel, 1, 1, 1])
                            net_input = bl_r + noise
                            KS_rp = np.repeat(KS, parallel * 3, axis=0)

                            opt_db = self.net(net_input, Fker_aug.cuda(), KS_rp, None)
                            output = torch.split(opt_db[-1], 3, dim=0)
                            output = torch.stack(output, dim=0)
                            output = torch.sum(output, dim=0)
                            result += torch.rot90(self.flip(output, flip), -rottime, dims=[-2, -1])

                        result /= ensemble
                        result = result.cpu()
                        result = torch.stack(torch.split(result, 1, dim=0), dim=-1)

                        if self.args.save_img:
                            imshow(result, str=bat['name'][0],
                                   dir=self.args.test_save_dir + self.args.dataset_name + '_' + name + '/')
                        t.update()
                        torch.cuda.empty_cache()

                    if self.args.calc:
                        PSNR, SSIM = aver_bmp_psnr_ssim_par(bat_opt, bat_y, bd_cut=self.args.bd_cut, to_int=True,
                                                            ssim_compute=True)

                        print('-------%s-------' % (name))
                        print('OUT_PSNR', '%2.2f' % PSNR)
                        print('OUT_SSIM', '%2.3f' % SSIM)
                        t2 = time()
                        print('TestTime:%d' % (t2 - t1))
                    else:
                        print('-------%s-------' % (name))
                        t2 = time()
                        print('TestTime:%d' % (t2 - t1))

    def load_model(self):
        ckp = torch.load(self.args.test_ckp_dir, map_location=lambda storage, loc: storage.cuda(self.args.gpu_idx))
        self.net.load_state_dict(ckp['model'])
        return ckp

    def eval_net(self, bl, *args):
        with torch.no_grad():
            self.net.eval()
            bl = bl.cuda()
            db = self.net(bl, *args)
        return db

    @staticmethod
    def _ker_to_list(ker):
        import numpy as np
        ker = ker.numpy()
        Kker = [None] * ker.shape[0]
        for i in range(ker.shape[0]):
            x, y = np.where(~np.isnan(ker[i]))
            x_max = np.max(x)
            y_max = np.max(y)
            Kker[i] = ker[i, :x_max, :y_max]
        return Kker

    def flip_Fker(self, Fker, time):
        if time == 0:
            return Fker
        elif time == 1:
            return torch.stack([torch.flip(cf.real(Fker), [2]), torch.flip(cf.image(Fker), [2])], dim=-1)
        elif time == 2:
            return torch.stack([torch.flip(cf.real(Fker), [3]), torch.flip(cf.image(Fker), [3])], dim=-1)

    def flip(self, x, time):
        if time == 0:
            return x
        elif time == 1:
            return torch.flip(x, [-2])
        elif time == 2:
            out = torch.flip(x, [-1])
            return out


class get_test_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(
            description='Learning Deep Non-Blind Deconvolution Without Ground Truth Images')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--parallel', default=True, type=bool, help='Parallel computation of the PSNR')
        self.parser.add_argument('--rewrite', default=False)
        # Problem Settings
        self.parser.add_argument('--dataset_name', default='Sun', choices=['Levin', 'Sun', 'Lai'])

        self.parser.add_argument('-s', '--sigma', default='2.55', choices=['0', '2.55'])

        self.parser.add_argument('--save_img', default=False, type=bool, help='save images into file')
        self.parser.add_argument('--ct', default=False)
        self.parser.add_argument('--test_save_dir',
                                 default='./deblur_errorKernel/UNID_errorKernel/')
        self.parser.add_argument('--test_ckp_dir',
                                 default='./pretrained_model/UNID-accurateKernel')

        self.parser.add_argument('--random', default=False)
        self.parser.add_argument('--calc_PSNR', default=False)
        self.parser.parse_args(namespace=self)

        # Predefined parameters

        if self.sigma == '0':
            self.taper = 'same'
            self.beta = [0.005, 0.1, 0.1, 0.1, 0.1]

        if self.sigma == '2.55':
            self.taper = 'valid'
            self.beta = [0.005, 0.1, 0.1, 0.1, 0.1]


        # Data Preparation
        self.ker_dir = {}
        self.ker_dir['cho'] = './data/{0}_NK/BD_cho_and_lee_tog_2009/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['fergus'] = './data/{0}_NK/BD_fergus_tog_2006/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['levin'] = './data/{0}_NK/BD_levin_cvpr_2011/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['pan'] = './data/{0}_NK/BD_pan_cvpr_2016/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['sun'] = './data/{0}_NK/BD_sun_iccp_2013/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['xu'] = './data/{0}_NK/BD_Xu_eccv_2010/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['michaeli'] = './data/{0}_NK/BD_Michaeli_eccv_2014/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['xu10'] = './data/{0}_NK/BD_xu_eccv_2010/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['xu13'] = './data/{0}_NK/BD_xu_cvpr_2013/kernel_estimates/'.format(self.dataset_name)
        self.ker_dir['perrone'] = './data/{0}_NK/BD_perrone_cvpr_2014/kernel_estimates/'.format(self.dataset_name)
        self.test_sp_dir = './data/{0}_NK/sharp/'.format(self.dataset_name)

        # Select Blind Deconvolution Kernels by the kernels.
        if self.dataset_name == 'Sun':
            self.test_bl_dir = './data/Sun_NK/sigma_{0}_ker_levin/'.format(self.sigma)
            self.test_ker = ['cho', 'xu', 'michaeli']
            self.tr_ker_dir = 'data/kernels/Levin09_v7.mat'
            self.bd_cut = 28
        elif self.dataset_name == 'Levin':
            self.test_bl_dir = './data/Levin_NK/sigma_{0}_ker_levin/'.format(self.sigma)
            self.test_ker = ['pan', 'sun']
            self.tr_ker_dir = 'data/kernels/Levin09_v7.mat'
            self.bd_cut = 15
        elif self.dataset_name == 'Lai':
            self.test_bl_dir = './data/Lai_NK/sigma_{0}_ker_lai/'.format(self.sigma)
            self.test_ker = ['xu10', 'xu13', 'sun', 'perrone']
            self.bd_cut = 15
        elif self.dataset_name == 'Lai_Real':
            self.test_bl_dir = './data/Lai_NK/real/blurry/'

        if self.taper == 'valid':
            if self.dataset_name != 'Lai_Real':
                self.test_bl_dir = self.test_bl_dir[:-1] + '_valid/'
        elif self.taper == 'taper':
            self.test_bl_dir = self.test_bl_dir[:-1] + '_taper/'

        self.img_num = None


def log(args):
    logdir = os.path.join(args.test_save_dir, 'scripts')
    os.makedirs(logdir, exist_ok=args.rewrite)

    # copyfile(os.path.basename(__file__), 'result/' + args.info + '/scripts/' + os.path.basename(__file__))
    copyfile('config.py', args.test_save_dir + '/scripts/config.py')
    copyfile('head.py', args.test_save_dir + '/scripts/head.py')
    copyfile('train.py', args.test_save_dir + '/scripts/train.py')
    copyfile('test.py', args.test_save_dir + '/scripts/test.py')
    copyfile('test_kernelError.py', args.test_save_dir + '/scripts/test_kernelError.py')
    copyfile('config_kernelError.py', args.test_save_dir + '/scripts/config_kernelError.py')

    copytree('./data_loader/', args.test_save_dir + '/scripts/data_loader')
    copytree('./model/', args.test_save_dir + '/scripts/model')
    copytree('./utils/', args.test_save_dir + '/scripts/utils')


if __name__ == "__main__":
    args = get_test_config()
    if args.save_img:
        log(args)
        print(args.test_save_dir)
    torch.cuda.set_device(args.gpu_idx)
    net = UNID(args).cuda()

    result = {}
    test_dset = {}
    if args.dataset_name == 'Lai':
        for name in args.test_ker:
            test_dset[name] = Test_Lai_NoiseKernel(args.test_sp_dir, args.test_bl_dir, args.ker_dir[name], args.taper)
    else:
        for name in args.test_ker:
            test_dset[name] = Test_NoiseKernel(args.test_sp_dir, args.test_bl_dir, args.ker_dir[name], args.tr_ker_dir,
                                               args.taper)

    test = Tester(args, net, test_dset)
    psnrs = []
    ssims = []
    if args.dataset_name == 'Lai':
        test.test_lai(10)
    else:
        test(10)

    print('[*] Finish!')
