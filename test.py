from torch.utils.data.dataloader import DataLoader
import torch
import argparse
from data_loader.dataset import Test_Dataset, Test_Dataset_possion
from utils.imtools import imshow, for_fft, for_fft_BCHW
import os
from time import time
from utils.imtools import Fker_ker_for_input
import utils.comfft as cf
import numpy as np
from tqdm import tqdm
from utils.noise_estimation import noise_estimate
from model.UNID import UNID


class Tester():
    def __init__(self, args, net, test_dset):
        self.args = args
        self.net = net
        self.test_DLoader = {}

        for name in test_dset.keys():
            self.test_DLoader[name] = DataLoader(test_dset[name], batch_size=1, shuffle=False,
                                                 num_workers=0, pin_memory=True)

        self.load_model()

        if self.args.save_img:
            for name in self.test_DLoader.keys():
                os.makedirs(self.args.test_save_dir + name + '/', exist_ok=True)

    def __call__(self):
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
                        m = i // 8 + 1
                        n = i % 8 + 1
                        if os.path.exists(os.path.join(self.args.test_save_dir + name,
                                                       'im_%d_ker_%d' % (m, n) + '.png')):
                            if self.args.ct:
                                t.update()
                                continue


                        bat_x.append(bat['bl'])
                        bat_y.append(bat['sp'])
                        bl = bat['bl'].cuda()

                        result = torch.zeros_like(bl)
                        # result2 = torch.zeros_like(bl)
                        B, C, H, W = bl.shape
                        # noise = torch.randn([200,C,H,W])* 1  * (self.args.test_sigma[sigma_count] / 255.)#0.01
                        # noise = noise.cuda()
                        ker, Fker, KS = Fker_ker_for_input(ker=bat['ker'], Fker=bat['Fker'])

                        # for aug in range(8):
                        #     mini_dataset = dataset_for_test(bat, 13, aug, name)
                        #     mini_dataload = DataLoader(mini_dataset, 4, num_workers=8)
                        #     flip = aug // 4
                        #     rottime = aug % 4
                        #     # print(aug)
                        #     for i, data in enumerate(mini_dataload):
                        #         net_input = data['net_input'].cuda()
                        #         Fker_aug = data['Fker'].cuda()
                        #         # KS_rp = data['KS']
                        #         KS_rp = np.repeat(KS, net_input.shape[0], axis=0)
                        #         ker_resize = data['ker_resize'].cuda()
                        #         # print(ker_resize.shape)
                        #         opt_db = self.eval_net(net_input, Fker_aug, KS_rp, ker_resize)
                        #         result += torch.rot90(self.flip(torch.sum(opt_db[-1], dim=0, keepdim=True), flip),
                        #                               -rottime, dims=[-2, -1])
                        #
                        # result /=104
                        # bat_opt.append(result.cpu())
                        # t.update()

                        parallel = 1
                        est_noise = noise_estimate(bl.cpu().numpy().squeeze(), 8)
                        for ni in range(10 // parallel):
                            aug = ni % 8

                            # rottime = np.random.randint(0,4)
                            # flip = np.random.randint(0,2)
                            rottime = aug % 4
                            flip = aug // 4

                            bl_r = self.flip(torch.rot90(bl, rottime, dims=[-2, -1]), flip)
                            B, C, H, W = bl_r.shape

                            # Fker_r = self.flip_Fker(torch.rot90(Fker, rottime, dims=[-3, -2]), flip)

                            x, y = np.where(~np.isnan(bat['ker'][0].numpy()))
                            x_max = np.max(x) + 1
                            y_max = np.max(y) + 1
                            k_center = bat['ker'][0, :x_max, :y_max]

                            k_aug = self.flip(torch.rot90(k_center, rottime, dims=[-2, -1]), flip).numpy()
                            # --------------------------------------------------------#
                            # k_aug = k_aug + np.random.randn(*k_aug.shape) * 0.001
                            # --------------------------------------------------------#

                            # k_aug = self.data_aug(k_center, aug).numpy()

                            ker_mat = torch.FloatTensor(for_fft(k_aug, shape=[H, W]))
                            Fker_aug = cf.fft(ker_mat)
                            # Fker_aug = Fker_aug.unsqueeze(0).unsqueeze(0)

                            Fker_aug = cf.fft(ker_mat).unsqueeze(0).unsqueeze(0).repeat([parallel, 1, 1, 1, 1])

                            knoise = np.random.randn(parallel, 1, k_aug.shape[0], k_aug.shape[1]) * 0.001
                            #
                            # # knoise = knoise.repeat([parallel, 1, 1, 1])
                            #
                            Fknoise = torch.FloatTensor(for_fft_BCHW(knoise, shape=[parallel, 1, H, W]))
                            Fknoise = torch.rfft(Fknoise, signal_ndim=2, onesided=False)
                            # # Fknoise = torch.randn_like(Fker_aug) * 0.04
                            # #
                            Fker_a_p = cf.toAmptitudeAndPhase(Fker_aug)
                            Fknoise_a_p = cf.toAmptitudeAndPhase(Fknoise)
                            #
                            Fknoise_shift = torch.stack([Fknoise_a_p[..., 0] * torch.cos(Fker_a_p[..., 1])
                                                            , Fknoise_a_p[..., 0] * torch.sin(Fker_a_p[..., 1])],
                                                        dim=-1)
                            Fker_noise = Fker_aug + Fknoise_shift
                            # #
                            Fker_noise = cf.div(Fker_noise, Fker_noise[:, :, :1, :1, :])
                            #
                            Fker_aug = Fker_noise.cuda()

                            if self.args.test_possion:
                                # noise = (torch.randn([parallel, C, H, W]) * 1 * bat['std']).cuda()
                                noise = (torch.randn([parallel, C, H, W]) * 1 * est_noise).cuda()
                            else:
                                # noise = (torch.randn([parallel, C, H, W]) * 1 * (
                                #             float(name.split('_')[1]) / 255.)).cuda()
                                noise = (torch.randn([parallel, C, H, W]) * 1 * est_noise).cuda()
                            net_input = bl_r + noise
                            # Fker_aug = Fker_aug.repeat([parallel, 1, 1, 1, 1]).cuda()
                            KS_rp = np.repeat(KS, parallel, axis=0)

                            ker_resize = self.flip(torch.rot90(bat['ker_resize'], rottime, dims=[-2, -1]), flip).cuda()

                            ker_resize = ker_resize.repeat([parallel, 1, 1, 1])

                            # ker_resize =(cv2.resize(k_aug, (15, 15)))
                            # ker_resize = ker_resize / np.sum(ker_resize).astype(np.float16)
                            # ker_resize =  torch.from_numpy(ker_resize).cuda()

                            # self.net.train()
                            opt_db = self.net(net_input, Fker_aug.cuda(), KS_rp, ker_resize)  # , opt_dn
                            # opt_db2 = self.net(bl+noise[ni],Fker,KS)
                            # result += opt_db[-1]
                            result += torch.rot90(self.flip(torch.sum(opt_db[-1], dim=0, keepdim=True), flip), -rottime,
                                                  dims=[-2, -1])
                            # result2 += opt_db2[-1]
                        result /= 10
                        result = result.cpu()
                        # result2 /= 1
                        # opt_db, opt_dn = self.eval_net(bat['bl'].cuda(), bat['Fker'].cuda())
                        bat_opt.append(result)
                        # bat_opt2.append(result2.cpu())
                        if self.args.save_img:
                            m = i // 8 + 1
                            n = i % 8 + 1
                            imshow(result, str='im_%d_ker_%d' % (m, n), dir=self.args.test_save_dir + name + '/')

                        t.update()

                    print('-------%s-------' % (name))
                    # print('INP_PSNR', '%2.2f' % aver_psnr_ds(bat_x, bat_y))
                    # print('OUT_PSNR', '%2.2f' % aver_psnr_ds(bat_opt, bat_y))
                    # # # print('OUT_PSNR', '%2.2f' % aver_psnr_ds(bat_opt2, bat_y))
                    # print('INP_SSIM', '%2.3f' % aver_ssim_ds(bat_x, bat_y))
                    # print('OUT_SSIM', '%2.3f' % aver_ssim_ds(bat_opt, bat_y))
                    # t2 = time()
                    # print('TestTime:%d' % (t2 - t1))

                    # if self.args.save_img:
                    #     for i in range(len(self.test_DLoader[name])):
                    #         m = i // 8 + 1
                    #         n = i % 8 + 1
                    #         # print(m,n)
                    #         imshow(bat_opt[i], str='im_%d_ker_%d' % (m, n), dir=self.args.test_save_dir + name + '/')
                    # return aver_psnr_ds(bat_opt, bat_y)

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





def log(args):
    logdir = os.path.join(args.test_save_dir, 'scripts')
    os.makedirs(logdir, exist_ok=True)
    from shutil import copyfile
    from utils.pytools import copytree, Unbuffered
    # copyfile(os.path.basename(__file__), 'result/' + args.info + '/scripts/' + os.path.basename(__file__))
    copyfile('config.py', args.test_save_dir + '/scripts/config.py')
    copyfile('head.py', args.test_save_dir + '/scripts/head.py')
    copyfile('train.py', args.test_save_dir + '/scripts/train.py')
    copyfile('test.py', args.test_save_dir + '/scripts/test.py')
    copytree('./data_loader/', args.test_save_dir + '/scripts/data_loader')
    copytree('./model/', args.test_save_dir + '/scripts/model')
    copytree('./utils/', args.test_save_dir + '/scripts/utils')


class get_test_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='Learning Deep Non-Blind Deconvolution Without Ground Truth Images')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--layers', type=int, default=4, help='net layers')
        self.parser.add_argument('--save_img', default=False, help='save images into file')
        self.parser.add_argument('--ct',default=False)
        self.parser.parse_args(namespace=self)

        self.beta = [0.005, 0.1, 0.1, 0.1, 0.1]

        self.dataset_name = ['Levin']  # , 'Sun' 'Set12',
        self.test_sigma = [2.55]  # 2.55, 5.1, 7.65, 10.2, 12.75  , 5.1, 7.65, 10.2, 12.75   2.55, 5.1, 7.65, 10.2,
        self.test_peak = [1024, 512, 256]
        self.test_ckp_dir = 'pretrained_model/UNID-errorKernel'
        self.test_bl_dir = {}
        self.test_sp_dir = {}

        self.ker_dir = './data/kernels/Levin09_v7.mat'

        for dset in self.dataset_name:
            self.test_sp_dir[dset] = './data/{}/sharp/'.format(dset)
            for sigma in self.test_sigma:
                self.test_bl_dir[dset + '_' + str(sigma)] = './data/' + dset + '/sigma_' + str(
                    sigma) + '_ker_levin_circular/BlurryNoiseDset/'

        self.test_save_dir = './deblur_thesis/circular_noFDN/'



if __name__ == "__main__":
    args = get_test_config()
    log(args)
    torch.cuda.set_device(args.gpu_idx)
    net = UNID(args).cuda()

    result = {}
    test_dset = {}
    for dset in args.dataset_name:
        for sigma in args.test_sigma:
            test_dset[dset + '_' + str(sigma)] = Test_Dataset(args.test_sp_dir[dset],
                                                              args.test_bl_dir[dset + '_' + str(sigma)],
                                                              args.ker_dir)

    test = Tester(args, net, test_dset)
    test()
    print('[*] Finish!')
