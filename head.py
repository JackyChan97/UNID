import os
import sys
import time

import torch

from utils.pytools import copytree, Unbuffered
from glob import glob

def log(args):
    ''' Folder settings when saving training results'''
    if not os.path.exists('result') and ~args.debug:
        os.makedirs('result')
    if not os.path.exists('result/' + args.info) and ~args.debug:
        os.mkdir('result/' + args.info)
    if not os.path.exists('result/' + args.info + '/img') and ~args.debug:
        os.mkdir('result/' + args.info + '/img')
    if not os.path.exists('result/' + args.info + '/scripts') and ~args.debug:
        os.mkdir('result/' + args.info + '/scripts')
    if not os.path.exists('result/' + args.info + '/ckp') and ~args.debug:
        os.mkdir('result/' + args.info + '/ckp')

    # if ~args.debug and args.log == True:
    #     sys.stdout = open('result/' + args.info + '/' + 'file.txt', 'a')
    #     sys.stderr = open('result/' + args.info + '/' + 'file.txt', 'a')

    print('[*] Info:', time.ctime())
    print('[*] Info:', os.path.basename(__file__))

    # if ~args.debug and args.log == True and args.resume == False:
    if not (args.resume or args.resume_from_last):
        assert (not os.path.exists('result/' + args.info + '/scripts/config.py')) or (
            args.rewrite), 'File already existed!!!'
        # if ~args.debug and args.resume == False:
        from shutil import copyfile
        copyfile(os.path.basename(__file__), 'result/' + args.info + '/scripts/' + os.path.basename(__file__))
        files = glob('./*.py')
        for f in files:
            copyfile(f, 'result/' + args.info + '/scripts/%s' % f)

        copytree('./data_loader/', 'result/' + args.info + '/scripts/data_loader')
        copytree('./model/', 'result/' + args.info + '/scripts/model')
        copytree('./utils/', 'result/' + args.info + '/scripts/utils')
        copytree('./data_loader/', 'result/' + args.info + '/scripts/data_loader')
        copytree('./model/', 'result/' + args.info + '/scripts/model')
        copytree('./utils/', 'result/' + args.info + '/scripts/utils')

    sys.stdout = Unbuffered(sys.stdout)
    # torch.cuda.set_device(args.gpu_idx)

    from torch import multiprocessing
    multiprocessing.set_sharing_strategy('file_system')
    torch.set_num_threads(1)
