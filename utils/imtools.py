''' kernel preparation and forward models '''
import numpy as np
from PIL import Image
from scipy.ndimage import filters
import torch
import torch.nn.functional as F

'''torch2np'''


def torch2np(x_tensor):
    if isinstance(x_tensor, np.ndarray):
        return x_tensor
    elif x_tensor.is_cuda == False:
        x = x_tensor.numpy()
        return x
    else:
        x = x_tensor.detach().cpu().numpy()
        return x


def np2torch(x, cuda=False):
    if isinstance(x, torch.Tensor):
        return x
    else:
        x = torch.from_numpy(x.copy())
        x = x.type(torch.float32)
        if cuda == True:
            x = x.cuda()
        return x


def for_fft(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(-2, -1))
    return ker_mat


def for_fft_BCHW(ker, shape):
    ker_mat = np.zeros(shape, dtype=np.float32)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape)[-2:] / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1], :ker_shape[2], :ker_shape[3]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(-2, -1))
    return ker_mat


def inverse_for_fft(ker_mat, shape):
    ker_shape = np.asarray(shape)
    circ = np.ndarray.astype(np.floor(ker_shape / 2), dtype=np.int)
    ker_mat = np.roll(ker_mat, circ, axis=(-2, -1))
    ker = ker_mat[:shape[0], :shape[1]]
    return ker


def inverse_for_fft_BCHW(ker_mat, shape):
    ker_shape = np.asarray(shape)
    circ = np.ndarray.astype(np.floor((ker_shape)[-2:] / 2), dtype=np.int)
    ker_mat = np.roll(ker_mat, circ, axis=(-2, -1))
    ker = ker_mat[:ker_shape[0], :ker_shape[1], :ker_shape[2], :ker_shape[3]]
    return ker


def fspecial(type, *args):
    dtype = np.float32
    if type == 'average':
        siz = (args[0], args[0])
        h = np.ones(siz) / np.prod(siz)
        return h.astype(dtype)
    elif type == 'gaussian':
        p2 = args[0]
        p3 = args[1]
        siz = np.array([(p2[0] - 1) / 2, (p2[1] - 1) / 2])
        std = p3
        x1 = np.arange(-siz[1], siz[1] + 1, 1)
        y1 = np.arange(-siz[0], siz[0] + 1, 1)
        x, y = np.meshgrid(x1, y1)
        arg = -(x * x + y * y) / (2 * std * std)
        h = np.exp(arg)
        sumh = sum(map(sum, h))
        if sumh != 0:
            h = h / sumh
        return h.astype(dtype)
    elif type == 'motion':
        p2 = args[0]
        p3 = args[1]
        len = max(1, p2)
        half = (len - 1) / 2
        phi = np.mod(p3, 180) / 180 * np.pi

        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        xsign = np.sign(cosphi)
        linewdt = 1

        eps = np.finfo(float).eps
        sx = np.fix(half * cosphi + linewdt * xsign - len * eps)
        sy = np.fix(half * sinphi + linewdt - len * eps)

        x1 = np.arange(0, sx + 1, xsign)
        y1 = np.arange(0, sy + 1, 1)
        x, y = np.meshgrid(x1, y1)

        dist2line = (y * cosphi - x * sinphi)
        rad = np.sqrt(x * x + y * y)

        lastpix = np.logical_and(rad >= half, np.abs(dist2line) <= linewdt)
        lastpix.astype(int)
        x2lastpix = half * lastpix - np.abs((x * lastpix + dist2line * lastpix * sinphi) / cosphi)
        dist2line = dist2line * (-1 * lastpix + 1) + np.sqrt(dist2line ** 2 + x2lastpix ** 2) * lastpix
        dist2line = linewdt + eps - np.abs(dist2line)
        logic = dist2line < 0
        dist2line = dist2line * (-1 * logic + 1)

        h1 = np.rot90(dist2line, 2)
        h1s = np.shape(h1)
        h = np.zeros(shape=(h1s[0] * 2 - 1, h1s[1] * 2 - 1))
        h[0:h1s[0], 0:h1s[1]] = h1
        h[h1s[0] - 1:, h1s[1] - 1:] = dist2line
        h = h / sum(map(sum, h)) + eps * len * len

        if cosphi > 0:
            h = np.flipud(h)

        return h.astype(dtype)


'''convolution operator'''


def cconv_torch(x, ker):
    with torch.no_grad():
        x_h, x_v = x.size()
        conv_ker = np.flip(np.flip(ker, 0), 1)
        ker = torch.FloatTensor(conv_ker.copy()).cuda()
        k_h, k_v = ker.size()
        k2_h = k_h // 2
        k2_v = k_v // 2
        x = torch.cat((x[-k2_h:, :], x, x[0:k2_h, :]), dim=0).cuda()
        x = torch.cat((x[:, -k2_v:], x, x[:, 0:k2_v]), dim=1).cuda()
        x = x.unsqueeze(0).cuda()
        x = x.unsqueeze(1).cuda()
        ker = ker.unsqueeze(0).cuda()
        ker = ker.unsqueeze(1).cuda()
        y1 = F.conv2d(x, ker).cuda()
        y1 = torch.squeeze(y1)
        y = y1[-x_h:, -x_v:]
    return y


def cconv_np(data, ker, mode='wrap'):
    # notice it might give false result when x is not the type.
    # Pay Attention, data and kernel is not interchangeable!!!
    if mode == 'wrap':
        return filters.convolve(data, ker, mode='wrap')
    elif mode == 'valid':
        return fftconvolve(data, ker, mode='valid')


def conv_valid(x, ker):
    '''valid convolution'''
    with torch.no_grad():
        conv_ker = np.flip(np.flip(ker, 0), 1)
        ker = torch.FloatTensor(conv_ker.copy()).cuda()
        ker = ker.unsqueeze(0).cuda()
        ker = ker.unsqueeze(1).cuda()
        y1 = F.conv2d(x, ker).cuda()
    return y1


def deconv_valid(x, ker):
    '''inverse operator of valid convolution'''
    conv_ker = np.flip(np.flip(ker, 0), 1)
    ker = torch.FloatTensor(conv_ker.copy()).cuda()

    ker = ker.unsqueeze(0).cuda()
    ker = ker.unsqueeze(1).cuda()
    y1 = F.conv_transpose2d(x, ker).cuda()
    return y1


def imshow(x_in, str, dir='tmp/'):
    x = torch2np(x_in)
    x = np.squeeze(x)
    if len(x.shape) == 2:
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int, 'L').save(dir + str + '.png')
    elif len(x.shape) == 3:
        x[x > 1] = 1
        x[x < 0] = 0
        x_int = np.uint8(np.around(x * 255))
        Image.fromarray(x_int, 'RGB').save(dir + str + '.png')


from scipy.signal import fftconvolve


def pad_for_kernel(img, kernel, mode):
    # hy = (kernel.shape[0] - 1) // 2
    # hx = (kernel.shape[0] - 1) - hy
    # wy = (kernel.shape[1] - 1) // 2
    # wx = (kernel.shape[1] - 1) - wy
    hx = (kernel.shape[0] - 1) // 2
    hy = (kernel.shape[0] - 1) - hx
    wx = (kernel.shape[1] - 1) // 2
    wy = (kernel.shape[1] - 1) - wx
    # p = [(d - 1) // 2 for d in kernel.shape]
    padding = [[hx, hy], [wx, wy]]
    return np.pad(img, padding, mode)


def edgetaper(img, kernel, n_tapers=3):
    '''tap edges for immitation of circulant boundary. '''
    alpha = edgetaper_alpha(kernel, img.shape)
    _kernel = kernel
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel(img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha * img + (1 - alpha) * blurred
    return img


def edgetaper_alpha(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1 - i), img_shape[i] - 1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def pad_for_ker_and_db(bl, ker, db=None):
    '''pad the kernel by edge tapper when db is None and pad by convolution with db. See FDN'''
    im_num = len(ker)
    bl_pad = torch.zeros_like(bl)
    for i in range(im_num):
        _bl = torch2np(bl[i:i + 1, ])
        if db is None:
            out = edgetaper(pad_for_kernel(_bl[0, 0, :, :], ker[i], 'edge'), ker[i])
            bl_pad[i:i + 1, ] = np2torch(out, cuda=True).unsqueeze(0).unsqueeze(0)
        else:
            _db = torch2np(db[i:i + 1, ])
            hy = (ker[i].shape[0] - 1) // 2
            hx = (ker[i].shape[0] - 1) - hy
            wy = (ker[i].shape[1] - 1) // 2
            wx = (ker[i].shape[1] - 1) - wy
            out = cconv_np(_db[0, 0,], ker[i])
            out[hx:-hy, wx:-wy] = _bl[0, 0, hx:-hy, wx:-wy]
            bl_pad[i:i + 1, ] = np2torch(out, cuda=True).unsqueeze(0).unsqueeze(0)
    return bl_pad


'''optimizer'''


def cg_torch(A, B, X0=None, n_loop=50, tol=1e-5, verbose=False):
    with torch.no_grad():
        X0 = X0
        if X0 is None:
            X0 = torch.zeros_like(B)
        r0 = A(X0) - B
        p = -r0
        X = X0
        # Iteration
        err0 = float("inf")
        for iter in range(n_loop):
            Ap = A(p)
            alpha = torch.dot(r0.view(-1), r0.view(-1)) / torch.dot(p.view(-1), Ap.view(-1))
            X = X + alpha * p
            r1 = r0 + alpha * Ap
            err1 = torch.norm(r1)
            if verbose == True:
                print('iter %d, err %2.2f' % (iter, err1.cpu().numpy()))

            beta = torch.dot(r1.view(-1), r1.view(-1)) / torch.dot(r0.view(-1), r0.view(-1))
            p = -r1 + beta * p
            r0 = r1
            if err1 < tol:
                return X
            if err1 > err0:
                pass
            else:
                err0 = err1

            if iter == n_loop - 1:
                # print('[!] CG Method reaches its maximum loop!, The final step err is {}'.format(err1))
                return X


def Fker_ker_for_input(ker=None, Fker=None):
    B, H, W = ker.shape

    if Fker is not None:
        # FFker = [None] * Fker.size(0)
        # for i in range(Fker.size(0)):
        #     FFker[i] = Fker[i,].cuda()
        FFker = Fker.cuda()
    else:
        FFker = None

    if ker is not None:
        Kker = np.zeros_like(ker)
        KS = [None] * ker.shape[0]
        for i in range(ker.shape[0]):
            x, y = np.where(~np.isnan(ker[i].numpy()))
            # x_xin = np.min(x)
            # y_min = np.min(y)
            x_max = np.max(x) + 1
            y_max = np.max(y) + 1
            KS[i] = [x_max, y_max]
            Kker[i, :x_max, :y_max] = ker[i, :x_max, :y_max]
            # roll the kernel to middle
            Kker[i] = np.roll(Kker[i], [(H // 2) - (x_max // 2), (W // 2 - y_max // 2)], [-2, -1])

    else:
        Kker = None

    Kker = np.array(Kker).astype(np.float32)
    return Kker, FFker, KS


from skimage.feature import register_translation
from skimage.transform import warp
from skimage.registration import phase_cross_correlation


''' Db-INV module without P-FCN'''


def ker_inv_by_fft(y, z, Fker, gamma, cuda=False):
    from utils.wavelet import generate_wavelet, wv_norm, Fwv

    Dec, _ = generate_wavelet(frame=1)
    chan_num = len(Dec)
    gamma = gamma.view(chan_num, 1, 1, 1)

    if cuda == True:
        gamma = gamma.cuda()

    im_num = y.size()[0]
    x = torch.zeros_like(y)
    for i in range(im_num):
        shape = y[i,].size()[-2:]
        Fw = Fwv(Dec, shape=shape)

        Fker_conj = cf.conj(Fker[i])
        Fw_conj = cf.conj(Fw)

        Fw = Fw.cuda()
        Fker_conj = Fker_conj.cuda()
        Fw_conj = Fw_conj.cuda()

        Fy = cf.fft(y[i,])
        Fz = cf.fft(z[i,]).cuda()
        Fx_num = cf.mul(Fker_conj, Fy) + torch.sum(gamma * cf.mul(Fw_conj, cf.mul(Fw, Fz)), dim=0)
        Fx_den = cf.abs_square(Fker[i], keepdim=True) + torch.sum(gamma * cf.mul(Fw_conj, Fw), dim=0)
        Fx = cf.div(Fx_num, Fx_den)
        x[i,] = cf.ifft(Fx)
    return x



from scipy.ndimage.interpolation import shift
from utils import comfft as cf


def image_register(bl, sp, ker, Fker, factor=10):
    from utils.wavelet import generate_wavelet, wv_norm, Fwv

    dec2d, _ = generate_wavelet(1, dim=2)
    norm = torch.from_numpy(wv_norm(dec2d))
    gamma = torch.ones(len(dec2d)) * 0.005 / norm

    bl = torch.from_numpy(bl.copy()).unsqueeze(0).cuda()
    dn = torch.zeros_like(bl)
    db0 = ker_inv_by_fft(bl, dn, [Fker.cuda()], gamma=gamma, cuda=True).squeeze().cpu().numpy()
    displace, _, _ = phase_cross_correlation(sp, db0, upsample_factor=factor)
    displace_inv = -displace
    # ker_aligned = warp(ker,shift_inv)
    ker_aligned = shift(ker, displace_inv, mode='wrap')

    # ### Check Correctness
    # nz_ker_aligned_mat = torch.FloatTensor(for_fft(ker_aligned, shape=np.shape(sp)))
    # nz_Fker_aligned = cf.fft(nz_ker_aligned_mat)
    # db0_aligned = InvModule(bl, dn, [nz_Fker_aligned.cuda()], gamma=gamma, cuda=True).squeeze().cpu().numpy()
    # ###

    return ker_aligned


def get_kernel_size(ker, arrayShape, threshold):
    h, w = arrayShape
    energy = np.sum(ker)
    kh = (h - 1) // 2
    kw = (w - 1) // 2
    lowerh = kh
    upperh = 1
    lowerw = kw
    upperw = 1

    while (abs(lowerh - upperh) > 2 and abs(lowerw - upperw) > 2):
        middleh = (upperh + lowerh) // 2
        middlew = (upperw + lowerw) // 2
        if np.sum(ker[middleh:-middleh, upperw:-upperw]) >= threshold * energy or np.sum(
                ker[upperh:-upperh, upperw:-upperw]) - np.sum(ker[middleh:-middleh, upperw:-upperw]) <= (
                1 - threshold) * energy:
            upperh = middleh
        elif np.sum(ker[middleh:-middleh, upperw:-upperw]) < threshold * energy:
            lowerh = middleh

        if np.sum(ker[upperh:-upperh, middlew:-middlew]) > threshold * energy or np.sum(
                ker[upperh:-upperh, upperw:-upperw]) - np.sum(ker[upperh:-upperh, middlew:-middlew]) <= (
                1 - threshold) * energy:
            upperw = middlew
        elif np.sum(ker[upperh:-upperh, middlew:-middlew]) < threshold * energy:
            lowerw = middlew

    return [h - upperw * 2, w - upperw * 2]


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def center_of_mass(im):
    H, W = np.shape(im)
    x = np.expand_dims(np.arange(0, W), 0) * np.ones_like(im)
    y = np.expand_dims(np.arange(0, H), 1) * np.ones_like(im)
    meanx = np.sum(x * im) / np.sum(im)
    meany = np.sum(y * im) / np.sum(im)

    return meany, meanx


def center_of_mass_torch(im):
    H, W = im.shape
    x = torch.arange(0, W).unsqueeze(0).cuda() * torch.ones_like(im)
    y = torch.arange(0, H).unsqueeze(1).cuda() * torch.ones_like(im)
    # x = np.expand_dims(np.arange(0,W),0) *np.ones_like(im)
    # y = np.expand_dims(np.arange(0,H),1)*np.ones_like(im)
    meanx = torch.sum(x * im) / torch.sum(im)
    meany = torch.sum(y * im) / torch.sum(im)
    # meanx = np.sum(x*im)/np.sum(im)
    # meany = np.sum(y*im)/np.sum(im)

    return meany, meanx
