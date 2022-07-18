''' complex package for pytorch'''
import torch
import torch.nn.functional as F
def conj(x):
    dim = len(x.size())-1
    x_conj = torch.stack((x[..., 0], -x[..., 1]), dim=dim)
    return x_conj

def abs_square(x, keepdim = False):
    x_abs = (x[..., 0] ** 2 + x[..., 1] ** 2)
    if keepdim == True:
        dim = len(x.size()) - 1
        x_abs = torch.stack((x_abs, torch.zeros_like(x_abs)), dim=dim)
    return x_abs

def real(x):
    return x[..., 0]

def image(x):
    return x[..., 1]

def mul(x, y):
    dim = len(x.size())-1
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    mul = torch.stack((real, image), dim=dim)
    return mul

def div(x, y):
    dim = len(x.size())-1
    y_abs = y[..., 0]**2 + y[..., 1]**2
    real = (x[...,0] * y[...,0] + x[...,1] * y[...,1]) / y_abs
    image = (x[...,1] * y[...,0] - x[...,0] * y[...,1]) /y_abs
    div = torch.stack((real, image), dim=dim)
    return div

def fft(x):
    Fx = torch.rfft(x,signal_ndim=2,onesided=False)
    return Fx

def ifft(Fx):
    x = torch.irfft(Fx,signal_ndim=2,onesided=False)
    return x

def toAmptitudeAndPhase(x):
    assert x.shape[-1] == 2, 'input must be complex value'

    apt = torch.sqrt(x[...,0]**2 + x[...,1]**2)
    phase = torch.atan2(image(x),real(x))

    return torch.stack([apt,phase],-1)
def fromAmptitudeAndPhase(x):
    assert x.shape[-1] == 2, 'input must be amptitude and phase'
    return torch.stack([x[...,0]*torch.cos(x[...,1]),x[...,0]*torch.sin(x[...,1])],dim=-1)

def psf2otf_torch(psf, outSize):
    # input shape BCHW outputshape BCHW
    pb, pc, ph, pw = psf.shape
    oh, ow = outSize
    padh = oh - ph
    padw = ow - pw
    psf = F.pad(psf, (0, padw, 0, padh))
    psf = torch.roll(psf, shifts=(-(ph // 2), -(pw // 2)), dims=(-2, -1))
    otf = torch.rfft(psf, signal_ndim=2, onesided=False)
    return otf