import os
from math import floor
from scipy.io import wavfile
from scipy import signal
import numpy as np
import torch


def torch2np(*tensors):
    # Converts torch tensors to numpy
    tensors = list(tensors)
    for i in range(len(tensors)):
        if type(tensors[i]).__module__ == 'torch':
            tensors[i] = tensors[i].detach().cpu().numpy()
        if type(tensors[i]).__module__ == 'numpy':
            if len(tensors[i].shape) == 0:
                tensors[i] = tensors[i].item()
    
    return tuple(tensors) if len(tensors) > 1 else tensors[0]

class MovingAverages:
    # Keeps moving averages
    def __init__(self):
        self.data = {}
        self.counts = {}

    def __call__(self, data_new):
        for key in data_new.keys():
            # TODO
            data_new[key] = torch2np(data_new[key])

            if key not in self.counts.keys():
                self.counts[key] = 1
                self.data[key] = data_new[key] * 1.0
            else:
                self.counts[key] += 1
                self.data[key] = (self.data[key] * (self.counts[key] - 1) + data_new[key]) / self.counts[key]
    
    def get(self):
        return self.data.copy()

    def reset(self):
        for key in self.data.keys():
            self.data[key] = 0.0
            self.counts[key] = 0.0


def read_audio(path, make_stereo=True):
    sr, audio = wavfile.read(path)
    audio = audio.T
    if np.issubdtype(audio.dtype, np.int16):
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) == 1:    # if mono
        audio = np.expand_dims(audio, axis=0)
        if make_stereo:
            audio = np.repeat(audio, 2, axis=0)
    return audio, sr


def mse(x, ref):
    # Mean-squared error
    mse_val = ((x - ref)**2).mean()
    return mse_val


def snr(x, ref):
    # Signal-to-noise ratio
    ref_pow = (ref**2).mean().mean() + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean().mean() + np.finfo('float32').eps
    snr_val = 10 * np.log10(ref_pow / dif_pow)
    return snr_val


def snr_torch(x, ref):
    # Signal-to-noise ratio for torch tensors
    ref_pow = (ref**2).mean(-1).mean(-1) + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean(-1).mean(-1) + np.finfo('float32').eps
    snr_val = 10 * torch.log10(ref_pow / dif_pow)
    return snr_val.mean().item()


def weights_init_normal(m):
    # Initializes weights using normal distribution
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Flatten(torch.nn.Module):
    # Flattens 2D tensors to 1D
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PixelUpscale(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales activations
    """
    def __init__(self, upscale_factor, odd_output=False):
        super(PixelUpscale, self).__init__()
        self.upscale_factor = upscale_factor
        self.odd_output = odd_output

    def forward(self, x):

        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        if self.odd_output:
            x = x[:, :, :-1]

        return x


def predict_size(input_size, kernel_size, padding=0, stride=1, dilation=1):
    # Predicts output size of convolutional layer
    return int(floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


def makedir(path):
    os.makedirs(path, exist_ok=True)
    return path


def pad_str_zeros(x, n):
    # Pads string with zeros to have length n
    x = str(x)
    for _ in range(len(x), n):
        x = '0' + x
    return x


def conv1d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1, pad_type='zero'):
    # Convolution which does not change input size
    if pad_type == 'zero':
        return torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) // 2, bias=bias, dilation=dilation)
    elif pad_type == 'reflection':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                         
        )
    elif pad_type == 'replication':
        return torch.nn.Sequential(
            torch.nn.ReplicationPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                         
        )
    else:
        raise ValueError('Unknown padding type')


def conv2d_same(in_channels, out_channels, kernel_size, bias=True, dilation=1, pad_type='zero'):
    # Convolution which does not change input size
    if len(kernel_size) == 1:
        kernel_size = (kernel_size, kernel_size)
    pad_size = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

    if pad_type == 'zero':
        return torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=pad_size, bias=bias, dilation=dilation)
    elif pad_type == 'reflection':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad2d(pad_size),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                         
        )
    elif pad_type == 'replication':
        return torch.nn.Sequential(
            torch.nn.ReplicationPad2d(pad_size),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation)                       
        )
    else:
        raise ValueError('Unknown padding type')


def conv1d_halve(in_channels, out_channels, kernel_size, bias=True, dilation=1, pad_type='zero'):
    # Convolution which halves the input size
    if pad_type == 'zero':
        return torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=2,
                               padding=(kernel_size - 1) // 2, bias=bias, dilation=dilation)
    elif pad_type == 'reflection':
        return torch.nn.Sequential(
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation, stride=2)                         
        )
    elif pad_type == 'replication':
        return torch.nn.Sequential(
            torch.nn.ReplicationPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias, dilation=dilation, stride=2)                         
        )
    else:
        raise ValueError('Unknown padding type')


def lowpass(sig, cutoff, filter_=('cheby1', 8), sr=44100):
    """Lowpasses input signal based on a cutoff frequency
    
    Arguments:
        sig {numpy 1d array} -- input signal
        cutoff {int} -- cutoff frequency
    
    Keyword Arguments:
        sr {int} -- sampling rate of the input signal (default: {44100})
        filter_type {str} -- type of filter, only butter and cheby1 are implemented (default: {'butter'})
    
    Returns:
        numpy 1d array -- lowpassed signal
    """
    nyq = sr / 2
    cutoff /= nyq

    if filter_[0] == 'butter':
        B, A = signal.butter(filter_[1], cutoff)
    elif filter_[0] == 'cheby1':
        B, A = signal.cheby1(filter_[1], 0.05, cutoff)
    elif filter_[0] == 'bessel':
        B, A = signal.bessel(filter_[1], cutoff, norm='mag')
    elif filter_[0] == 'ellip':
        B, A = signal.ellip(filter_[1], 0.05, 20, cutoff)

    sig_lp = signal.filtfilt(B, A, sig)
    return sig_lp.astype(np.float32)
