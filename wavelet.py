import numpy as np
import pywt
from compress import get_size
import os
import tempfile
# ACKNOWLEDGEMENT:
# Code taken from Reinhard Heckel:
# https://github.com/reinhardh/supplement_deep_decoder/blob/9031ea9e698a2d5f144dd6086d5a70ab07d7b4b3/include/wavelet.py#L13

# https://inst.eecs.berkeley.edu/~ee123/sp16/hw/hw9_Compressed_Sensing.html
#####


def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))


def unstack_coeffs(Wim):
    L1, L2 = np.hsplit(Wim, 2)
    LL, HL = np.vsplit(L1, 2)
    LH, HH = np.vsplit(L2, 2)
    return LL, [LH, HL, HH]


def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels - 1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0, c)
    coeffs.insert(0, LL)
    return coeffs


# TODO - explore alternate wavelets. 'db2'? 'bior'?
def dwt2(im):
    coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=4)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = coeffs2img(Wim, levels)
    return Wim


def idwt2(Wim):
    coeffs = img2coeffs(Wim, levels=4)
    return pywt.waverec2(coeffs, wavelet='db4', mode='per')


#####


def wavelet_threshold(image, compression_ratio):
    compression_ratio = max(1, compression_ratio // 3)
    if compression_ratio == 1:
        raise ValueError('compression_ratio must be >= 6')
    result = np.zeros(image.shape, dtype=np.float32)
    total_fs = 0.
    for c in range(3):
        im = image[:, :, c]
        im, fs = wavelet_threshold_single_channel(im, compression_ratio)
        total_fs += fs
        result[:, :, c] = im
    return result, total_fs


def wavelet_threshold_single_channel(image, compression_ratio):
    Wim = dwt2(image)
    m = np.sort(abs(Wim.ravel()))[::-1]
    if compression_ratio == 1:
        ndx = len(m) - 1
    else:
        ndx = int(len(m) / compression_ratio)
    thr = m[ndx]
    Wim_thr = Wim * (abs(Wim) > thr)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, 'wim_thr.npz')
        np.savez_compressed(fp, Wim_thr)
        fs = get_size(fp)

    im2 = idwt2(Wim_thr)
    return im2, fs
