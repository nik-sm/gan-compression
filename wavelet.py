import numpy as np
import pywt
# ACKNOWLEDGEMENT:
# Code taken from Reinhard Heckel:
# https://github.com/reinhardh/supplement_deep_decoder/blob/9031ea9e698a2d5f144dd6086d5a70ab07d7b4b3/include/wavelet.py#L13


# https://inst.eecs.berkeley.edu/~ee123/sp16/hw/hw9_Compressed_Sensing.html
#####

def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def unstack_coeffs(Wim):
        L1, L2  = np.hsplit(Wim, 2) 
        LL, HL = np.vsplit(L1, 2)
        LH, HH = np.vsplit(L2, 2)
        return LL, [LH, HL, HH]

    
def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels-1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0,c)
    coeffs.insert(0, LL)
    return coeffs
    
    
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
    result = np.zeros(image.shape)
    for c in range(3):
        im = image[:,:,c]
        im = wavelet_threshold_single_channel(im, compression_ratio)
        result[:,:,c] = im
    return result

def wavelet_threshold_single_channel(image, compression_ratio):
    Wim = dwt2(image)
    m = np.sort(abs(Wim.ravel()))[::-1]
    ndx = int(len(m) / compression_ratio)
    thr = m[ndx]
    Wim_thr = Wim * (abs(Wim) > thr)
    im2 = idwt2(Wim_thr)
    return im2


def wavelet_threshold_prev(image, wavelet='db1', ncoeff=None, mode='hard'):
    """
    image - 
    wavelet - wavelet family to use
    ncoeff - number of coefficients to keep per color channel
    mode - soft vs hard thresholding

    return: reconstructed image
    """
    wavelet = pywt.Wavelet(wavelet)

    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = [slice(s) for s in image.shape]

    ## Determine the number of wavelet decomposition levels
    # Determine the maximum number of possible levels for image
    dlen = wavelet.dec_len
    wavelet_levels = np.min(
        [pywt.dwt_max_level(s, dlen) for s in image.shape])

    # Skip coarsest wavelet scales (see Notes in docstring).
    wavelet_levels = max(wavelet_levels - 3, 1)
    print(f'wavelet_levels : {wavelet_levels}')

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    a = []
    for level in dcoeffs:
        print(f'dcoeffs level: {level.keys()}')
        for key in level:
            a.append(np.ndarray.flatten(level[key]))
    a = np.concatenate(a)
    a = np.sort( np.abs(a) )    

    sh = coeffs[0].shape
    basecoeffs = sh[0]*sh[1]
    print(f'basecoeffs: {basecoeffs}')
    print(f'ncoeff: {ncoeff}')
    threshold = a[- (ncoeff - basecoeffs)]
    print(f'threshold: {threshold}')
    print(len(a))
    print(a)
    exit()
    # [ x, y, z, .........  a ]
    
    # A single threshold for all coefficient arrays
    denoised_detail = [{key: pywt.threshold(level[key],value=threshold,
                                mode=mode) for key in level} for level in dcoeffs]
   
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]
