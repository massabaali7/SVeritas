import numpy as np

def compute_snr(clean, noisy):
    snrs = []
    for _noisy in noisy:
        noise = _noisy - clean
        snr = 10 * np.log10((clean**2).sum() / (noise**2).sum())
        snrs.append(snr)
    snrs = np.array(snrs)
    return snrs
