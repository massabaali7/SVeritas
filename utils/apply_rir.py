import torch
import torchaudio 
from joblib import Parallel, delayed
from tqdm import trange, tqdm

def apply_rir(args):
    x, (rir_raw, sample_rate) = args
    rir = rir_raw[:, int(sample_rate * .01) : ]
    rir = rir / torch.norm(rir, p=2)
    rir = rir[0].reshape(-1).to(x.device)
    x_ = torchaudio.functional.fftconvolve(x, rir)
    x_ = x_[:x.shape[0]]
    return x_

def apply_rirs(x, rirs):
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)
    # x_out = Pool(cpu_count()).map(apply_rir, [(i, x, rir) for i,rir in enumerate(rirs)])
    x_out = Parallel(n_jobs=5)(delayed(apply_rir)((x, rir)) for rir in tqdm(rirs))
    # pad x_out to have same length as x
    x_out = [torch.nn.functional.pad(x_, (0, x.shape[0] - x_.shape[0])) for x_ in x_out]
    x_out = torch.stack(x_out, 0)
    return x_out