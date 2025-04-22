import torch
from torchaudio import functional as F
import os 
import torchaudio
import time
import numpy as np 

# Noise directories

class EnvNoise(torch.nn.Module):
    seeds = [4117371, 7124264, 1832224, 8042969, 4454604, 5347561, 7059465,
                3774329, 1412644, 1519183, 6969162, 7885564, 3707167, 5816443,
                9477077, 9822365, 7482569, 7792808, 9120101, 5467473]
    
    def __init__(self, snr, noise_dir=None) -> None:
        super().__init__()
        self.snr = snr
        self.noise_dir = noise_dir if noise_dir is not None else "./noise_data/MS-SNSD/noise_test"
        #self.noise_files = [x for x in os.listdir(self.noise_dir) if x.endswith('.wav')]
        self.noise_files = []
        for root, _, files in os.walk(self.noise_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.noise_files.append(os.path.join(root, f))
        if not self.noise_files:
            raise RuntimeError(f"No noise files found in {self.noise_dir}")

        # seed = self.seeds[int(snr % len(self.seeds))]
        
    def __repr__(self):
        return f"EnvNoise({self.snr} dB)"

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)        
        noise_file = self.noise_files[rng.choice(len(self.noise_files))]

        #noise_file = os.path.join(self.noise_dir, self.noise_files[rng.choice(len(self.noise_files))])
        noise_raw, sample_rate = torchaudio.load(noise_file)
        noise = noise_raw[..., :xlen]
        while noise.shape[-1] < xlen:
            noise = torch.cat([noise, noise], -1)
            noise = noise[..., :xlen]
        if x.ndim == 1:
            noise = noise[0].reshape(-1).to(x.device)
        elif x.ndim == 2:
            noise = noise[:x.shape[0], :x.shape[1]].to(x.device)
            if noise.shape[0] != x.shape[0]:
                noise = noise.repeat(x.shape[0] // noise.shape[0] + 1, 1)[:x.shape[0]]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        # noise = noise[0].reshape(-1).to(x.device)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        x_ = F.add_noise(x, noise, snr)
        return x_
    
class EnvNoiseESC50(EnvNoise):
    def __init__(self, snr) -> None:
        super().__init__(snr, noise_dir ="./noise_data/ESC-50-master/audio")
    def __repr__(self):
        return f"EnvNoiseESC50({self.snr} dB)"

class EnvNoiseMUSAN(EnvNoise):
    def __init__(self, snr) -> None:
        super().__init__(snr, "./noise_data/musan/noise")
        print(f"Loaded {len(self.noise_files)} noise files from {self.noise_dir}")
    def __repr__(self):
        return f"EnvNoiseMusan({self.snr} dB)"
class EnvNoiseWHAM(EnvNoise):
    def __init__(self, snr) -> None:
        super().__init__(snr, "./noise_data/wham_noise/tt")
        print(f"Loaded {len(self.noise_files)} noise files from {self.noise_dir}")
    def __repr__(self):
        return f"EnvNoiseWHAM({self.snr} dB)"
    
class EnvNoiseDeterministic(EnvNoise):
    def __init__(self, snr, noise_dir=None) -> None:
        super().__init__(snr, noise_dir)
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        self.noise_files = rng.choice(self.noise_files, 1)

class GaussianNoise(torch.nn.Module):
    def __init__(self, snr) -> None:
        super().__init__()
        self.snr = snr
    
    def __repr__(self):
        return f"GaussianNoise({self.snr} dB)"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        # if not isinstance(xlen, torch.Tensor):
        #     xlen = torch.LongTensor()
        rng = torch.Generator(x.device)
        rng = rng.manual_seed(rng.seed())
        d = torch.empty_like(x).normal_(0, 1, generator=rng)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr)

class UniformNoise(torch.nn.Module):
    def __init__(self, snr) -> None:
        super().__init__()
        self.snr = snr
    
    def __repr__(self):
        return f"UniformNoise({self.snr} dB)"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        d = torch.empty_like(x).uniform_(-1, 1)
        snr = torch.zeros(x.shape[:-1], device=x.device) + self.snr
        return F.add_noise(x, d, snr)
