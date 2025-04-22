import torch
from torchaudio import functional as F
import os 
import torchaudio
import time
import numpy as np 
import pandas as pd
import soundfile as sf 
class RIR(torch.nn.Module):
    seed = 9983137
    def __init__(self, sev, rir_dir=None, rir_t60_file='rir_t60.csv') -> None:
        super().__init__()
        assert sev <= 4
        self.rir_dir = rir_dir if rir_dir is not None else f'{os.environ["noise_dir"]}/RIRS_NOISES/simulated_rirs'
        # self.rir_files = [x for x in os.listdir(rir_dir) if x.endswith('.wav')]
        rir_files = []
        for root, dirs, files in os.walk(self.rir_dir):
            for name in files:
                if name.endswith('wav'):
                    rir_files.append(os.path.join(root, name))
        rir_t60_df = pd.read_csv(rir_t60_file)
        unique_t60 = rir_t60_df['RT60'].unique()
        t60_sevs = np.linspace(unique_t60.min(), unique_t60.max(), 5)
        print(t60_sevs, t60_sevs[sev])
        if sev == 0:
            filtered_rows = rir_t60_df[rir_t60_df['RT60'] <= t60_sevs[sev]]
        else:
            filtered_rows = rir_t60_df[(rir_t60_df['RT60'] <= t60_sevs[sev]) & (rir_t60_df['RT60'] > t60_sevs[sev-1])]
        self.rir_files = filtered_rows['filename'].values
        print(filtered_rows["RT60"].min(), filtered_rows["RT60"].max())
        print(f'using {len(self.rir_files)} rirs with average RT60={filtered_rows["RT60"].mean()}')
        # {x['filename']: x['snr'] for x in pd.read_csv(rir_snr_file).to_dict('records')}
        # self.rng = np.random.default_rng(self.seed)

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[rng.choice(len(self.rir_files))])
            rir_raw, sample_rate = torchaudio.load(rir_file)
            rir = rir_raw[:, int(sample_rate * .01) : ]
            return rir
        rir = get_random_rir()
        rir = rir / torch.norm(rir, p=2)
        rir = rir[0].reshape(-1).to(x.device)
        x_ = torchaudio.functional.fftconvolve(x, rir)
        return x_

class RealRIR(torch.nn.Module):
    seed = 9983137
    def __init__(self, sev, rir_dir=None, rir_t60_file='rir_snr.csv') -> None:
        super().__init__()
        assert sev <= 4
        self.rir_dir = rir_dir if rir_dir is not None else f'{os.environ["noise_dir"]}/RIRS_NOISES/real_rirs_isotropic_noises'
        # self.rir_files = [x for x in os.listdir(rir_dir) if x.endswith('.wav')]
        rir_files = []
        for root, dirs, files in os.walk(self.rir_dir):
            for name in files:
                if name.endswith('wav'):
                    rir_files.append(os.path.join(root, name))
        rir_metric_df = pd.read_csv(rir_t60_file)
        unique_srmr = rir_metric_df['srmr'].unique()
        srmr_sevs = np.linspace(unique_srmr.max(), unique_srmr.min(), 5)
        print(srmr_sevs, srmr_sevs[sev])
        if sev == 0:
            filtered_rows = rir_metric_df[(srmr_sevs[sev] <= rir_metric_df['srmr']) & (rir_metric_df['srmr'] <= srmr_sevs[sev-1])]
        else:
            filtered_rows = rir_metric_df[(srmr_sevs[sev] <= rir_metric_df['srmr']) & (rir_metric_df['srmr'] < srmr_sevs[sev-1])]
        self.rir_files = filtered_rows['filename'].values
        print(filtered_rows["srmr"].min(), filtered_rows["srmr"].max())
        print(f'using {len(self.rir_files)} rirs with average SRMR={filtered_rows["srmr"].mean()}')
        # {x['filename']: x['snr'] for x in pd.read_csv(rir_snr_file).to_dict('records')}
        # self.rng = np.random.default_rng(self.seed)
    def load_audio(self, path):
        audio, sr = sf.read(path)
        audio = torch.FloatTensor(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio[:, 0].unsqueeze(0)
        else:
            raise ValueError(f'Invalid audio shape {audio.shape}')
        audio = torch.FloatTensor(audio)
        return audio, sr
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[rng.choice(len(self.rir_files))])
            rir_raw, sample_rate = self.load_audio(rir_file)
            if sample_rate != 16000:
                rir_raw = torchaudio.transforms.Resample(sample_rate, 16000)(rir_raw)
            rir = rir_raw[:, int(sample_rate * .01) : ]
            return rir
        rir = get_random_rir()
        rir = rir / torch.norm(rir, p=2)
        rir = rir[0].reshape(-1).to(x.device)
        x_ = torchaudio.functional.fftconvolve(x, rir)
        return x_

class RIR_RoomSize(torch.nn.Module):
    def __init__(self, room_type, rir_dir=None) -> None:
        super().__init__()
        self.rir_dir = rir_dir if rir_dir is not None else f'{os.environ["noise_dir"]}/RIRS_NOISES/simulated_rirs'
        rir_files = []
        for root, dirs, files in os.walk(self.rir_dir):
            for name in files:
                if name.endswith('wav'):
                    if room_type in root:
                        rir_files.append(os.path.join(root, name))
        self.rir_files = rir_files
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        seed = time.time_ns()+os.getpid()
        rng = np.random.default_rng(seed)
        def get_random_rir():
            rir_file = os.path.join(self.rir_dir, self.rir_files[rng.choice(len(self.rir_files))])
            rir_raw, sample_rate = torchaudio.load(rir_file)
            rir = rir_raw[:, int(sample_rate * .01) : ]
            return rir
        rir = get_random_rir()
        rir = rir / torch.norm(rir, p=2)
        rir = rir[0].reshape(-1).to(x.device)
        x_ = torchaudio.functional.fftconvolve(x, rir)
        return x_
