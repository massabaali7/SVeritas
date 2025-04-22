import torch
import torchaudio

class ResamplingNoise(torch.nn.Module):
    def __init__(self, factor, orig_freq=16000) -> None:
        super().__init__()
        print(f'factor={factor}', f'orig_freq={orig_freq}', f'new_freq={int(factor*orig_freq)}')
        self.ds = torchaudio.transforms.Resample(int(orig_freq*factor), orig_freq)
        self.us = torchaudio.transforms.Resample(orig_freq, int(orig_freq*factor))
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x_ = self.ds(self.us(x))
        return x_
