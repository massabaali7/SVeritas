import torch
import torchaudio.transforms as audio_transforms

class Speed(torch.nn.Module):
    def __init__(self, factor, orig_freq=16000) -> None:
        super().__init__()
        self.factor = factor
        self.transform = audio_transforms.Speed(orig_freq, factor)
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        x_, new_lens =  self.transform.to(x.device)(x)
        return x_

