import torch
import torchaudio.transforms as audio_transforms

class Pitch(torch.nn.Module):
    def __init__(self, shift, orig_freq=16000) -> None:
        super().__init__()
        self.shift = shift
        self.transform = audio_transforms.PitchShift(orig_freq, shift)
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        xlen = x.shape[-1]
        x_ =  self.transform.to(x.device)(x)
        return x_
