import torch
import torchaudio

class Gain(torchaudio.transforms.Vol):
    def __init__(self, gain) -> None:
        super().__init__(gain)
    
    def __repr__(self):
        return f"Gain({self.gain})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        return super().forward(x)
