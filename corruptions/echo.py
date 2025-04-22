import torch
import torchaudio

class Echo(torch.nn.Module):
    def __init__(self, delay, decay=0.3, sample_rate=16000) -> None:
        super().__init__()
        self.delay = delay
        self.decay = decay
        self.sample_rate = sample_rate
    
    def __repr__(self):
        return f"Echo({self.delay}, {self.decay})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, [['echo', '0.8', '0.9', str(self.delay), '0.3']])[0].squeeze(0)
