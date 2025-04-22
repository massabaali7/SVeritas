import torch
import torchaudio

class SoxEffect(torch.nn.Module):
    def __init__(self, effect, *args, sample_rate=16000) -> None:
        super().__init__()
        self.effect = effect
        self.args = args
        self.sample_rate = sample_rate
    
    def __repr__(self):
        return f"SoxEffect({self.effect}, {self.args})"
    
    def forward(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate, [[self.effect] + [str(a) for a in self.args]])[0].squeeze(0)
