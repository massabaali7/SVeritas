import torch 

class Compose(torch.nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
    def __repr__(self):
        return f"Compose({self.transforms})"
    def forward(self, speech, *args, **kwargs):
        x = speech
        for t in self.transforms:
            x = t(x)
        return x

