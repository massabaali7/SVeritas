from datasets import load_dataset
import torch

class VoxTube(torch.nn.Module):
    def __init__(
        self,
        dataset_name: str = "voice-is-cool/voxtube",
        split: str = 'train',
        cache_dir: str = None
    ):
        super(VoxTube, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir

    def forward(self, audio):
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir
        )
        return dataset
