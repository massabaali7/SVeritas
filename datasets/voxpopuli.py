from datasets import load_dataset
import torch

class VoxPopuli(torch.nn.Module):
    def __init__(
        self, 
        dataset_name: str = "macabdul9/voxpopuli_en_accented_SpeakerVerification", 
        split: str = 'test', 
        cache_dir: str = None
    ):
        super(VoxPopuli, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir

    def forward(self, audio=None):
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir
        )
        return dataset
