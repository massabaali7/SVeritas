from datasets import load_dataset
import torch

class VCTK(torch.nn.Module):
    def __init__(
        self,
        dataset_name: str = "DynamicSuperb/SpeakerVerification_VCTK",
        split: str = 'test',
        cache_dir: str = None
    ):
        super(VCTK, self).__init__()
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
