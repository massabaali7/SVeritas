from datasets import load_dataset
import torch

class Ami(torch.nn.Module):
    def __init__(
        self,
        dataset_name: str = "edinburghcstr/ami",
        split: str = "test",
        partition: str = "ihm",
        cache_dir: str = None
    ):
        super(AMI, self).__init__()
        self.dataset_name = dataset_name
        self.partition    = partition
        self.split        = split
        self.cache_dir    = cache_dir

    def forward(self, audio):
        return load_dataset(
            self.dataset_name,
            self.partition,
            split=self.split,
            cache_dir=self.cache_dir
        )
