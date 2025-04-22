from datasets import load_dataset
import torch

class LibriSpeech(torch.nn.Module):
    def __init__(self, dataset_name="librispeech_asr", split = 'test', partition = "clean", cache_dir=None):
        super(LibriSpeech, self).__init__()
        self.cache_dir = cache_dir
        self.split = split
        self.dataset_name = dataset_name
        self.partition = partition 
    def forward(self, audio):
        dataset = load_dataset(self.dataset_name,self.partition, split=self.split)
        return dataset

