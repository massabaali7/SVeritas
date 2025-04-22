from huggingface_hub import hf_hub_download
import torch

class ECAPA2(torch.nn.Module):
    def __init__(self, model_location = 'cpu', cache_dir=None):
        super(ECAPA2, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
    def forward(self, audio):
        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=self.cache_dir)
        if self.model_location == 'cpu':
            ecapa2 = torch.jit.load(model_file, map_location=self.model_location)
        elif self.model_location == 'cuda':
            ecapa2 = torch.jit.load(model_file, map_location=self.model_location)
            ecapa2.half() # optional, but results in faster inference        
        embeddings = ecapa2(audio)
        # check later whether we need to squeeze unsqueeze etc.
        return embeddings
