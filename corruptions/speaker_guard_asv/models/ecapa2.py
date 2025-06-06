from huggingface_hub import hf_hub_download
import torch

class ECAPA2(torch.nn.Module):
    def __init__(self, model_location = 'cpu', cache_dir=None, needs_gradients=False):
        super(ECAPA2, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.needs_gradients  = needs_gradients
        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=cache_dir)
        if self.model_location == 'cpu':
            self.ecapa2 = torch.jit.load(model_file, map_location=model_location)
        elif self.model_location == 'cuda':
            self.ecapa2 = torch.jit.load(model_file, map_location=model_location)
            self.ecapa2.half() # optional, but results in faster inference

    def forward(self, audio):
        audio = audio.squeeze(0)
        if not self.needs_gradients:
            with torch.no_grad():
                embeddings = self.ecapa2(audio.to(self.model_location))
        else:
            embeddings = self.ecapa2(audio.to(self.model_location))
        # check later whether we need to squeeze unsqueeze etc.
        return embeddings
