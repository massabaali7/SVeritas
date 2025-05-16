from huggingface_hub import hf_hub_download
import torch
from speechbrain.inference.speaker import SpeakerRecognition

class ECAPA(torch.nn.Module):
    def __init__(self, model_name_or_path="speechbrain/spkrec-ecapa-voxceleb", model_location = 'cpu', cache_dir="pretrained_models/spkrec-ecapa-voxceleb", needs_gradients=False):
        super(ECAPA, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.model_name_or_path = model_name_or_path
        self.needs_gradients = needs_gradients
        self.ecapa = SpeakerRecognition.from_hparams(source=model_name_or_path, run_opts={"device":model_location}, savedir=cache_dir)

    def forward(self, audio):
        if not self.needs_gradients:
            with torch.no_grad():
                embeddings = self.ecapa.encode_batch(audio.squeeze(0).to(self.model_location)).squeeze()
        else:
            embeddings = self.ecapa.encode_batch(audio.squeeze(0).to(self.model_location)).squeeze()

        return embeddings
