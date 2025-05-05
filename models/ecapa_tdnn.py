from huggingface_hub import hf_hub_download
import torch
from speechbrain.inference.speaker import SpeakerRecognition

class ECAPA(torch.nn.Module):
    def __init__(self, model_name_or_path="speechbrain/spkrec-ecapa-voxceleb", model_location = 'cpu', cache_dir="pretrained_models/spkrec-ecapa-voxceleb"):
        super(ECAPA, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.model_name_or_path = model_name_or_path
        self.ecapa = SpeakerRecognition.from_hparams(source=model_name_or_path, run_opts={"device":model_location}, savedir=cache_dir)
    def forward(self, audio):
        audio = torch.tensor(audio).unsqueeze(0).to(self.model_location)
        embeddings = self.ecapa(audio)
        return embeddings
