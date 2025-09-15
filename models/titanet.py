from huggingface_hub import hf_hub_download
import torch
import nemo.collections.asr as nemo_asr

class Titanet(torch.nn.Module):
    def __init__(self, model_name_or_path="nvidia/speakerverification_en_titanet_large", model_location = 'cpu', cache_dir=""):
        super(Titanet, self).__init__()
        self.cache_dir = cache_dir
        self.model_location = model_location
        self.model_name_or_path = model_name_or_path
        self.titanet = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name_or_path).to(model_location)

    def forward(self, audio):
        # embeddings = self.titanet(audio.squeeze(0).to(self.model_location)).squeeze()
        embeddings = self.titanet.get_embedding(audio).to(self.model_location)


        return embeddings
