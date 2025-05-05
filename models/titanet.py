import nemo.collections.asr as nemo_asr
import torch

class Titanet(torch.nn.Module):
    def __init__(self, model_name = "nvidia/speakerverification_en_titanet_large", model_location = 'cpu'):
        super(Titanet, self).__init__()
        self.model_location = model_location
        self.model_name = model_name
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name).to(model_location)
    def forward(self, audio):
        embeddings = self.speaker_model.get_embedding(audio)
        # check later whether we need to squeeze unsqueeze etc.
        return embeddings
