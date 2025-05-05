from pyannote.audio import Model
from pyannote.audio import Inference
import torch

class resnet34_LM(torch.nn.Module):
    def __init__(self, model_name = "pyannote/wespeaker-voxceleb-resnet34-LM", model_location = 'cpu'):
        super(resnet34_LM, self).__init__()
        self.model_location = model_location
        self.model_name = model_name
        self.speaker_model = Model.from_pretrained(model_name).to(model_location)
    def forward(self, audio):
        inference = Inference(self.speaker_model, window="whole")
        embeddings = inference(audio)
        # check later whether we need to squeeze unsqueeze etc.
        return embeddings
