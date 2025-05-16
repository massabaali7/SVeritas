from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
from scipy.io import wavfile
import numpy as np

class wavLMBase(torch.nn.Module):
    def __init__(self, model_name = "microsoft/wavlm-base-sv", sr = 16000, model_location = 'cpu', needs_gradients=False):
        super(wavLMBase, self).__init__()
        self.needs_gradients = needs_gradients
        self.model_location = model_location
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(model_location)
        self.sr = sr
    def forward(self, wav):
        # Uncomment if you want to load on wav file
        # sr, wav = wavfile.read(audio)
        # # If stereo, take one channel
        # if wav.ndim > 1:
        #     wav = wav[:, 0]

        inputs = self.feature_extractor(
            wav.squeeze(),
            sampling_rate=self.sr,
            return_tensors="pt",
        )
        # Move input tensor to same device as model
        inputs = {key: val for key, val in inputs.items()}
        if self.needs_gradients:
            inputs["input_values"] = wav
            inputs["attention_mask"] = inputs["attention_mask"].to(wav.device)
            embeddings = self.model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        else:
            with torch.no_grad():
                embeddings = self.model(**inputs).embeddings
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        return embeddings
