from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
from scipy.io import wavfile
import numpy as np

class wavLMBase(torch.nn.Module):
    def __init__(self, model_name = "microsoft/wavlm-base-sv", model_location = 'cpu'):
        super(wavLMBase, self).__init__()
        self.model_location = model_location
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMForXVector.from_pretrained(model_name).to(model_location)
    def forward(self, audio):
        sr, wav = wavfile.read(audio)
        if wav.dtype != 'float32':
            wav = wav / (2**15 if wav.dtype == 'int16' else np.max(np.abs(wav)))
            wav = wav.astype('float32')
        # If stereo, take one channel
        if wav.ndim > 1:
            wav = wav[:, 0]
        inputs = self.feature_extractor(wav, sampling_rate=sr, return_tensors="pt")
        embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(self.model_location)
        # check later whether we need to squeeze unsqueeze etc.
        return embeddings
