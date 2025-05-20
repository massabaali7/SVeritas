from huggingface_hub import hf_hub_download
import torch
from .mfa_conformer import conformer_cat
import pytorch_lightning as pl
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np 
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from os import pread
import torchaudio
import random
import numpy as np
from scipy.io import wavfile

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter', torch.FloatTensor(
                [-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        assert len(
            inputs.size()) == 2, 'The number of dimensions of inputs tensor must be 2!'
        # reflect padding to match lengths of in/out
        inputs = inputs.unsqueeze(1)
        inputs = F.pad(inputs, (1, 0), 'reflect')
        return F.conv1d(inputs, self.flipped_filter).squeeze(1)

class Mel_Spectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop=160, n_mels=80, coef=0.97, requires_grad=False):
        super(Mel_Spectrogram, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop = hop

        self.pre_emphasis = PreEmphasis(coef)
        mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        self.mel_basis = nn.Parameter(
            torch.FloatTensor(mel_basis), requires_grad=requires_grad)
        self.instance_norm = nn.InstanceNorm1d(num_features=n_mels)
        window = torch.hamming_window(self.win_length)
        self.window = nn.Parameter(
            torch.FloatTensor(window), requires_grad=False)

    def forward(self, x):
        x = self.pre_emphasis(x)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, win_length=self.win_length, return_complex=True)
        x = torch.abs(x)
        x += 1e-9
        x = torch.log(x)
        x = torch.matmul(self.mel_basis, x)
        x = self.instance_norm(x)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        return x

def load_audio(filename, second=3):
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if second <= 0:
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    return waveform.copy()

class CassavaPLModule(pl.LightningModule):
    def __init__(self, hparams, model):
        super(CassavaPLModule, self).__init__()
        self.model = model
        
    def forward(self, x):
        return self.model(x)       

class MFA_Conformer(torch.nn.Module):
    def __init__(self, model_location = 'cuda', ckpt = "/storage1/avis_spk_id/baseline_framework/ckpt/epoch=97_cosine_eer=0.83.ckpt", lr=0.0045, needs_gradients=False):
        super(MFA_Conformer, self).__init__()
        self.model_location = model_location
        self.needs_gradients = needs_gradients
        model_init = conformer_cat(n_mels=80, num_blocks=6, output_size=256, 
        embedding_dim=192, input_layer="conv2d2", pos_enc_layer_type="rel_pos").to(model_location)
        self.features = Mel_Spectrogram()
        self.ckpt = ckpt
        self.model = CassavaPLModule.load_from_checkpoint(ckpt, hparams={'lr':lr, 'batch_size':1}, model=model_init, strict=False)
        self.model.eval()
        self.model.cuda()
        self.model.freeze()  #Will get a CUDA memory error without this

    def forward(self, audio):
        # Uncomment if you're loading the wav file  
        #speech_sample, sampling_rate = torchaudio.load(speech_wavefile) 
        #speech_sample =torch.FloatTensor(load_audio(speech_wavefile,second=-1)) 
        # speech_sample = speech_sample.unsqueeze(0)
        if len(audio.size()) > 2:
            audio = audio.squeeze(0).to(self.model_location)
        else:
            audio = audio.to(self.model_location)

        if not self.needs_gradients:
            feats = self.features(audio).to(self.model_location)
            # Get embedding
            with torch.no_grad():
                embeddings = self.model(feats)
        else:
            feats = self.features(audio).to(self.model_location)
            embeddings = self.model(feats)

        return embeddings
