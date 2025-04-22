import soundfile as sf
import torch

def load_audio(path):
    audio, sr = sf.read(path)
    audio = torch.FloatTensor(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio[:, 0].unsqueeze(0)
    else:
        raise ValueError(f'Invalid audio shape {audio.shape}')
    return audio, sr
