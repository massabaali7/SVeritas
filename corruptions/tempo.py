import torch
import torchaudio
from .sox_effect import SoxEffect

class Tempo(SoxEffect):
    def __init__(self, factor, sample_rate=16000) -> None:
        args = [factor, 30]
        super().__init__('tempo', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Tempo({self.args})"
