import torch
import torchaudio
from .sox_effect import SoxEffect

class ChorusFilter(SoxEffect):
    def __init__(self, delay, sample_rate=16000) -> None:
        args = f"0.7 0.9 {delay} 0.4 0.25 2 -t {delay+10} 0.3 0.4 2 -s".split()
        super().__init__('chorus', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Chorus({self.args})"
