import torch
import torchaudio
from .sox_effect import SoxEffect

class HighPassFilter(SoxEffect):
    def __init__(self, freq, sample_rate=16000) -> None:
        args = [str(freq)]
        super().__init__('highpass', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"HighPassFilter({self.args})"

class LowPassFilter(SoxEffect):
    def __init__(self, freq, sample_rate=16000) -> None:
        args = [str(freq)]
        super().__init__('lowpass', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"LowPassFilter({self.args})"
