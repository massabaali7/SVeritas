import torch
import torchaudio
from .sox_effect import SoxEffect

class TrebleFilter(SoxEffect):
    def __init__(self, gain, sample_rate=16000) -> None:
        args = [str(gain)]
        super().__init__('treble', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Treble({self.args})"

class BassFilter(SoxEffect):
    def __init__(self, gain, sample_rate=16000) -> None:
        args = [str(gain)]
        super().__init__('bass', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Treble({self.args})"
