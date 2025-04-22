import torch
import torchaudio
from .sox_effect import SoxEffect

class TremoloFilter(SoxEffect):
    def __init__(self, depth, sample_rate=16000) -> None:
        args = ['20', f'{depth}']
        super().__init__('tremolo', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Tremolo({self.args})"
