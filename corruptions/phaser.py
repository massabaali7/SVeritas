import torch
import torchaudio
from .sox_effect import SoxEffect

class Phaser(SoxEffect):
    def __init__(self, decay, sample_rate=16000) -> None:
        args = [0.6, 0.8, 3, decay, 2, '-t']
        super().__init__('phaser', *args, sample_rate=sample_rate)
    
    def __repr__(self):
        return f"Phaser({self.args})"
