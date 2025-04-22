from .noise import EnvNoise
import os 

class MusicMUSAN(EnvNoise):
     def __init__(self, snr) -> None:
        super().__init__(snr, noise_dir='./noise_data/musan/music')
     def __repr__(self):
        return f"MusicMusan({self.snr} dB)"

        # self.noise_files = []
        # for root, dirs, files in os.walk(self.noise_dir):
        #     for name in files:
        #         if name.endswith('wav'):
        #             self.noise_files.append(os.path.join(root, name))

class SpeechMUSAN(EnvNoise):
     def __init__(self, snr) -> None:
        super().__init__(snr, noise_dir='./noise_data/musan/speech')
     def __repr__(self):
        return f"SpeechMusan({self.snr} dB)"
