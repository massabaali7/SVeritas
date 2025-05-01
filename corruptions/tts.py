import torch
import torch.nn
import sys
import torchaudio
import os
sys.path.append('/u/mbaali/CosyVoice/third_party/Matcha-TTS')
sys.path.append('/u/mbaali/CosyVoice/')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav


class CosyVoiceTTS(torch.nn.Module):
    def __init__(self, pretrained_name='pretrained_models/CosyVoice2-0.5B', tts_type="cross-lingual", tts_dir="", output_dir="", sr=16000) -> None:
        super().__init__()
        self.cosyvoice = CosyVoice2(tts_dir + pretrained_name, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
        self.sr = sr
        self.tts_type = tts_type
        self.output_dir = output_dir

    def forward(self, x, prompt, lang):
        prompt_speech_16k = load_wav(x, self.sr)
        file_name, ext = os.path.splitext(os.path.basename(x))
        assert self.cosyvoice.add_zero_shot_spk(f'{prompt}', prompt_speech_16k, 'my_zero_shot_spk') is True
        if self.tts_type == 'zero_shot':
            for i, j in enumerate(self.cosyvoice.inference_zero_shot(f'{prompt}', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
                torchaudio.save(self.output_dir + file_name + '_ZS_tts.wav', j['tts_speech'], self.cosyvoice.sample_rate)
        elif self.tts_type == 'cross-lingual':
            for i, j in enumerate(self.cosyvoice.inference_cross_lingual(f'<|{lang}|> {prompt}', prompt_speech_16k, stream=False)):
                torchaudio.save(self.output_dir + file_name + '_CL_tts.wav', j['tts_speech'], self.cosyvoice.sample_rate)