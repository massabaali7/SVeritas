from .noise import UniformNoise, GaussianNoise, EnvNoise, EnvNoiseESC50, EnvNoiseMUSAN, EnvNoiseWHAM
from .speed import Speed
from .pitch import Pitch
from .rir import RIR, RealRIR
#from .voice_conversion import VoiceConversionVCTK, BarkTTSSpa
from .resample import ResamplingNoise
from .gain import Gain
from .echo import Echo
from .phaser import Phaser
from .tempo import Tempo
from .filtering import LowPassFilter, HighPassFilter
from .music import MusicMUSAN, SpeechMUSAN
from .tremolo import TremoloFilter
from .tone import TrebleFilter, BassFilter
from .chorus import ChorusFilter
from .tts import CosyVoiceTTS
#from .adv import UniversalAdversarialPerturbation
from .codec import codec_configs, CodecAug

def build_augmentation(simulate, config):
    if simulate == 'UniformNoise':
        aug = UniformNoise(config['NOISE_SNRS'])
    elif simulate == 'GuassianNoise':
        aug = GaussianNoise(config['NOISE_SNRS'])
    elif simulate == 'env_noise':
        aug = EnvNoise(config['NOISE_SNRS'])
    elif simulate == 'env_noise_esc50':
        aug = EnvNoiseESC50(config['NOISE_SNRS'])
    elif simulate == 'env_noise_musan':
        aug = EnvNoiseMUSAN(config['NOISE_SNRS'])
    elif simulate == 'env_noise_wham':
        aug = EnvNoiseWHAM(config['NOISE_SNRS'])
    elif simulate == 'speedup':
        aug = Speed(config['SPEEDUP_FACTORS'],config['sample_rate'])
    elif simulate == 'slowdown':
        aug = Speed(config['SLOWDOWN_FACTORS'],config['sample_rate'])
    elif simulate == 'pitch_up':
        aug = Pitch(config['PITCH_UP_STEPS'],config['sample_rate'])
    elif simulate == 'pitch_down':
        aug = Pitch(config['PITCH_DOWN_STEPS'],config['sample_rate'])
    elif simulate == 'rir':
        aug = RIR([0, 1, 2, 3, 4])
    elif simulate == 'real_rir':
        aug = RealRIR([0, 1, 2, 3, 4])
    elif simulate == 'tts':
        aug = CosyVoiceTTS(pretrained_name='pretrained_models/CosyVoice2-0.5B', tts_type = config['tts_type'], tts_dir=config['tts_dir'], output_dir = config['output_dir'], sr=config['sample_rate'])
    elif simulate == 'resample':
        aug = ResamplingNoise(config['RESAMPLING_FACTORS'],config['sample_rate'])
    elif simulate == 'gain':
        aug = Gain(config['GAIN_FACTORS'])
    elif simulate == 'echo':
        aug = Echo(config['ECHO_DELAYS'],sample_rate=config['sample_rate'])
    elif simulate == 'phaser':
        aug = Phaser(config['PHASER_DECAYS'],config['sample_rate'])
    elif simulate == 'tempo_up':
        aug = Tempo(config['SPEEDUP_FACTORS'],config['sample_rate'])
    elif simulate == 'tempo_down':
        aug = Tempo(config['SLOWDOWN_FACTORS'],config['sample_rate'])
    elif simulate == 'lowpass':
        aug = LowPassFilter(config['LOWPASS_FREQS'],config['sample_rate'])
    elif simulate == 'highpass':
        aug = HighPassFilter(config['HIGHPASS_FREQS'],config['sample_rate'])
    elif simulate == 'music':
        aug = MusicMUSAN(config['NOISE_SNRS'])
    elif simulate == 'crosstalk':
        aug = SpeechMUSAN(config['NOISE_SNRS'])
    elif simulate == 'tremolo':
        aug = TremoloFilter(config['TREMOLO_DEPTHS'],config['sample_rate'])
    elif simulate == 'treble':
        aug = TrebleFilter(config['TREBLE_GAIN'], config['sample_rate'])
    elif simulate == 'bass':
        aug = BassFilter(config['BASS_GAIN'])
    elif simulate == 'chorus':
        aug = ChorusFilter(config['CHORUS_DELAY'],config['sample_rate'])
    elif simulate in codec_configs:
        aug = CodecAug(simulate, config['sample_rate'])
    elif simulate == 'accent':
        aug = None
    elif simulate == 'itw_nf':
        aug = None
    elif simulate == 'itw_ff':
        aug = None
    elif simulate == 'itw_nf_ami':
        aug = None
    elif simulate == 'itw_ff_ami':
        aug = None
    elif simulate == 'universal_adv':
        aug = UniversalAdversarialPerturbation(ADV_SNRS)
    else:
        raise ValueError(f"Unknown augmentation type: {simulate}")

    return aug
