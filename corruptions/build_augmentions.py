from .noise import UniformNoise, GaussianNoise, EnvNoise, EnvNoiseESC50, EnvNoiseMUSAN, EnvNoiseWHAM
from .rir import RIR, RealRIR
from .resample import ResamplingNoise
from .gain import Gain
from .echo import Echo
from .filtering import LowPassFilter, HighPassFilter
from .music import MusicMUSAN, SpeechMUSAN
from .codec import codec_configs, CodecAug

from .speaker_guard_asv import FGSMAttack, PGDAttack, CW2Attack, CWInfAttack, BlackFakeBobAttack, BlackSirenAttack
#from .tts import CosyVoiceTTS

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
    elif simulate == 'music':
        aug = MusicMUSAN(config['NOISE_SNRS'])
    elif simulate == 'crosstalk':
        aug = SpeechMUSAN(config['NOISE_SNRS'])
    elif simulate == 'rir':
        aug = RIR(4) #[0, 1, 2, 3, 4])
    elif simulate == 'real_rir':
        aug = RealRIR(4) #[0, 1, 2, 3, 4])
    elif simulate == 'echo':
        aug = Echo(config['ECHO_DELAYS'],sample_rate=config['sample_rate'])
    elif simulate == 'tts':
        aug = CosyVoiceTTS(pretrained_name='pretrained_models/CosyVoice2-0.5B', tts_type = config['tts_type'], tts_dir=config['tts_dir'], output_dir = config['output_dir'], sr=config['sample_rate'])
    elif simulate == 'resample':
        aug = ResamplingNoise(config['RESAMPLING_FACTORS'],config['sample_rate'])
    elif simulate == 'gain':
        aug = Gain(config['GAIN_FACTORS'])
    elif simulate == 'lowpass':
        aug = LowPassFilter(config['LOWPASS_FREQS'],config['sample_rate'])
    elif simulate == 'highpass':
        aug = HighPassFilter(config['HIGHPASS_FREQS'],config['sample_rate'])
    elif simulate in codec_configs:
        aug = CodecAug(simulate, config['sample_rate'])
    elif simulate == 'fgsm_adv':
        aug = FGSMAttack(config["model"], config)
    elif simulate == 'pgd_adv':
        aug = PGDAttack(config["model"], config)
    elif simulate == 'cw2_adv':
        aug = CW2Attack(config["model"], config)
    elif simulate == 'cwinf_adv':
        aug = CWInfAttack(config["model"], config)
    elif simulate == 'fkb_adv':
        aug = BlackFakeBobAttack(config["model"], config)
    elif simulate == 'sa_adv':
        aug = BlackSirenAttack(config["model"], config)
    else:
        raise ValueError(f"Unknown augmentation type: {simulate}")

    return aug
