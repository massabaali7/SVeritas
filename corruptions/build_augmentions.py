from .noise import UniformNoise, GaussianNoise, EnvNoise, EnvNoiseESC50, EnvNoiseMUSAN, EnvNoiseWHAM
from .rir import RIR, RealRIR
from .resample import ResamplingNoise
from .gain import Gain
from .echo import Echo
from .filtering import LowPassFilter, HighPassFilter
from .music import MusicMUSAN, SpeechMUSAN
from .adv import FGSM, PGD, CWL2, CWLInf, BIM
from .codec import codec_configs, CodecAug
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
        aug = FGSM(config["model"], config["params"], config["targetd"])
    elif simulate == 'fgsm_adv':
        aug = BIM(config["model"], config["params"], config["targetd"])
    elif simulate == 'pgd_adv':
        aug = PGD(config["model"], config["params"], config["targetd"])
    elif simulate == 'cw_l2_adv':
        aug = CWL2(config["model"], config["params"], config["targetd"])
    elif simulate == 'cw_linf_adv':
        aug = CWLInf(config["model"], config["params"], config["targetd"])
    else:
        raise ValueError(f"Unknown augmentation type: {simulate}")

    return aug