from .libirspeech import LibriSpeech
from .tedlium     import TEDLIUM
from .ami         import AMI
from .voxpopuli   import VoxPopuli
from .vctk        import VCTK
from .voxtube     import VoxTube

def build_dataset(dataset_name, config, device):
    if dataset_name == 'LibriSpeech': 
        dataset = LibriSpeech(
            "librispeech_asr", 
            split = 'test', 
            partition = "clean", 
            cache_dir=config['data_cache_dir'])
    elif dataset_name == 'TEDLIUM':
        dataset = TEDLIUM(
            "LIUM/tedlium",       
            split='test',          
            partition="release3", 
            cache_dir=config['data_cache_dir'])
    elif dataset_name == 'AMI':
        dataset = AMI(
            "edinburghcstr/ami",  
            split='test',        
            partition="ihm",   #ihm or sdm     
            cache_dir=config['data_cache_dir'])
    elif dataset_name == 'VoxPopuli':
        dataset = VoxPopuli(
            "macabdul9/voxpopuli_en_accented_SpeakerVerification", 
            split='test', 
            cache_dir=config['data_cache_dir'])
    elif dataset_name == 'VCTK':
        dataset = VCTK(
            "DynamicSuperb/SpeakerVerification_VCTK", 
            split='test', 
            cache_dir=config['data_cache_dir'])   
    elif dataset_name == 'VoxTube':
        dataset = VoxTube(
            "voice-is-cool/voxtube",
            split='train', 
            cache_dir=config['data_cache_dir'])
    else: 
        raise NotImplementedError

    return dataset



