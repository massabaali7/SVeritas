from .libirspeech import LibriSpeech
from .tedlium    import TEDLIUM
from .ami        import AMI
from .voxpopuli import VoxPopuli

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
            dataset_name="macabdul9/voxpopuli_en_accented_SpeakerVerification", 
            split='test', 
            cache_dir=config['data_cache_dir'])
    elif dataset_name == 'VCTK':
        dataset = VCTK(
            dataset_name="DynamicSuperb/SpeakerVerification_VCTK", 
            split='test', 
            cache_dir=config['data_cache_dir']   
    else: 
        raise NotImplementedError

    return dataset



