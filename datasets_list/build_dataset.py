from .librispeech import LibriSpeech
# from .tedlium     import TEDLIUM
# from .ami         import AMI
# from .voxpopuli   import VoxPopuli
# from .vctk        import VCTK
# from .voxtube     import VoxTube
from .ml_tedx import MLingual_TEDx
from .ami import AMI
from .EARS import EARS
from .TIMIT import TIMIT
def build_dataset(dataset_name, config, device, segment_file, audio_root, data_lang):
    if dataset_name == 'TIMIT':
        dataset = TIMIT(root=config['audio_root'], sample_rate=config['sample_rate'],syn = False)
    elif dataset_name == 'TIMIT_SYN':
        dataset = TIMIT(root=config['audio_root'], sample_rate=config['sample_rate'],syn = True)
    elif dataset_name == 'multilingualTEDx':
        #dataset = MLingual_TEDx(segment_file = config['segment_file'], root_directory = config['audio_root'], lang = config['data_lang'])
        dataset = MLingual_TEDx(segment_file = segment_file, root_directory = audio_root, lang = data_lang)
    elif dataset_name == 'AMI_nearfield':
        dataset = AMI(mic_type='ihm', split='test', cache_dir=config['data_cache_dir'])
    elif dataset_name == 'AMI_farfield':
        dataset = AMI(mic_type='sdm', split='test', cache_dir=config['data_cache_dir'])
    elif dataset_name == 'EARS':
        dataset = EARS(root=config['audio_root'], meta_path=config['meta_path'],sample_rate=16000)
        
    else: 
        raise NotImplementedError

    return dataset



