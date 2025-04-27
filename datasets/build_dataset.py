from .libirspeech import LibriSpeech

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
            cache_dir=config['data_cache_dir']
    else: 
        raise NotImplementedError

    return dataset



