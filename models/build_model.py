from .ecapa2 import ECAPA2
from .ecapa_tdnn import ECAPA
from .redimnet import Redimnet

def build_model(model, config, device):
    if model == 'ECAPA2':
        model = ECAPA2(model_location = config['device'], cache_dir=config['model_dir'])
    elif model == 'ECAPA':
        model = ECAPA(model_name_or_path="speechbrain/spkrec-ecapa-voxceleb", model_location = config['device'], cache_dir=config['model_dir'])
    elif model == 'Redimnet':
        model = Redimnet(model_name_or_path = 'IDRnD/ReDimNet', model_location = config['device'], model_name = 'b6', train_type = 'ft_lm', dataset ='vox2', cache_dir=config['model_dir'])
    else: 
        raise NotImplementedError

    return model



