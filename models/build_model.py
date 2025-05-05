from .ecapa2 import ECAPA2
from .ecapa_tdnn import ECAPA
from .redimnet import Redimnet
from .titanet import Titanet
from .resnet34LM import resnet34_LM
from .wavLMBase import wavLMBase

def build_model(model, config):
    if model == 'ECAPA2':
        model = ECAPA2(model_location = config['device'], cache_dir=config['model_dir'])
    elif model == 'ECAPA':
        model = ECAPA(model_name_or_path="speechbrain/spkrec-ecapa-voxceleb", model_location = config['device'], cache_dir=config['model_dir'])
    elif model == 'Redimnet':
        model = Redimnet(model_name_or_path = 'IDRnD/ReDimNet', model_location = config['device'], model_name = 'b6', train_type = 'ft_lm', dataset ='vox2', cache_dir=config['model_dir'])
    elif model == 'TitaNet':
        model = Titanet(model_name = "nvidia/speakerverification_en_titanet_large", model_location = config['device'])
    elif model == 'ResNet34LM':
        model = resnet34_LM(model_name = "pyannote/wespeaker-voxceleb-resnet34-LM", model_location = config['device'])
    elif model == 'wavLMBase':
        model = wavLMBase(model_name = "microsoft/wavlm-base-sv", model_location = config['device'])
    elif model == 'wavLMBasePlus':
        model = wavLMBase(model_name = "microsoft/wavlm-base-plus-sv", model_location = config['device'])
    
    else: 
        raise NotImplementedError

    return model



