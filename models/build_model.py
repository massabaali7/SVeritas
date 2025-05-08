from .ecapa2 import ECAPA2
from .ecapa_tdnn import ECAPA
from .redimnet import Redimnet
from .titanet import Titanet
from .resnet34LM import resnet34_LM
from .wavLMBase import wavLMBase
from .mfa_conformer_lightning import MFA_Conformer
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
    elif model == 'MFA_Conformer':
        model = MFA_Conformer(model_location = config['device'], ckpt = "/storage1/avis_spk_id/baseline_framework/ckpt/epoch=17_cosine_eer=0.86.ckpt", lr=0.001 )
    elif model == 'MFA_Conformer_CAARMA':
        model = MFA_Conformer(model_location = config['device'], ckpt = "/storage1/avis_spk_id/baseline_framework/ckpt/epoch=97_cosine_eer=0.83.ckpt", lr=0.0045 )

    else: 
        raise NotImplementedError

    return model



