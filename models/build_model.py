# from .ecapa2 import ECAPA2
from .ecapa_tdnn import ECAPA
from .redimnet import Redimnet
from .wavLMBase import wavLMBase
from .mfa_conformer_lightning import MFA_Conformer

def build_model(model, config):
    if model == 'ECAPA2':
        model = ECAPA2(model_location = config['device'], cache_dir=config['model_dir'], needs_gradients=config["requires_grad"])
    elif model == 'ECAPA':
        model = ECAPA(model_name_or_path="speechbrain/spkrec-ecapa-voxceleb", model_location = config['device'], cache_dir=config['model_dir'], needs_gradients=config["requires_grad"])
    elif model == 'Redimnet':
        model = Redimnet(model_name_or_path = 'IDRnD/ReDimNet', model_location = config['device'], model_name = 'b6', train_type = 'ft_lm', dataset ='vox2', cache_dir=config['model_dir'], needs_gradients=config["requires_grad"])
    elif model == 'wavLMBase':
        model = wavLMBase(model_name = "microsoft/wavlm-base-sv",sr = config['sample_rate'],  model_location = config['device'], needs_gradients=config["requires_grad"])
    elif model == 'wavLMBasePlus':
        model = wavLMBase(model_name = "microsoft/wavlm-base-plus-sv",sr = config['sample_rate'], model_location = config['device'], needs_gradients=config["requires_grad"])
    elif model == 'MFA_Conformer':
        model = MFA_Conformer(model_location = config['device'], ckpt = "/ocean/projects/cis220031p/mbaali/sv_benchmark/epoch=17_cosine_eer=0.86.ckpt", lr=0.001, needs_gradients=config["requires_grad"])

    else: 
        raise NotImplementedError

    return model

