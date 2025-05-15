import os
from glob import glob
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import pickle
from copy import deepcopy
from glob import glob
import random
from sklearn.model_selection import train_test_split
import json
import os
import numpy as np 
import librosa
import torch
import soundfile as sf
import pandas as pd
import ast 
def safe_parse_metadict(meta_str):
    try:
        return ast.literal_eval(meta_str)
    except Exception as e:
        print(f"Failed to parse metadict: {e}")
        return {}

class EARS(Dataset):
    """
    EARS dataset for 10sec or less that 10sec segments.
    Returns:
        audio: torch.Tensor in (1,16000) or (1, <16000), audio waveform
        sid: str (p103), speaker id
        metadict: dict, metadata
    """
    def __init__(self, root, meta_path,sample_rate):
        super().__init__()
        self.root = root
 
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.sample_rate = sample_rate
        self.new_data = [{"file_name": fname} for fname in self.meta.keys()]
    
    def __len__(self):
        return len(self.new_data)

    def __getitem__(self, idx):
        filename = self.new_data[idx]["file_name"]
        sid      = filename.split("/")[0]
        audio_path = os.path.join(self.root, filename)
        audio, sample_rate = torchaudio.load(audio_path)
        duration = audio.shape[1] / sample_rate

        if sample_rate != self.sample_rate:
            audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)

        meta_dict = safe_parse_metadict(self.meta[filename]["metadict"])
        age = meta_dict.get("age", "unknown")
        ethnicity = meta_dict.get("ethnicity", "unknown")
        gender = meta_dict.get("gender", "unknown")
        lang = meta_dict.get("native language", "unknown")

        res_dict = {
            "filename":filename,
            "audio_tensor": audio,
            "sid": sid,
            "age": age,
            "ethnicity":ethnicity ,
            "gender": gender,
            "native language": lang ,
            "duration": duration,
        }
        return res_dict