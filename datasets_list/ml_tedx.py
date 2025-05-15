import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset
import os 
import torchaudio 

class MLingual_TEDx(Dataset):
    def __init__(self, segment_file = "/storage1/data/Multilingual TEDx/ar-ar/data/test/txt/segments", root_directory = "/storage1/data/Multilingual TEDx/ar-ar/data/test/wav/", lang = 'ar'):
        
        self.segments = np.loadtxt(segment_file, str)
        self.root = root_directory
        self.lang = lang
        print("Dataset load {} speakers".format(len(set(os.listdir(self.root)))))

    def __len__(self):
        return len(self.segments)
    def __getitem__(self, index):
        filename = self.segments[index][0]
        spkID = self.segments[index][1]
        audio, sample_rate = torchaudio.load(self.root + spkID + ".flac")
        start_time = int(float(self.segments[index][2]) * sample_rate)
        end_time = int(float(self.segments[index][3]) * sample_rate)
        
        waveform_1 = audio[:,start_time:end_time]
        duration = waveform_1.shape[1] / sample_rate
        sample = {
        'waveform':  waveform_1,
        'fileID': filename,
        'spk_id': spkID,
        'duration': duration,
        'lang': self.lang
        }
        return sample

# if __name__ == "__main__":
#     dataset = MLingual_TEDx(segment_file = "/storage1/data/Multilingual TEDx/ar-ar/data/test/txt/segments", root_directory = "/storage1/data/Multilingual TEDx/ar-ar/data/test/wav/", lang = 'ar')
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

#     for batch in dataloader:
#         print(batch['fileID'], batch['duration'])
#         break



# AR: /storage1/data/Multilingual TEDx/ar-ar
# DE: /storage1/data/Multilingual TEDx/de-de
# EL: /storage1/data/Multilingual TEDx/el-el
# ES: /storage1/data/Multilingual TEDx/es-es
# FR: /storage1/data/Multilingual TEDx/fr-fr
# IT: /storage1/data/Multilingual TEDx/it-it
# PT: /storage1/data/Multilingual TEDx/pt-pt
# RU: /storage1/data/Multilingual TEDx/ru-ru