import os
import argparse
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Supported audio file extensions
AUDIO_EXTENSIONS = ['.wav', '.flac', '.mp3']

def is_audio_file(filename):
    return any(filename.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)

class CommonAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if is_audio_file(fname):
                    self.file_paths.append(os.path.join(root, fname))
        if not self.file_paths:
            raise RuntimeError(f"No audio files found in {root_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(path)
        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'file_path': path
        }


def dataloader(dataset_path):
    dataset = CommonAudioDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, sample in enumerate(loader):
        waveform = sample['waveform'][0]
        sr = sample['sample_rate'][0].item() if isinstance(sample['sample_rate'], torch.Tensor) else sample['sample_rate'][0]
        path = sample['file_path'][0]

        print(f"Sample {i+1}/{len(loader)}")
        print(f"Loaded file: {path}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample rate: {sr}")
        print("-" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Common audio dataloader checker")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the root directory of the audio dataset")
    args = parser.parse_args()
    dataloader(args.dataset_path)




# run the command
# python dataloader_unified.py --dataset_path <dataset_path>


