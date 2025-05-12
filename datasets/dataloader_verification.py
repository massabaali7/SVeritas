import torchaudio
from torch.utils.data import Dataset

class VerificationDataset(Dataset):
    def __init__(self, file_path, delimiter=' '):
        """
        Args:
            file_path (str): Path to a csv file with three columns:
                             audio_file_1, audio_file_2, label (0 or 1)
            delimiter (str): Delimiter used in the file (default is space)
        """
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) != 3:
                    raise ValueError(f"Expected 3 columns per row, but got: {parts}")
                path1, path2, label_str = parts
                label = int(label_str)
                if label not in (0, 1):
                    raise ValueError(f"Label must be 0 or 1, got: {label}")
                self.samples.append((path1.strip(), path2.strip(), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_1, path_2, label = self.samples[idx]
        wav_1, sr_1 = torchaudio.load(path1)
        wav_2, sr_2 = torchaudio.load(path2)

		assert sr_1 == sr_2

		wav_1 = wav_1.squeeze().unsqueeze(dim=0)
		wav_2 = wav_2.squeeze().unsqueeze(dim=0)

        return {
            'waveform': torch.cat((wav_1, wav_2), dim=0),
            'sample_rate': sr_1,
            'file_path': path1,
            'label': label
        }

