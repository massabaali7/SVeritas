import torch
from torch.utils.data import Dataset, DataLoader
import os

class SpeakerVerificationDataset(Dataset):
    def __init__(self, trials):
        """
        trials: list or np.array of shape [N, 3], each row is [label, path1, path2]
        root: root directory for file paths
        """
        self.trials = trials
    def __len__(self):
        return len(self.trials)
    def __getitem__(self, idx):
        label, file1, file2 = self.trials[idx]
        label = int(label)  # ensure it's integer 0 or 1
        return file1, file2, label
    def val_dataloader(self):
        trials = np.loadtxt(self.trials, dtype=str)
        print(f"number of trials: {len(trials)}")
        dataset = SpeakerVerificationDataset(trials)
        loader = DataLoader(dataset,
                            num_workers=10,
                            shuffle=True,  # shuffle for training-like randomness
                            batch_size=1)  # one trial per batch
        return loader
