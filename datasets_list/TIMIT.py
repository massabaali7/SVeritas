import os
from torch.utils.data import Dataset, DataLoader
import torchaudio

class TIMIT(Dataset):
    def __init__(self, root, sample_rate, syn = False):
        super().__init__()
        self.root = root
        self.sample_rate = sample_rate

        self.data = []
        for dirpath, _, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(".wav"):
                    wav_path = os.path.join(dirpath, file)
                    if not syn:
                        txt_path = wav_path.replace(".WAV.wav", ".TXT")
                    else:
                        txt_path = wav_path.replace(".wav", ".TXT")
                    if os.path.exists(txt_path):
                        self.data.append((wav_path, txt_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_path, txt_path = self.data[idx]
        audio, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio)

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            text = lines[0].strip().split(' ', 1)[-1] if len(lines) == 1 else lines[1].strip()

        # speaker ID is the folder name before the file
        parts = wav_path.split(os.sep)
        sid = parts[-2]        # e.g. FAKS0
        rel_path = parts[-1] #os.path.relpath(wav_path, self.root)

        return {
            "filename": rel_path,
            "audio_tensor": audio,
            "txt": text,
            "sid": sid,
        }

# if __name__ == "__main__":
#     root = "/ocean/projects/cis220031p/shared/processed/TIMIT/timit_styletts/TEST/"
#     root = "/ocean/projects/cis220031p/abdulhan/timit_data/data/TEST/"
#     dataset = TIMIT(root, sample_rate=16000)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

#     for batch in dataloader:
#         print(batch['filename'])
#         print(batch['audio_tensor'])
#         print(batch['txt'])
#         print(batch['sid'])
#         break

