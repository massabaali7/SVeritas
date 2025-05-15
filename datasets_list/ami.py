import os
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchaudio
from collections import defaultdict

class AMI(Dataset):
    def __init__(
        self,
        mic_type='ihm',
        split='test',
        cache_dir='/ocean/projects/cis220031p/mbaali/new_cache_dir/',
        segment_file_path='segment',
        min_duration=3.0,
        max_duration=10.0
    ):
        self.mic_type = mic_type
        self.split = split
        self.cache_dir = cache_dir
        self.segment_file_path = segment_file_path+"_"+mic_type+".txt"
        self.min_duration = min_duration
        self.max_duration = max_duration

        if os.path.exists(self.segment_file_path):
            print(f"Loading preselected audio IDs from: {self.segment_file_path}")
            with open(self.segment_file_path, 'r') as f:
                self.file_ids = set(line.strip() for line in f)
        else:
            print("Segment file not found. Generating new segment file...")
            self.file_ids = self._generate_segment_file()

        # Load full dataset and filter
        full_dataset = load_dataset(
            'edinburghcstr/ami',
            self.mic_type,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        self.dataset = [sample for sample in full_dataset if sample['audio_id'] in self.file_ids]
        print(f"[{self.mic_type.upper()}] Loaded {len(self.dataset)} samples from {len(self.file_ids)} file IDs.")

    def _generate_segment_file(self):
        full_dataset = load_dataset(
            'edinburghcstr/ami',
            self.mic_type,
            split=self.split,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )

        speaker_to_files = defaultdict(list)
        selected_file_ids = []

        for sample in full_dataset:
            duration = sample['end_time'] - sample['begin_time']
            speaker = sample['speaker_id']
            if self.min_duration <= duration <= self.max_duration:
                if len(speaker_to_files[speaker]) < 20:
                    speaker_to_files[speaker].append(sample['audio_id'])
                    selected_file_ids.append(sample['audio_id'])

        with open(self.segment_file_path, 'w') as f:
            for file_id in selected_file_ids:
                f.write(file_id + '\n')

        print(f"Generated segment file: {self.segment_file_path} with {len(selected_file_ids)} utterances.")
        return set(selected_file_ids)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return {
            'audio': sample['audio']['array'],
            'sample_rate': sample['audio']['sampling_rate'],
            'file': sample['audio_id'],
            'text': sample['text'],
            'speaker': sample['speaker_id'],
            'mic_type': self.mic_type
        }

# Example usage:
# if __name__ == "__main__":
#     for final_name, subset in [('nearfield', 'ihm'), ('farfield', 'sdm')]:
#         dataset = AMI(mic_type=subset)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

#     for batch in dataloader:
#         print(batch)
#         print(batch['file'])
#         break

# meeting_id', 'audio_id', 'text', 'audio', 'begin_time', 'end_time', 'microphone_id', 'speaker_id
# save i mean the IDs that u chose! 