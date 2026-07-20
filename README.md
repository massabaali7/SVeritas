# SVeritas: Speaker Verification Benchmark Framework

A modular and extensible framework to benchmark **Speaker Verification** systems under clean and corrupted conditions with different models and augmentations.

---

## 📁 Directory Structure

```bash
speaker-verification-benchmark/
├── datasets/                    # Dataset loaders (LibriSpeech, VoxCeleb, etc.)
│   ├── build_data.py
│   ├── tedlium.py
│   ├── vctk.py
│   ├── ami.py
│   ├── librispeech.py
│   └── voxpopuli.py
├── models/                # Speaker embedding models
│   ├── ecapa2.py
│   ├── ecapa_tdnn.py
│   ├── redimnet.py
│   └── build_model.py
├── corruptions/           # Audio corruptions & augmentations
│   ├── echo.py
│   ├── gain.py
│   ├── rir.py
│   ├── resample.py
│   ├── filtering.py
│   ├── music.py
│   ├── speaker_guard_asv/ # Adversarial Attacks
│   ├── load_augmentations.py
│   └── build_aug.py
├── utils/                 # Utility scripts
│   ├── apply_rir.py
│   ├── compute_snrs.py
│   └── load_audio.py
├── evaluation/            # Metrics
│   ├── eer.py
│   └── minDCF.py
├── noise_data/            # External noises for augmentation
│   ├── ms-snsd/
│   ├── esc_dir/
│   ├── wham_noise/
│   ├── rir_noise/
│   └── musan/
├── output_dir/            # Output samples or logs
│   └── samples/
├── testing/               # Testing and simulation scripts
│   └── simulate_single_sample.py
├── config/                # YAML configs
│   └── default.yaml
├── main.py                # Entry point
├── requirements.txt       # Python dependencies
└── README.md
```

# Speaker Verification Benchmark Framework

A modular and extensible framework for evaluating speaker verification systems under a variety of corruptions and model architectures.

## 🔧 Structure

- `datasets/`: Load and configure datasets.
- `models/`: Pre-trained speaker embedding models.
- `corruptions/`: Augmentation and noise simulations.
- `utils/`: Common audio processing tools.
- `evaluation/`: Scoring functions (EER, minDCF).
- `testing/`: Sample simulation for debugging.
- `config/`: Configuration files.
- `main.py`: Main script for training/testing pipelines.

# 🧩 Installation
```bash
git clone https://github.com/massabaali7/SV_benchmark.git
cd speaker-verification-benchmark
pip install -r requirements.txt
```

# 📚 How to Add a Dataset
```python
from datasets import load_dataset
import torch

class MyDataset(torch.nn.Module):
    def __init__(self, ...):
        super(MyDataset, self).__init__()
        # Init logic here

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        # Return sample
```

Add it to `build_dataset.py`:
```python
from .mydataset import MyDataset

def build_dataset(dataset_name, config, device):
    if dataset_name == 'MyDataset':
        return MyDataset(...)
    ...
```

# 🧠 Adding a Model
Create a new file in `models/`, e.g., `my_model.py`
```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self, ...):
        super(MyModel, self).__init__()
        # Load or define model

    def forward(self, audio):
        # Return embeddings
        pass
```

Add it to `build_model.py`:
```python
from .my_model import MyModel

def build_model(model, config, device):
    if model == 'MyModel':
        return MyModel(...)
    ...
```

# 🎛️ Adding an Augmentation
Create a new file in `corruptions/`, e.g., `my_aug.py`
```python
import torch

class MyAug(torch.nn.Module):
    def __init__(self, ...):
        super(MyAug, self).__init__()
        # Setup

    def forward(self, x):
        # Return augmented audio
        pass
```

Register in `build_aug.py`:
```python
from .my_aug import MyAug

def build_augmentation(simulate, config):
    if simulate == 'MyAug':
        return MyAug(...)
    ...
```
# Run simulation on a single sample
```python testing/simulate_single_sample.py --simulate 'GuassianNoise' --waveform filename.wav ```

# Run embedding extractor
```python embedding_extraction_EARS.py --config config/default.yaml ```

# Run pairing by group (To compute EER on subgroup)
```python pairing_by_group.py --config config/default.yaml --group_by gender,age ```

## References
```https://github.com/ahmedshah1494/speech_robust_bench ```

## 📌 Citation

If you find this useful in your research, please cite us:

```bibtex
@misc{SVeritas,
  title = {SVeritas: Benchmark for Robust Speaker Verification under Diverse Conditions},
  author = {Massa Baali and Sarthak Bisht and Francisco Teixeira and Kateryna Shapovalenko and Shravanth Srinivas and Rita Singh and Bhiksha Raj},
  year={2025},
  eprint={2509.17091},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2509.17091},
  primaryClass={cs.CL}
}
```
