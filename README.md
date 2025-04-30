# Speaker Verification Benchmark Framework

A modular and extensible framework to benchmark **Speaker Verification** systems under clean and corrupted conditions with different models and augmentations.

---

## 📁 Directory Structure

```bash
speaker-verification-benchmark/
├── datasets/              # Dataset loaders (LibriSpeech, VoxCeleb, etc.)
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
│   ├── pitch.py
│   ├── rir.py
│   ├── resample.py
│   ├── filtering.py
│   ├── speed.py
│   ├── tempo.py
│   ├── chorus.py
│   ├── tremolo.py
│   ├── tone.py
│   ├── phaser.py
│   ├── music.py
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

    def forward(self, audio):
        # Return dataset/audio
        pass
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
```python testing/simulate_single_sample.py --simulate 'echo' --waveform filename.wav ```
