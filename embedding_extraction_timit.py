import argparse
import yaml
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.build_model import build_model
from datasets_list.build_dataset import build_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings for utterances.")

    # Config
    parser.add_argument("--config", type=str, default="./config/default.yaml", help="Config file with parameters")

    # Override args
    parser.add_argument("--dataset_name", type=str, default="multilingualTEDx")
    parser.add_argument("--sample_rate", type=str, default="16000", help="sampling rate")
    parser.add_argument("--model_name", type=str, help="Name of the embedding model")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings", help="Directory to save embeddings")

    # first parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_args = yaml.safe_load(f)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args, config_args


def main():
    args, config = parse_args()

    # Setup
    device = torch.device(config["device"])
    list_models = ['Redimnet', 'ECAPA','MFA_Conformer', 'MFA_Conformer_CAARMA', 'wavLMBase', 'wavLMBasePlus'] 
    
    dataset = build_dataset(config["dataset_name"], config, device, "", "", "")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Loaded {len(dataset)} utterances.")
    for model_name in list_models:
        model = build_model(model_name, config).to(device)
        # Iterate and extract embeddings
        for batch in dataloader:
            
            if config["dataset_name"] == "TIMIT_SYN":
                utt_id = batch['filename'][0].split(".wav")[0]         # e.g., 'xzMcjMdrLIo_0273'
            else:
                utt_id = batch['filename'][0].split(".WAV")[0] 
            speaker_id = batch['sid'][0]  # e.g., 'xzMcjMdrLIo'
            waveform = batch['audio_tensor']    # Replace this for one file with waveform,sr = torchaudio.load("file.wav")
            # waveform = waveform.float() 
            # if model_name == 'Redimnet':
            #     if waveform.dim() == 2:  # If tensor is [C, L], reshape to [B, C, L]
            #         waveform = waveform.unsqueeze(0) 
            embedding = model(waveform)  
            # # Save embeddings
            spk_dir = os.path.join(config['embeddings_dir'], config["dataset_name"], speaker_id, model_name)
            os.makedirs(spk_dir, exist_ok=True)
            filename = f"{utt_id}.npy"
            filepath = os.path.join(spk_dir, filename)
            np.save(filepath, embedding.cpu().numpy())
            print(f"Embeddings saved {filepath}")


if __name__ == "__main__":
    main()
