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
    parser.add_argument("--segment_file", type=str, help="Path to segment file")
    parser.add_argument("--audio_root", type=str, help="Root directory of audio files")
    parser.add_argument("--lang", type=str, default="ar", help="Language code")
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
    languages = {'el', 'es', 'fr','it','pt','ru'} #data_lang
    dataset = build_dataset(config["dataset_name"], config, device)
    model = build_model(config["model_name"], config).to(device)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Loaded {len(dataset)} utterances.")

    # Iterate and extract embeddings
    for batch in dataloader:
        
        utt_id = batch['fileID'][0]         # e.g., 'xzMcjMdrLIo_0273'
        speaker_id = batch['spk_id'][0]  # e.g., 'xzMcjMdrLIo'
        waveform = batch['waveform']    # Replace this for one file with waveform,sr = torchaudio.load("file.wav")

        embedding = model(waveform)  

        # # Save embeddings
        spk_dir = os.path.join(config['embeddings_dir'], config["dataset_name"], config['data_lang'], speaker_id, str(config["model_name"]))
        os.makedirs(spk_dir, exist_ok=True)

        filename = f"{utt_id}.npy"
        filepath = os.path.join(spk_dir, filename)
        np.save(filepath, embedding.cpu().numpy())
        print(f"Embeddings saved {filepath}")


if __name__ == "__main__":
    main()
