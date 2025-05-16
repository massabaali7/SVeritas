import argparse
import os
import tqdm
import torch
import torchaudio
import torchaudio.transforms as T
import yaml

from corruptions.build_augmentions import build_augmentation
from models.build_model import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the YAML config file.", default="./config/config.yaml")
    parser.add_argument("--simulate", type=str, help="Name of the augmentation")
    parser.add_argument("--model_name", type=str, help="Name of the embedding model")
    parser.add_argument("--output_dir", type=str, default='./output_dir/', help="output directory")
    parser.add_argument("--trial_file", type=str, help="Path to the trial file.")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    return args, config_args

def main():

    args, config = parse_args()
    
    # Build model
    device = torch.device(config["device"])
    model = build_model(args.model_name, config).to(device)

    # Build aug
    config["model"] = model
    augmentation = build_augmentation(args.simulate, config)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.trial_file, "r") as f:
        for i, line in tqdm.tqdm(enumerate(f)):
            path_1, path_2, label = line.strip().split(" ")

            label = int(label)

            if label == 0:
                tgt_label = 1
            else:
                tgt_label = 0

            wav, sr = torchaudio.load(path_1)
            wav_tgt, sr = torchaudio.load(path_2)
            augmented_wav = augmentation(wav, wav_tgt, tgt_label)

            print("Final score:", augmentation.classifier.predict(augmented_wav))
            print("Target label:", tgt_label)
            print("True label:", label)
            filename = os.path.splitext(os.path.basename(path_1))[0]
            output_name = (path_1.split("/")[-1].split(".wav")[0]) + "_" + str(augmentation) + ".wav"
            torchaudio.save(args.output_dir+output_name, torch.FloatTensor(augmented_wav).squeeze().unsqueeze(0), sample_rate=sr)
            print(f"Saved augmented audio to {output_name}")

if __name__ == "__main__":
    main()

