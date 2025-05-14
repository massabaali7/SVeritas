import argparse
import yaml
import sys
import os 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from corruptions.build_augmentions import build_augmentation
from utils.load_audio import load_audio
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser(description="Simulating a speech sample.")
    
    # config file
    parser.add_argument("--config", type=str, default='./config/adv_corruption.yaml', help="config file used to specify parameters")

    # data 
    parser.add_argument("--simulate", type=str, default='FGSM', help="Simulation scenario")
    parser.add_argument("--waveform", type=str, default='sample_1.wav', help="input filename")
    parser.add_argument("--waveform_tgt", type=str, default='sample_2.wav', help="input filename 2")
    parser.add_argument("--output_dir", type=str, default='./output_dir/', help="output directory")

    
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
    
    # parse arguments
    args, config = parse_args()
    augment = build_augmentation(args.simulate, config)

    speech_array, sample_rate = load_audio(args.waveform)
    speech_array_2, sample_rate = load_audio(args.waveform_tgt)

    label = 0 # Ex. trial is False

    augmented_array = augment(speech_array, speech_array_2, y=label)

    # Reshape to 2D: [channels, samples]
    if augmented_array.dim() == 1:
        augmented_array = augmented_array.unsqueeze(0)
    output_name = (args.waveform.split(".wav")[0]) + "_" + args.simulate + ".wav"

    # Save to output file
    torchaudio.save(args.output_dir+output_name, augmented_array.cpu(), sample_rate=sample_rate)

    print(f"Saved augmented audio to: {args.output_dir+output_name}")
    
if __name__ == "__main__":
    main()

