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
    parser.add_argument("--config", type=str, default='./config/default.yaml', help="config file used to specify parameters")

    # data 
    parser.add_argument("--simulate", type=str, default='echo', help="Simulation scenario")
    parser.add_argument("--waveform", type=str, default='sample.wav', help="input filename")
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
    simulations = ['GuassianNoise', 'env_noise', 'music', 'crosstalk', 'rir', 'real_rir', 'echo', 'resample', 'gain', 'lowpass', 'highpass','codec']
    for simulate in simulations:
        augment = build_augmentation(simulate, config)
        speech_array, sample_rate = load_audio(args.waveform)
        augmented_array = augment(speech_array)
        # Reshape to 2D: [channels, samples]
        if augmented_array.dim() == 1:
            augmented_array = augmented_array.unsqueeze(0)
        output_name = (args.waveform.split(".wav")[0]) + "_" + simulate + ".wav"
        # Save to output file
        torchaudio.save(args.output_dir+output_name, augmented_array.cpu(), sample_rate=sample_rate)

        print(f"Saved augmented audio to: {args.output_dir+output_name}")
    
if __name__ == "__main__":
    main()

