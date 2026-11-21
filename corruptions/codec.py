import soundfile as sf
import subprocess
import sys
import io
import numpy as np
import torch
import torch

# Codec configurations
codec_configs = {
    'pcm_mulaw':    {'codec': 'pcm_mulaw',          'ar': 8000,     'bitrate': '64k',   'fmt': 'wav'},  # G.711 µ-law
    'pcm_alaw':     {'codec': 'pcm_alaw',           'ar': 8000,     'bitrate': '64k',   'fmt': 'wav'},  # G.711 A-law                                   
    'g722':         {'codec': 'g722',               'ar': 16000,    'bitrate': '64k',   'fmt': 'g722'}, # G.722
    'g726':         {'codec': 'adpcm_g726',         'ar': 8000,     'bitrate': '40k',   'fmt': 'wav'},  # G.726
    'gsm':          {'codec': 'gsm',                'ar': 8000,     'bitrate': '13.2k', 'fmt': 'gsm'},  # GSM 6.10 FR
    'ilbc':         {'codec': 'ilbc',               'ar': 8000,     'bitrate': '15.2k', 'fmt': 'ilbc'}, # iLBC  
    'amr_nb':       {'codec': 'libopencore_amrnb',  'ar': 8000,     'bitrate': '12.2k', 'fmt': 'amr'},  # AMR NB
    'amr_wb':       {'codec': 'libvo_amrwbenc',     'ar': 16000,    'bitrate': '23.85k','fmt': 'amr'},  # AMR WB
    'speex_nb':     {'codec': 'libspeex',           'ar': 8000,     'bitrate': '8k',    'fmt': 'ogg'},  # Speex NB  
    'speex_wb':     {'codec': 'libspeex',           'ar': 16000,    'bitrate': '11.2k', 'fmt': 'ogg'},  # Speex WB    
    'opus_nb':      {'codec': 'libopus',            'ar': 8000,     'bitrate': '32k',   'fmt': 'ogg'},  # Opus NB
    'opus_mb':      {'codec': 'libopus',            'ar': 12000,    'bitrate': '48k',   'fmt': 'ogg'},  # Opus MB
    'opus_wb':      {'codec': 'libopus',            'ar': 16000,    'bitrate': '64k',   'fmt': 'ogg'},  # Opus WB
}

class CodecAug(torch.nn.Module):
    """
    Wrap a single codec as an nn.Module augmentation.
    """
    def __init__(self, codec_name: str, sample_rate: int):
        super().__init__()
        if codec_name not in codec_configs:
            raise ValueError(f"Unknown codec: {codec_name}")
        self.codec_name = codec_name
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # convert torch -> numpy
        arr = x.detach().cpu().numpy()
        cfg = codec_configs[self.codec_name]
        if x.ndim == 1:
            decoded = codec_roundtrip(arr, self.sample_rate, cfg)
        else:
            # [channels, frames]: the narrowband codecs are mono, so code each channel on its own
            decoded = np.stack([codec_roundtrip(ch, self.sample_rate, cfg) for ch in arr])
        return torch.from_numpy(decoded).to(device=x.device, dtype=x.dtype)

def codec_roundtrip(data, sr, cfg):
    """
    Encodes a mono waveform (1D float array in [-1,1]) with the codec in cfg,
    decodes it back to PCM at the original sample rate and returns a float32
    array of the same length as the input.
    """
    buf_in = io.BytesIO()
    sf.write(buf_in, data, sr, format='WAV')

    # 1) encode. The codec has to be written to a container here; putting a
    #    second -acodec on the same output would replace the first one.
    enc_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', 'wav', '-i', 'pipe:0',
        '-c:a', cfg['codec'], '-ac', '1',
    ]
    if 'ar' in cfg:
        enc_cmd += ['-ar', str(cfg['ar'])]
    if 'bitrate' in cfg:
        enc_cmd += ['-b:a', cfg['bitrate']]
    enc_cmd += ['-f', cfg['fmt'], 'pipe:1']
    enc = subprocess.run(enc_cmd, input=buf_in.getvalue(), capture_output=True)
    if enc.returncode != 0 or not enc.stdout:
        raise RuntimeError(f"ffmpeg encoding with {cfg['codec']} failed: {enc.stderr.decode(errors='replace')}")

    # 2) decode back to raw PCM16 LE at the original sample rate
    dec_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-f', cfg['fmt'], '-i', 'pipe:0',
        '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1',
        '-f', 's16le', 'pipe:1',
    ]
    dec = subprocess.run(dec_cmd, input=enc.stdout, capture_output=True)
    if dec.returncode != 0 or not dec.stdout:
        raise RuntimeError(f"ffmpeg decoding of {cfg['codec']} failed: {dec.stderr.decode(errors='replace')}")

    pcm = np.frombuffer(dec.stdout, dtype=np.int16).astype(np.float32) / 32768.0   # normalize back to [-1,1]
    # codecs pad to whole frames: keep the input length
    n = len(data)
    if len(pcm) < n:
        pcm = np.pad(pcm, (0, n - len(pcm)))
    return pcm[:n]

def encode_decode(path_to_audio):
    """
    Loads an audio file, then for each codec:
      • encodes the audio
      • immediately decodes to raw PCM16 LE
      • returns the waveform as a NumPy array
    """
    # 1) Read original audio & metadata
    data, sr = sf.read(path_to_audio)                    # float64 in [-1,1]
    orig_channels = data.shape[1] if data.ndim > 1 else 1
    
    results = {}
    for name, cfg in codec_configs.items():
        if orig_channels > 1:
            results[name] = np.stack([codec_roundtrip(data[:, c], sr, cfg) for c in range(orig_channels)], axis=1)
        else:
            results[name] = codec_roundtrip(data, sr, cfg)

    return results

def compute_mse_snr(orig, decoded):
    """
    orig, decoded: 1D or 2D float32 arrays in [-1,1]
    returns (mse, snr_db)
    """
    # align lengths
    n = min(orig.shape[0], decoded.shape[0])
    o = orig[:n]
    d = decoded[:n]
    # MSE
    mse = np.mean((o - d)**2)
    # Signal power / noise power
    sig_p = np.mean(o**2)
    noise_p = np.mean((o - d)**2)
    snr = 10 * np.log10(sig_p / (noise_p + 1e-12))
    return mse, snr

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python codec.py path/to/audio.wav")
        sys.exit(1)
    path = sys.argv[1]
    orig, sr = sf.read(path, dtype='float32')
    outputs = encode_decode(path)

    for codec, wav in outputs.items():
        print(f"{codec}: shape={wav.shape}, dtype={wav.dtype}")
    print()

    for name, dec in outputs.items():
        mse, snr = compute_mse_snr(orig, dec)
        print(f"{name:12s} → MSE: {mse:.4e},   SNR: {snr:.1f} dB")