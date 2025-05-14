import soundfile as sf
import subprocess
import sys
import io
import numpy as np
import torch

# Codec configurations
codec_configs = {
    'pcm_mulaw':    {'codec': 'pcm_mulaw',          'ar': 8000,     'bitrate': '64k'},      # G.711 µ-law
    'pcm_alaw':     {'codec': 'pcm_alaw',           'ar': 8000,     'bitrate': '64k'},      # G.711 A-law                                   
    'g722':         {'codec': 'g722',               'ar': 16000,    'bitrate': '64k'},      # G.722
    'g726':         {'codec': 'adpcm_g726',         'ar': 8000,     'bitrate': '40k'},      # G.726
    'gsm':          {'codec': 'gsm',                'ar': 8000,     'bitrate': '13.2k'},    # GSM 6.10 FR
    'ilbc':         {'codec': 'ilbc',               'ar': 8000,     'bitrate': '15.2k'},    # iLBC  
    'amr_nb':       {'codec': 'libopencore_amrnb',  'ar': 8000,     'bitrate': '12.2k'},    # AMR NB
    'amr_wb':       {'codec': 'libvo_amrwbenc',     'ar': 16000,    'bitrate': '23.85k'},   # AMR WB
    'speex_nb':     {'codec': 'libspeex',           'ar': 8000,     'bitrate': '8k'},       # Speex NB  
    'speex_wb':     {'codec': 'libspeex',           'ar': 16000,    'bitrate': '11.2k'},    # Speex WB    
    'opus_nb':      {'codec': 'libopus',            'ar': 8000,     'bitrate': '32k'},      # Opus NB
    'opus_mb':      {'codec': 'libopus',            'ar': 12000,    'bitrate': '48k'},      # Opus MB
    'opus_wb':      {'codec': 'libopus',            'ar': 16000,    'bitrate': '64k'},      # Opus WB
    'mp3':          {'codec': 'libmp3lame',         'ar': 16000,    'bitrate': '128k'},     # MP3 (LAME)
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
        # ffmpeg path uses float64
        arr = arr.astype(np.float64)
        decoded = encode_decode(arr, self.sample_rate)[self.codec_name]
        t = torch.from_numpy(decoded).to(x.device)
        # handle dims: numpy frames x channels -> torch [channels, frames]
        if x.ndim == 1:
            return t.reshape(-1)
        else:
            return t.T

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
    
    # 2) Serialize to WAV for ffmpeg input
    buf_in = io.BytesIO()
    sf.write(buf_in, data, sr, format='WAV')
    wav_bytes = buf_in.getvalue()

    results = {}
    for name, cfg in codec_configs.items():
        # build ffmpeg command
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-f', 'wav', '-i', 'pipe:0',                 # read input WAV
            '-c:a', cfg['codec'],
        ]
        if 'ar' in cfg:
            cmd += ['-ar', str(cfg['ar'])]
        if 'bitrate' in cfg:
            cmd += ['-b:a', cfg['bitrate']]
        # now decode back to raw PCM16 LE
        cmd += [
            '-acodec', 'pcm_s16le',
            '-f', 's16le',
            # preserve channel count:
            '-ac', str(orig_channels),
            'pipe:1'
        ]

        # run ffmpeg
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out_bytes, _ = proc.communicate(wav_bytes)

        # --- second pass: resample to 16k if needed ---
        if cfg.get('ar', None) != 16000:
            cmd2 = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-f', 's16le',
                '-ar', str(cfg['ar']),                       # original rate
                '-ac', str(orig_channels),
                '-i', 'pipe:0',
                '-ar', '16000',                              # target rate
                '-acodec', 'pcm_s16le',
                '-f', 's16le',
                '-ac', str(orig_channels),
                'pipe:1'
            ]
            proc2 = subprocess.Popen(cmd2, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            out_bytes, _ = proc2.communicate(out_bytes)

        # convert raw PCM bytes to numpy array
        pcm = np.frombuffer(out_bytes, dtype=np.int16)
        # reshape to (n_frames, n_channels)
        if orig_channels > 1:
            pcm = pcm.reshape(-1, orig_channels)
        results[name] = pcm.astype(np.float32) / 32768.0   # normalize back to [-1,1]

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
