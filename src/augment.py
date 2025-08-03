import librosa
import numpy as np
import random

# Rename librosa's pitch_shift to avoid conflict
from librosa.effects import pitch_shift as librosa_pitch_shift

def pitch_shift(audio, sr, n_steps):
    return librosa_pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def stitch_clips(clips, sr, seg_len=5.0):
    """Stitch random segments from different ragas to simulate transitions."""
    stitched = []
    for clip in clips:
        start = random.randint(0, max(0, len(clip) - int(seg_len * sr)))
        stitched.append(clip[start:start + int(seg_len * sr)])
    return np.concatenate(stitched)
