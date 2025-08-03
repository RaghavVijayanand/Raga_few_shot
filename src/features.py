import librosa
import numpy as np
import torch

def audio_to_melspec(audio, sr, n_mels=128, hop_length=512, n_fft=2048, max_time_frames=None):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    if max_time_frames is not None:
        # Pad or truncate to max_time_frames
        if S_db.shape[1] < max_time_frames:
            pad_width = max_time_frames - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        elif S_db.shape[1] > max_time_frames:
            S_db = S_db[:, :max_time_frames]
            
    return S_db

def audio_to_mfcc(audio, sr, n_mfcc=40, hop_length=512, n_fft=2048, max_time_frames=None):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    
    if max_time_frames is not None:
        # Pad or truncate to max_time_frames
        if mfcc.shape[1] < max_time_frames:
            pad_width = max_time_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > max_time_frames:
            mfcc = mfcc[:, :max_time_frames]
            
    return mfcc
