import librosa
import numpy as np
import torch
from scipy.fft import dct

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

def audio_to_cfcc(audio, sr, n_chroma=12, n_cfcc=13, hop_length=512, n_fft=2048, max_time_frames=None):
    """
    Extract CFCC (Chroma Feature Cepstral Coefficients) from audio.
    Returns both CFCC coefficients and chroma filterbank energies.
    
    This implementation follows the approach where:
    1. First, chroma filterbank energies are extracted (representing pitch classes)
    2. Then, DCT is applied to get cepstral coefficients (CFCC)
    3. Both representations are merged into one 2D vector
    
    The dataset structure expects:
    - First file in each raga folder is the reference dataset
    - Other files are improved versions of the audio for that raga
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_chroma: Number of chroma bins (default: 12 for 12 semitones)
        n_cfcc: Number of CFCC coefficients to return
        hop_length: Hop length for STFT
        n_fft: FFT window size
        max_time_frames: Maximum time frames to pad/truncate to
    
    Returns:
        cfcc_coeffs: CFCC coefficients (n_cfcc, time_frames)
        chroma_energies: Chroma filterbank energies (n_chroma, time_frames)
        merged_features: Combined 2D vector representation (n_chroma + n_cfcc, time_frames)
    """
    # Extract chroma features (filterbank energies)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma, 
                                        hop_length=hop_length, n_fft=n_fft)
    
    # Apply logarithm to chroma energies (similar to mel-spectrogram processing)
    log_chroma = np.log(chroma + 1e-10)  # Add small epsilon to avoid log(0)
    
    # Apply DCT to get cepstral coefficients (CFCC)
    # Note: DCT can only produce as many coefficients as input features (n_chroma)
    cfcc_coeffs = dct(log_chroma, axis=0, norm='ortho')
    # Take the requested number of coefficients, but limited by available coefficients
    n_cfcc_actual = min(n_cfcc, cfcc_coeffs.shape[0])
    cfcc_coeffs = cfcc_coeffs[:n_cfcc_actual, :]
    
    # Handle time frame padding/truncation
    if max_time_frames is not None:
        # Process CFCC coefficients
        if cfcc_coeffs.shape[1] < max_time_frames:
            pad_width = max_time_frames - cfcc_coeffs.shape[1]
            cfcc_coeffs = np.pad(cfcc_coeffs, ((0, 0), (0, pad_width)), mode='constant')
        elif cfcc_coeffs.shape[1] > max_time_frames:
            cfcc_coeffs = cfcc_coeffs[:, :max_time_frames]
        
        # Process chroma energies
        if chroma.shape[1] < max_time_frames:
            pad_width = max_time_frames - chroma.shape[1]
            chroma = np.pad(chroma, ((0, 0), (0, pad_width)), mode='constant')
        elif chroma.shape[1] > max_time_frames:
            chroma = chroma[:, :max_time_frames]
    
    # Merge CFCC coefficients and chroma energies into one 2D vector representation
    merged_features = np.concatenate([cfcc_coeffs, chroma], axis=0)
    
    return cfcc_coeffs, chroma, merged_features

def extract_cfcc_features(audio, sr, n_chroma=12, n_cfcc=13, hop_length=512, n_fft=2048, max_time_frames=None):
    """
    Convenience function to extract only merged CFCC features.
    This is the main function to use for the dataset.
    """
    _, _, merged_features = audio_to_cfcc(audio, sr, n_chroma, n_cfcc, hop_length, n_fft, max_time_frames)
    return merged_features
