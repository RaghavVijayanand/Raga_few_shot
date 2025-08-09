#!/usr/bin/env python3
"""
Test script to verify CFCC feature extraction functionality.
"""

import numpy as np
import librosa
from src.features import audio_to_cfcc, extract_cfcc_features

def test_cfcc_extraction():
    """Test CFCC feature extraction with synthetic audio."""
    
    # Generate a simple synthetic audio signal
    sr = 22050
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple harmonic signal with multiple frequencies (like a musical note)
    fundamental = 440  # A4 note
    audio = (np.sin(2 * np.pi * fundamental * t) + 
             0.5 * np.sin(2 * np.pi * 2 * fundamental * t) + 
             0.3 * np.sin(2 * np.pi * 3 * fundamental * t))
    
    # Add some noise
    audio += 0.1 * np.random.randn(len(audio))
    
    print("Testing CFCC feature extraction...")
    print(f"Audio duration: {duration} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Audio shape: {audio.shape}")
    
    # Test CFCC extraction
    n_chroma = 12
    n_cfcc = 13
    max_time_frames = 216  # Common size for 2-second audio
    
    cfcc_coeffs, chroma_energies, merged_features = audio_to_cfcc(
        audio, sr, n_chroma=n_chroma, n_cfcc=n_cfcc, max_time_frames=max_time_frames
    )
    
    print(f"\nCFCC coefficients shape: {cfcc_coeffs.shape}")
    print(f"Chroma energies shape: {chroma_energies.shape}")
    print(f"Merged features shape: {merged_features.shape}")
    
    # Test convenience function
    merged_features_2 = extract_cfcc_features(
        audio, sr, n_chroma=n_chroma, n_cfcc=n_cfcc, max_time_frames=max_time_frames
    )
    
    print(f"Convenience function output shape: {merged_features_2.shape}")
    
    # Verify that both methods give same result
    assert np.allclose(merged_features, merged_features_2), "Convenience function should match main function"
    
    # Check expected dimensions
    # Note: CFCC can only produce as many coefficients as chroma bins (12)
    # So even if we request 13 CFCC coefficients, we get min(13, 12) = 12
    n_cfcc_actual = min(n_cfcc, n_chroma)  # min(13, 12) = 12
    expected_height = n_chroma + n_cfcc_actual  # 12 + 12 = 24
    expected_width = max_time_frames     # 216
    
    print(f"Requested CFCC coefficients: {n_cfcc}, Actual: {n_cfcc_actual}")
    print(f"Expected merged features shape: ({expected_height}, {expected_width})")
    
    assert merged_features.shape == (expected_height, expected_width), \
        f"Expected shape ({expected_height}, {expected_width}), got {merged_features.shape}"
    
    print(f"\n✓ CFCC feature extraction working correctly!")
    print(f"✓ Final feature dimensions: {merged_features.shape}")
    print(f"✓ Features contain {n_cfcc_actual} CFCC coefficients + {n_chroma} chroma energies = {expected_height} features")
    
    # Show some statistics
    print(f"\nFeature statistics:")
    print(f"Mean: {np.mean(merged_features):.4f}")
    print(f"Std: {np.std(merged_features):.4f}")
    print(f"Min: {np.min(merged_features):.4f}")
    print(f"Max: {np.max(merged_features):.4f}")

if __name__ == "__main__":
    test_cfcc_extraction()
