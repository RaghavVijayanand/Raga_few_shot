"""
CFCC (Chroma Feature Cepstral Coefficients) Implementation Documentation

This implementation provides CFCC feature extraction for Raga classification with the following components:

1. **Chroma Filterbank Energies**: 
   - Extract 12-dimensional chroma features representing the 12 semitones
   - These capture the harmonic content and pitch relationships essential for Raga identification
   - Apply logarithm for better dynamic range representation

2. **CFCC Coefficients**:
   - Apply Discrete Cosine Transform (DCT) to log-chroma energies
   - Extract the first 13 CFCC coefficients (similar to MFCC approach)
   - DCT decorrelates the features and provides compact representation

3. **Merged 2D Vector Representation**:
   - Concatenate CFCC coefficients and chroma energies
   - Results in (13 + 12) = 25 dimensional features per time frame
   - Final shape: (25, time_frames) for each audio sample

## Dataset Structure Understanding:

Based on the dataset structure, each raga folder contains multiple audio files:
- The first file (e.g., *_01_1_cent.wav) serves as the reference recording
- Subsequent files are improved/variant versions of the same raga
- This few-shot learning setup allows the model to learn from:
  - Reference examples (support set)
  - Improved versions (query set)

## Feature Pipeline:

Audio → STFT → Chroma Features → Log(Chroma) → DCT → CFCC Coefficients
                     ↓
              Chroma Energies ← 
                     ↓
            [CFCC Coeffs, Chroma Energies] → Merged 2D Vector

## Usage in Few-Shot Learning:

The merged CFCC features provide rich harmonic and timbral information suitable for:
- Distinguishing between different ragas
- Learning from few examples per raga
- Handling variations in the same raga (reference vs improved versions)

## Key Parameters:

- n_chroma: 12 (12 semitones in Western music)
- n_cfcc: 13 (first 13 DCT coefficients)
- hop_length: 512 (time resolution)
- n_fft: 2048 (frequency resolution)
- max_time_frames: Variable (depends on audio duration)

## Output Dimensions:

For a typical 5-second audio at 22050 Hz with hop_length=512:
- Time frames ≈ (5 * 22050) / 512 ≈ 215 frames
- Feature shape: (25, 215) per audio sample
"""
