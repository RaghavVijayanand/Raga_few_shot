# Few-Shot Raga Segmentation Pipeline

## Overview
This project implements a few-shot learning pipeline for classifying and segmenting ragas from audio using a Kolmogorov-Arnold Network (KAN) with a CNN backbone (ResNet18) and Prototypical Networks.

## Architecture
- **CNN Backbone**: ResNet18 adapted for 1-channel mel spectrograms
- **KAN Head**: Kolmogorov-Arnold Network for feature transformation
- **Prototypical Networks**: Few-shot learning approach using support/query episodes
- **Audio Processing**: Mel spectrograms with fixed dimensions for consistency

## Structure
- `src/augment.py`: Audio augmentation and transition simulation
- `src/features.py`: Mel spectrogram and MFCC extraction with padding/truncation
- `src/dataset.py`: Dataset and few-shot episode generation
- `src/models/`: Model components (CNN, KAN, ProtoNet)
- `src/train.py`: Training loop with few-shot episodes
- `src/inference.py`: Inference and segmentation for full songs
- `src/utils.py`: Visualization utilities
- `src/generate_test_audio.py`: Generate test songs by stitching raga segments

## Key Features
- **Few-shot learning**: Uses prototypical networks with support/query episodes
- **Fixed dimensions**: All mel spectrograms are padded/truncated to consistent size
- **Modular design**: Easy to swap CNN backbone, KAN implementation, or few-shot method
- **Data augmentation**: Pitch shifting, time stretching, and noise addition
- **Visualization**: Time-aligned raga segmentation plots

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Place audio files in `PallaviData/<RagaName>/*.wav`. The dataset should have:
- ~30-40 different ragas
- ~8 audio clips per raga (~1-2 minutes each)

### 3. Train the model
```bash
python train.py
```
This will:
- Generate 100 few-shot episodes per epoch
- Save checkpoints every 10 epochs
- Store final model in `checkpoints/final_model.pth`

### 4. Generate test audio (optional)
```bash
python src/generate_test_audio.py --output test_song.wav --duration 60
```

### 5. Run inference
```bash
python src/inference.py --audio test_song.wav --output segmentation.png
```

## Configuration
Key parameters in `train.py`:
- `N_WAY`: Number of classes per episode (default: 5)
- `N_SHOT`: Support examples per class (default: 3)
- `N_QUERY`: Query examples per class (default: 2)
- `N_EPOCHS`: Training epochs (default: 50)
- `N_MELS`: Mel spectrogram bins (default: 128)
- `FEAT_DIM`: Feature dimension (default: 128)

## Trade-offs and Considerations

### KAN vs MLP
- **KAN**: Theoretical advantages in function approximation, but current implementation uses MLP as placeholder
- **MLP**: Simpler, more stable training, widely used in few-shot learning
- **Recommendation**: Start with MLP, experiment with actual KAN implementation later

### CNN vs Transformer
- **CNN (ResNet)**: Efficient for spectrograms, good local feature extraction
- **Transformer**: Better for sequential patterns, but requires more data
- **Current choice**: CNN for simplicity and efficiency

### Few-shot Methods
- **Prototypical Networks**: Simple, effective for small datasets
- **Matching Networks**: More complex, can handle variable support sizes
- **Relation Networks**: Learns similarity metric, but needs more training

## Evaluation Metrics
- **Accuracy**: Per-segment classification accuracy
- **F1-score**: Harmonic mean of precision and recall
- **Temporal overlap**: IoU between predicted and ground truth segments

## Future Improvements
1. **Sequential modeling**: Add LSTM/Transformer for temporal context
2. **Attention mechanisms**: Self-attention in CNN backbone
3. **Advanced KAN**: Implement actual Kolmogorov-Arnold Network
4. **Data augmentation**: More sophisticated transition simulation
5. **Multi-scale features**: Combine MFCC and mel spectrograms
6. **Ensemble methods**: Combine multiple few-shot approaches

## Notes
- The KAN model is currently implemented as an MLP placeholder
- All audio is processed at 22.05kHz sample rate
- Mel spectrograms are fixed at 128 bins × 216 time frames
- The pipeline is designed for few-shot learning with limited data
