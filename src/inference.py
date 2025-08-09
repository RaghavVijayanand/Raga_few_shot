import os
import sys

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchaudio
import numpy as np
import os
import librosa
from src.features import extract_cfcc_features
from src.models.cnn_backbone import CNNBackbone
from src.models.kan import KANHead
from src.models.proto_head import ProtoNetHead
from src.utils import plot_raga_segments
import argparse

# Constants
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
HOP_LENGTH = 512  # for CFCC feature extraction
MAX_TIME_FRAMES = int(np.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH))  # Calculate max time frames
N_CHROMA = 12  # Number of chroma bins
N_CFCC = 13    # Number of CFCC coefficients
CFCC_HEIGHT = N_CHROMA + N_CFCC  # Total CFCC feature height

def load_model(checkpoint_path, feat_dim=128):
    """Load trained model from checkpoint"""
    # Note: CFCC can only produce as many coefficients as chroma bins (12)
    # So even if we request 13 CFCC coefficients, we get min(13, 12) = 12
    n_cfcc_actual = min(N_CFCC, N_CHROMA)
    cfcc_height = N_CHROMA + n_cfcc_actual # 12 + 12 = 24

    # Initialize models
    cnn = CNNBackbone(out_dim=feat_dim, input_height=cfcc_height)
    kan = KANHead(in_dim=feat_dim, hidden_dim=128, out_dim=feat_dim)  # Match training config
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    kan.load_state_dict(checkpoint['kan_state_dict'])
    
    # Set to evaluation mode
    cnn.eval()
    kan.eval()
    
    return cnn, kan

def create_support_set(dataset_path, cnn, kan, n_way=None, n_shot=2, feat_dim=128):
    """Create a support set from the dataset for few-shot inference"""
    import os
    import random
    import torchaudio
    import librosa
    
    # Collect all ragas
    ragas = {}
    for raga in os.listdir(dataset_path):
        raga_dir = os.path.join(dataset_path, raga)
        if not os.path.isdir(raga_dir):
            continue
        files = [os.path.join(raga_dir, fname) for fname in os.listdir(raga_dir) if fname.endswith('.wav')]
        if len(files) >= n_shot:  # Only include ragas with enough files
            ragas[raga] = files
    
    # Use all available ragas if n_way is not specified
    if n_way is None:
        n_way = len(ragas)
    else:
        n_way = min(n_way, len(ragas))
    
    # Select ragas (all of them for inference, or random subset if specified)
    if n_way == len(ragas):
        selected_ragas = list(ragas.keys())
    else:
        selected_ragas = random.sample(list(ragas.keys()), n_way)
    
    print(f"Creating support set with {len(selected_ragas)} ragas: {selected_ragas}")
    
    support_data, support_labels = [], []
    label_map = {}
    
    for i, raga in enumerate(selected_ragas):
        files = ragas[raga]
        # Select n_shot files for this raga
        selected_files = random.sample(files, min(len(files), n_shot))
        
        # Process support examples
        for j in range(len(selected_files)):
            try:
                path = selected_files[j]
                audio, sr = torchaudio.load(path)
                audio = audio.mean(dim=0).numpy()  # mono
                if sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                
                # Random crop to 5 seconds
                if len(audio) > int(DURATION * SAMPLE_RATE):
                    max_offset = len(audio) - int(DURATION * SAMPLE_RATE)
                    offset = random.randint(0, max_offset)
                    audio = audio[offset:offset + int(DURATION * SAMPLE_RATE)]
                else:
                    # Pad if too short
                    pad_len = int(DURATION * SAMPLE_RATE) - len(audio)
                    audio = np.pad(audio, (0, pad_len))
                
                # Feature extraction
                cfcc_features = extract_cfcc_features(audio, SAMPLE_RATE, n_chroma=N_CHROMA, n_cfcc=N_CFCC, max_time_frames=MAX_TIME_FRAMES)
                cfcc_features = torch.tensor(cfcc_features).float()
                support_data.append(cfcc_features)
                support_labels.append(i)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        label_map[i] = raga
    
    # Extract features
    with torch.no_grad():
        support_data = torch.stack(support_data).unsqueeze(1)  # (n_way*n_shot, 1, cfcc_height, time)
        support_feats = cnn(support_data)
        support_feats = kan(support_feats)  # Apply KAN transformation
    
    return support_feats, torch.tensor(support_labels), label_map

def segment_audio(audio_path, cnn, kan, proto, support_feats, support_labels, 
                  sr=SAMPLE_RATE, top_db=30.0): # Changed default top_db to 30.0
    """Segment audio into raga segments, skipping silence"""
    # Load audio
    waveform, file_sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0).numpy()  # mono
    
    # Resample if needed
    if file_sr != sr:
        waveform = librosa.resample(waveform, orig_sr=file_sr, target_sr=sr)
    
    # Detect non-silent segments
    # librosa.effects.split returns a list of [start_sample, end_sample] for non-silent segments
    # Using default frame_length=2048 and hop_length=512 for split
    active_segments = librosa.effects.split(waveform, top_db=top_db, frame_length=2048, hop_length=512) 
    
    segments = []
    times = []
    
    min_segment_duration = 7.0 # seconds

    for start_sample, end_sample in active_segments:
        segment_duration = (end_sample - start_sample) / sr
        
        # Skip segments shorter than the minimum duration
        if segment_duration < min_segment_duration:
            continue

        # Extract audio segment
        audio_seg = waveform[start_sample:end_sample]
        
        # Ensure segment is not empty (though the duration check should cover this)
        if len(audio_seg) == 0:
            continue

        # Convert to CFCC features
        # extract_cfcc_features pads if too short, and might truncate if too long based on MAX_TIME_FRAMES
        cfcc_features = extract_cfcc_features(audio_seg, sr, n_chroma=N_CHROMA, n_cfcc=N_CFCC, max_time_frames=MAX_TIME_FRAMES)
        cfcc_features = torch.tensor(cfcc_features).unsqueeze(0).unsqueeze(0).float()  # (1,1,cfcc_height,time)
        
        # Extract features
        with torch.no_grad():
            feat = cnn(cfcc_features)
            feat = kan(feat)
        
        # Classify using prototypical network
        logits = proto(support_feats, support_labels, feat)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        confidence = probs.max(dim=1)[0].item()
        
        # Only include predictions with reasonable confidence
        if confidence > 0.3:  # Threshold for confidence
            segments.append(pred)
        else:
            segments.append(-1)  # Unknown/low confidence
            
        times.append(start_sample / sr) # Store the start time of the segment
    
    # Add the end time of the last segment to complete the timeline for plotting
    if times: # If there were any detected segments
        last_segment_end_time = active_segments[-1][1] / sr
        times.append(last_segment_end_time)
    else: # Handle case where no sound is detected
        times.append(0) # Default to 0 if no segments found

    return times, segments

def main():
    parser = argparse.ArgumentParser(description='Raga segmentation using few-shot learning')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file for segmentation')
    parser.add_argument('--model', type=str, default='checkpoints/final_model.pth', help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='../PallaviData', help='Path to dataset for support set')
    parser.add_argument('--output', type=str, default='segmentation.png', help='Output visualization path')
    parser.add_argument('--window', type=float, default=5.0, help='Window size in seconds (not used with silence skipping)')
    parser.add_argument('--hop', type=float, default=2.5, help='Hop size in seconds (not used with silence skipping)')
    parser.add_argument('--top_db', type=float, default=30.0, help='Threshold for silence detection (dB)') # Changed default to 30.0
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")
    
    # Load model
    print("Loading model...")
    cnn, kan = load_model(args.model)
    proto = ProtoNetHead()
    
    # Count the number of ragas in the dataset that have enough files
    dataset_path = args.dataset
    num_ragas = 0
    if os.path.isdir(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                # Check if it contains enough .wav files
                wav_files = [fname for fname in os.listdir(item_path) if fname.endswith('.wav')]
                if len(wav_files) >= 2:  # Need at least 2 files for 2-shot
                    num_ragas += 1

    # Use all available ragas for inference
    n_way = num_ragas
    n_shot = 2  # 2-shot as requested
    
    print(f"Found {num_ragas} ragas with sufficient files for 2-shot learning")

    # Create support set
    print("Creating support set...")
    support_feats, support_labels, label_map = create_support_set(
        args.dataset, cnn, kan, n_way=None, n_shot=n_shot  # Use all ragas
    )
    
    # Segment audio using silence skipping
    print(f"Segmenting audio: {args.audio}")
    times, segments = segment_audio(
        args.audio, cnn, kan, proto, support_feats, support_labels,
        sr=SAMPLE_RATE, top_db=args.top_db # Pass top_db to segment_audio
    )
    
    # Print results
    print("Segmentation results:")
    if segments: # Only print if there are segments
        for i, (start, end, label) in enumerate(zip(times[:-1], times[1:], segments)):
            if label == -1:
                print(f"{start:.2f}s - {end:.2f}s: Unknown/Low confidence")
            else:
                print(f"{start:.2f}s - {end:.2f}s: {label_map[label]}")
    else:
        print("No sound segments detected.")
    
    # Visualize results
    print(f"Saving visualization to {args.output}")
    plot_raga_segments(times, segments, label_map, out_path=args.output)
    
    print("Segmentation completed!")

if __name__ == "__main__":
    main()
