import os
import sys
import random
import numpy as np
import torchaudio
import librosa

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.augment import stitch_clips

def generate_test_song(dataset_path, output_path, duration=60, sr=22050):
    """
    Generate a test song by stitching segments from different ragas
    
    Args:
        dataset_path: Path to the raga dataset
        output_path: Path to save the generated test song
        duration: Duration of the test song in seconds
        sr: Sample rate
    """
    # Collect all ragas and their files
    ragas = {}
    for raga in os.listdir(dataset_path):
        raga_dir = os.path.join(dataset_path, raga)
        if not os.path.isdir(raga_dir):
            continue
        files = [os.path.join(raga_dir, fname) for fname in os.listdir(raga_dir) if fname.endswith('.wav')]
        if files:
            ragas[raga] = files
    
    # Generate test song by stitching segments
    segments = []
    segment_labels = []
    current_time = 0
    
    while current_time < duration:
        # Select a random raga
        raga = random.choice(list(ragas.keys()))
        files = ragas[raga]
        
        # Select a random file from this raga
        file_path = random.choice(files)
        
        # Load audio
        audio, file_sr = torchaudio.load(file_path)
        audio = audio.mean(dim=0).numpy()  # mono
        
        # Resample if needed
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        
        # Determine segment length (between 5 and 15 seconds)
        seg_len = random.uniform(5, min(15, duration - current_time))
        
        # Extract segment
        if len(audio) > int(seg_len * sr):
            max_offset = len(audio) - int(seg_len * sr)
            offset = random.randint(0, max_offset)
            segment = audio[offset:offset + int(seg_len * sr)]
        else:
            # If segment is too short, pad it
            segment = audio
            if len(segment) < int(seg_len * sr):
                pad_len = int(seg_len * sr) - len(segment)
                segment = np.pad(segment, (0, pad_len))
            else:
                segment = segment[:int(seg_len * sr)]
        
        segments.append(segment)
        segment_labels.append((current_time, current_time + seg_len, raga))
        current_time += seg_len
    
    # Concatenate all segments
    test_song = np.concatenate(segments)
    
    # Save the test song
    torchaudio.save(output_path, torch.tensor(test_song).unsqueeze(0), sr)
    
    # Print the ground truth segmentation
    print("Ground truth segmentation:")
    for start, end, raga in segment_labels:
        print(f"{start:.2f}s - {end:.2f}s: {raga}")
    
    return segment_labels

if __name__ == "__main__":
    import torch
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test audio by stitching raga segments')
    parser.add_argument('--dataset', type=str, default='PallaviData', help='Path to dataset')
    parser.add_argument('--output', type=str, default='test_song.wav', help='Output audio path')
    parser.add_argument('--duration', type=int, default=60, help='Duration of test song in seconds')
    
    args = parser.parse_args()
    
    print(f"Generating test song: {args.output}")
    segment_labels = generate_test_song(args.dataset, args.output, args.duration)
    print(f"Test song saved to {args.output}")
