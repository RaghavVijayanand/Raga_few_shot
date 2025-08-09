import os
import torch
import torchaudio
import random
import librosa
import numpy as np
from torch.utils.data import Dataset
from augment import pitch_shift, time_stretch, add_noise
from features import extract_cfcc_features

class RagaDataset(Dataset):
    def __init__(self, root_dir, sr=22050, duration=5.0, augment=False, n_chroma=12, n_cfcc=13, max_time_frames=None):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.n_chroma = n_chroma
        self.n_cfcc = n_cfcc
        self.max_time_frames = max_time_frames
        self.ragas = self._collect_ragas()
        self.label2idx = {raga: i for i, raga in enumerate(sorted(self.ragas.keys()))}

    def _collect_ragas(self):
        ragas = {}
        for raga in os.listdir(self.root_dir):
            raga_dir = os.path.join(self.root_dir, raga)
            if not os.path.isdir(raga_dir):
                continue
            files = [os.path.join(raga_dir, fname) for fname in os.listdir(raga_dir) if fname.endswith('.wav')]
            if files:
                ragas[raga] = files
        return ragas

    def __len__(self):
        # Return total number of audio files
        return sum(len(files) for files in self.ragas.values())

    def __getitem__(self, idx):
        # Get a random raga and file
        raga = random.choice(list(self.ragas.keys()))
        files = self.ragas[raga]
        path = random.choice(files)
        
        audio, sr = torchaudio.load(path)
        audio = audio.mean(dim=0).numpy()  # mono
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # Augmentation
        if self.augment:
            if random.random() < 0.3:
                audio = pitch_shift(audio, self.sr, n_steps=random.choice([-2, -1, 1, 2]))
            if random.random() < 0.3:
                audio = time_stretch(audio, rate=random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                audio = add_noise(audio)
        
        # Random crop or pad to fixed duration
        target_length = int(self.duration * self.sr)
        if len(audio) > target_length:
            max_offset = len(audio) - target_length
            offset = random.randint(0, max_offset)
            audio = audio[offset:offset + target_length]
        else:
            # Pad if too short
            pad_len = target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode='constant')
        
        # Feature extraction
        # The max_time_frames parameter ensures all features have the same time dimension
        cfcc_features = extract_cfcc_features(audio, self.sr, n_chroma=self.n_chroma, n_cfcc=self.n_cfcc, max_time_frames=self.max_time_frames)
        cfcc_features = torch.tensor(cfcc_features).float()
        return cfcc_features, self.label2idx[raga]

    def get_episode(self, n_way=5, n_shot=3, n_query=2):
        """Generate a few-shot episode with n_way classes, n_shot support, n_query query examples"""
        # Filter ragas that have enough files
        valid_ragas = {raga: files for raga, files in self.ragas.items() 
                      if len(files) >= n_shot + n_query}
        
        if len(valid_ragas) < n_way:
            # If not enough ragas with sufficient files, use all available and adjust n_way
            n_way = min(n_way, len(valid_ragas))
            if n_way == 0:
                raise ValueError("No ragas have enough files for the requested episode configuration")
        
        # Select n_way random ragas from valid ones
        selected_ragas = random.sample(list(valid_ragas.keys()), n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for i, raga in enumerate(selected_ragas):
            files = valid_ragas[raga]
            # Select n_shot + n_query files for this raga
            selected_files = random.sample(files, n_shot + n_query)
            
            # Process support examples
            for j in range(n_shot):
                cfcc_features = self._process_file(selected_files[j])
                if cfcc_features is not None:
                    support_data.append(cfcc_features)
                    support_labels.append(i)
            
            # Process query examples
            for j in range(n_shot, n_shot + n_query):
                cfcc_features = self._process_file(selected_files[j])
                if cfcc_features is not None:
                    query_data.append(cfcc_features)
                    query_labels.append(i)
        
        # Ensure we have the expected number of examples
        if len(support_data) < n_way * n_shot or len(query_data) < n_way * n_query:
            # Recursively try again if we didn't get enough valid examples
            return self.get_episode(n_way, n_shot, n_query)
        
        return (torch.stack(support_data), torch.tensor(support_labels)), (torch.stack(query_data), torch.tensor(query_labels))
    
    def _process_file(self, path):
        """Process a single audio file and return CFCC features"""
        try:
            audio, sr = torchaudio.load(path)
            audio = audio.mean(dim=0).numpy()  # mono
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            
            # Augmentation
            if self.augment:
                if random.random() < 0.3:
                    audio = pitch_shift(audio, self.sr, n_steps=random.choice([-2, -1, 1, 2]))
                if random.random() < 0.3:
                    audio = time_stretch(audio, rate=random.uniform(0.8, 1.2))
                if random.random() < 0.3:
                    audio = add_noise(audio)

            # Random crop or pad to fixed duration
            target_length = int(self.duration * self.sr)
            if len(audio) > target_length:
                max_offset = len(audio) - target_length
                offset = random.randint(0, max_offset)
                audio = audio[offset:offset + target_length]
            else:
                # Pad if too short
                pad_len = target_length - len(audio)
                audio = np.pad(audio, (0, pad_len), mode='constant')
            
            # Feature extraction
            cfcc_features = extract_cfcc_features(audio, self.sr, n_chroma=self.n_chroma, n_cfcc=self.n_cfcc, max_time_frames=self.max_time_frames)
            return torch.tensor(cfcc_features).float()
            
        except Exception as e:
            print(f"Error processing file {path}: {e}")
            return None
