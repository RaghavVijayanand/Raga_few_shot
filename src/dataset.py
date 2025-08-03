import os
import torch
import torchaudio
import random
import librosa
from torch.utils.data import Dataset
from .augment import pitch_shift, time_stretch, add_noise
from .features import audio_to_melspec

class RagaDataset(Dataset):
    def __init__(self, root_dir, sr=22050, duration=5.0, augment=False, n_mels=128, max_time_frames=None):
        self.root_dir = root_dir
        self.sr = sr
        self.duration = duration
        self.augment = augment
        self.n_mels = n_mels
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
        
        # Random crop
        if len(audio) > int(self.duration * self.sr):
            max_offset = len(audio) - int(self.duration * self.sr)
            offset = random.randint(0, max_offset)
            audio = audio[offset:offset + int(self.duration * self.sr)]
        else:
            # Pad if too short
            pad_len = int(self.duration * self.sr) - len(audio)
            audio = torch.nn.functional.pad(torch.tensor(audio), (0, pad_len)).numpy()
        
        # Augmentation
        if self.augment:
            if random.random() < 0.3:
                audio = pitch_shift(audio, self.sr, n_steps=random.choice([-2, -1, 1, 2]))
            if random.random() < 0.3:
                audio = time_stretch(audio, rate=random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                audio = add_noise(audio)
        
        # Feature extraction
        melspec = audio_to_melspec(audio, self.sr, n_mels=self.n_mels, max_time_frames=self.max_time_frames)
        melspec = torch.tensor(melspec).float()
        return melspec, self.label2idx[raga]

    def get_episode(self, n_way=5, n_shot=3, n_query=2):
        """Generate a few-shot episode with n_way classes, n_shot support, n_query query examples"""
        # Select n_way random ragas
        selected_ragas = random.sample(list(self.ragas.keys()), n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for i, raga in enumerate(selected_ragas):
            files = self.ragas[raga]
            # Select n_shot + n_query files for this raga
            selected_files = random.sample(files, min(len(files), n_shot + n_query))
            
            # Process support examples
            for j in range(min(n_shot, len(selected_files))):
                path = selected_files[j]
                audio, sr = torchaudio.load(path)
                audio = audio.mean(dim=0).numpy()  # mono
                if sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                
                # Random crop
                if len(audio) > int(self.duration * self.sr):
                    max_offset = len(audio) - int(self.duration * self.sr)
                    offset = random.randint(0, max_offset)
                    audio = audio[offset:offset + int(self.duration * self.sr)]
                else:
                    # Pad if too short
                    pad_len = int(self.duration * self.sr) - len(audio)
                    audio = torch.nn.functional.pad(torch.tensor(audio), (0, pad_len)).numpy()
                
                # Augmentation
                if self.augment:
                    if random.random() < 0.3:
                        audio = pitch_shift(audio, self.sr, n_steps=random.choice([-2, -1, 1, 2]))
                    if random.random() < 0.3:
                        audio = time_stretch(audio, rate=random.uniform(0.8, 1.2))
                    if random.random() < 0.3:
                        audio = add_noise(audio)
                
                # Feature extraction
                melspec = audio_to_melspec(audio, self.sr, n_mels=self.n_mels, max_time_frames=self.max_time_frames)
                melspec = torch.tensor(melspec).float()
                support_data.append(melspec)
                support_labels.append(i)
            
            # Process query examples
            for j in range(n_shot, min(n_shot + n_query, len(selected_files))):
                path = selected_files[j]
                audio, sr = torchaudio.load(path)
                audio = audio.mean(dim=0).numpy()  # mono
                if sr != self.sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
                
                # Random crop
                if len(audio) > int(self.duration * self.sr):
                    max_offset = len(audio) - int(self.duration * self.sr)
                    offset = random.randint(0, max_offset)
                    audio = audio[offset:offset + int(self.duration * self.sr)]
                else:
                    # Pad if too short
                    pad_len = int(self.duration * self.sr) - len(audio)
                    audio = torch.nn.functional.pad(torch.tensor(audio), (0, pad_len)).numpy()
                
                # Feature extraction
                melspec = audio_to_melspec(audio, self.sr, n_mels=self.n_mels, max_time_frames=self.max_time_frames)
                melspec = torch.tensor(melspec).float()
                query_data.append(melspec)
                query_labels.append(i)
        
        return (torch.stack(support_data), torch.tensor(support_labels)), (torch.stack(query_data), torch.tensor(query_labels))
