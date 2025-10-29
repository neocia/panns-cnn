import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import numpy as np

# --- Dataset class ---
class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, label_to_idx, target_sr=16000, set="Unknown"):
        self.samples = []
        self.labels = []
        self.label_to_idx = label_to_idx
        self.target_sr = target_sr
        print(f"Initializing {set} dataset with target sample rate: {target_sr} Hz")
        for label in os.listdir(root_dir):
            # print(f"Processing label: {label}")
            class_dir = os.path.join(root_dir, label)
            if not os.path.isdir(class_dir):
                print(f"Skipping {class_dir}, not a directory.")
                continue
            if label.startswith('.') or label.startswith('_'):
                print(f"Skipping hidden directory: {class_dir}")
                continue
            for wav_path in glob.glob(os.path.join(class_dir, "*.wav")):
                self.samples.append(wav_path)
                self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        label = self.labels[idx]
        waveform, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != self.target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        return waveform, label
        
# --- Dataset class with train/eval/test split ---
class SplitAudioFolderDataset(Dataset):
    def __init__(self, root_dir, target_sr=16000, split_type=None):
        """
        Load all samples from labeled folders and assign to train/eval/test based on split_type.
        
        Args:
            root_dir: Root directory containing labeled subdirectories
            target_sr: Target sample rate for audio
            split_type: One of 'train', 'eval', 'test', or None (for loading all data)
        """
        self.samples = []
        self.labels = []
        self.target_sr = target_sr
        
        # Discover labels
        self.label_to_idx = {}
        labels = sorted([d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.') and not d.startswith('_')])
        for idx, label in enumerate(labels):
            self.label_to_idx[label] = idx
            
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Load all samples
        all_samples = []
        all_labels = []
        
        for label in labels:
            class_dir = os.path.join(root_dir, label)
            # print(f"Processing label: {label}")
            for wav_path in glob.glob(os.path.join(class_dir, "*.wav")):
                all_samples.append(wav_path)
                all_labels.append(self.label_to_idx[label])
                
        # Create indices for train/eval/test split
        if split_type is not None:
            # Set random seed for reproducibility
            np.random.seed(42)
            indices = np.arange(len(all_samples))
            np.random.shuffle(indices)
            
            # Calculate split sizes
            train_size = int(0.7 * len(indices))
            eval_size = int(0.2 * len(indices))
            test_size = len(indices) - train_size - eval_size
            
            # Get indices for each split
            train_indices = indices[:train_size]
            eval_indices = indices[train_size:train_size + eval_size]
            test_indices = indices[train_size + eval_size:]
            
            # Assign samples based on split_type
            if split_type == 'train':
                split_indices = train_indices
                print(f"Created training split with {len(split_indices)} samples ({train_size/len(indices):.1%})")
            elif split_type == 'eval':
                split_indices = eval_indices
                print(f"Created evaluation split with {len(split_indices)} samples ({eval_size/len(indices):.1%})")
            elif split_type == 'test':
                split_indices = test_indices
                print(f"Created test split with {len(split_indices)} samples ({test_size/len(indices):.1%})")
            
            # Filter samples for this split
            self.samples = [all_samples[i] for i in split_indices]
            self.labels = [all_labels[i] for i in split_indices]
        else:
            # Use all samples
            self.samples = all_samples
            self.labels = all_labels
            
        print(f"Loaded {len(self.samples)} samples with {len(self.label_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path = self.samples[idx]
        label = self.labels[idx]
        waveform, sr = librosa.load(wav_path, sr=None, mono=True)
        if sr != self.target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        return waveform, label

# --- Discover labels from train data ---
def get_label_dicts(train_dir):
    labels = sorted([d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.') and not d.startswith('.')])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label

# --- Main training setup ---
if __name__ == "__main__":
    # Original approach with separate directories
    print("=== Original approach with separate directories ===")
    train_dir = "./data/train"
    val_dir = "./data/val"
    test_dir = "./data/test"

    label_to_idx, idx_to_label = get_label_dicts(train_dir)
    num_classes = len(label_to_idx)

    train_dataset = AudioFolderDataset(train_dir, label_to_idx, target_sr=16000, set="train")
    val_dataset = AudioFolderDataset(val_dir, label_to_idx, target_sr=16000, set="val")
    test_dataset = AudioFolderDataset(test_dir, label_to_idx, target_sr=16000, set="test")

    print(f"train_dataset size: {len(train_dataset)}")
    print(f"val_dataset size: {len(val_dataset)}")
    print(f"test_dataset size: {len(test_dataset)}")

    # New approach with programmatic splitting
    print("\n=== New approach with programmatic splitting ===")
    data_dir = "./data/all"  # Directory containing all labeled data
    
    # Create datasets with automatic splitting
    train_split_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='train')
    eval_split_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='eval')
    test_split_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='test')
    
    # Create dataloaders
    train_split_loader = DataLoader(train_split_dataset, batch_size=8, shuffle=True)
    eval_split_loader = DataLoader(eval_split_dataset, batch_size=8, shuffle=False)
    test_split_loader = DataLoader(test_split_dataset, batch_size=8, shuffle=False)
    
    print(f"Train split size: {len(train_split_dataset)}")
    print(f"Eval split size: {len(eval_split_dataset)}")
    print(f"Test split size: {len(test_split_dataset)}")
    print(f"Classes: {train_split_dataset.label_to_idx}")