#!/usr/bin/env python3
"""
Example script demonstrating how to use SplitAudioFolderDataset to load audio samples
from labeled folders and split them into train/eval/test sets.
"""

import os
from AudioFolderDataset import SplitAudioFolderDataset
from torch.utils.data import DataLoader

def main():
    # Path to directory containing labeled audio samples
    # Structure should be:
    # data/
    #   ├── label1/
    #   │   ├── sample1.wav
    #   │   ├── sample2.wav
    #   │   └── ...
    #   ├── label2/
    #   │   ├── sample1.wav
    #   │   └── ...
    #   └── ...
    data_dir = "./data"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist. Please create it and add labeled audio samples.")
        print("Expected structure: data/label1/, data/label2/, etc.")
        return
    
    # Create datasets with automatic 70/20/10 splitting
    print("Creating train/eval/test splits with 70%/20%/10% ratio...")
    train_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='train')
    eval_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='eval')
    test_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='test')
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Classes: {train_dataset.label_to_idx}")
    print(f"Train size: {len(train_dataset)} samples")
    print(f"Eval size: {len(eval_dataset)} samples")
    print(f"Test size: {len(test_dataset)} samples")
    
    total_samples = len(train_dataset) + len(eval_dataset) + len(test_dataset)
    print(f"Total samples: {total_samples}")
    
    # Calculate actual split percentages
    train_pct = len(train_dataset) / total_samples
    eval_pct = len(eval_dataset) / total_samples
    test_pct = len(test_dataset) / total_samples
    print(f"Actual split ratio: {train_pct:.1%}/{eval_pct:.1%}/{test_pct:.1%}")
    
    # Print class distribution in each split
    print("\nClass distribution:")
    for split_name, dataset in [("Train", train_dataset), ("Eval", eval_dataset), ("Test", test_dataset)]:
        class_counts = {}
        for label in dataset.labels:
            label_name = dataset.idx_to_label[label]
            class_counts[label_name] = class_counts.get(label_name, 0) + 1
        
        print(f"{split_name} split:")
        for label, count in class_counts.items():
            print(f"  - {label}: {count} samples ({count/len(dataset):.1%})")

if __name__ == "__main__":
    main()