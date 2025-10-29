#!/usr/bin/env python3
"""
Script to analyze model performance and diagnose issues with low accuracy.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from AudioFolderDataset import SplitAudioFolderDataset
import librosa

def analyze_class_distribution(data_dir):
    """Analyze the distribution of samples across classes."""
    # Load all samples
    full_dataset = SplitAudioFolderDataset(data_dir, split_type=None)
    
    # Count samples per class
    class_counts = {}
    for label in full_dataset.labels:
        label_name = full_dataset.idx_to_label[label]
        class_counts[label_name] = class_counts.get(label_name, 0) + 1
    
    # Print class distribution
    print("\nClass Distribution:")
    print("-" * 40)
    for label, count in class_counts.items():
        print(f"{label}: {count} samples ({count/len(full_dataset):.1%})")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print(f"Class distribution plot saved as 'class_distribution.png'")
    
    return class_counts

def analyze_confusion_matrix(test_dataset, predictions, labels):
    """Generate and analyze confusion matrix."""
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    # Get class names
    class_names = [test_dataset.idx_to_label[i] for i in range(len(test_dataset.idx_to_label))]
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved as 'confusion_matrix.png'")
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 40)
    print(classification_report(labels, predictions, target_names=class_names))
    
    # Identify most confused pairs
    print("\nMost Confused Class Pairs:")
    print("-" * 40)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                print(f"{class_names[i]} confused as {class_names[j]}: {cm[i, j]} times")

def analyze_audio_characteristics(data_dir, num_samples=3):
    """Analyze audio characteristics of a few samples from each class."""
    # Load dataset
    full_dataset = SplitAudioFolderDataset(data_dir, split_type=None)
    
    # Group samples by class
    samples_by_class = {}
    for i, label in enumerate(full_dataset.labels):
        label_name = full_dataset.idx_to_label[label]
        if label_name not in samples_by_class:
            samples_by_class[label_name] = []
        samples_by_class[label_name].append(full_dataset.samples[i])
    
    # Analyze a few samples from each class
    for class_name, samples in samples_by_class.items():
        print(f"\nAnalyzing {num_samples} samples from class '{class_name}':")
        for i, sample_path in enumerate(samples[:num_samples]):
            # Load audio
            y, sr = librosa.load(sample_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate audio features
            rms = np.sqrt(np.mean(y**2))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            print(f"  Sample {i+1}: {os.path.basename(sample_path)}")
            print(f"    Duration: {duration:.2f}s")
            print(f"    RMS energy: {rms:.6f}")
            print(f"    Zero crossing rate: {zcr:.6f}")
            print(f"    Spectral centroid: {spec_centroid:.2f} Hz")

def main():
    data_dir = "./data"
    
    # Analyze class distribution
    class_counts = analyze_class_distribution(data_dir)
    
    # Check if we have saved predictions
    if os.path.exists('test_predictions.npz'):
        data = np.load('test_predictions.npz')
        predictions = data['predictions']
        labels = data['labels']
        
        # Load test dataset to get class names
        test_dataset = SplitAudioFolderDataset(data_dir, split_type='test')
        
        # Analyze confusion matrix
        analyze_confusion_matrix(test_dataset, predictions, labels)
    else:
        print("\nNo saved predictions found. Run the training script first.")
    
    # Analyze audio characteristics
    analyze_audio_characteristics(data_dir)

if __name__ == "__main__":
    main()