# PANN Model Fine-Tuning for Audio Classification

## This repository is a fork of the original (https://github.com/MartinHodges/audio-pann-train  and https://github.com/qiuqiangkong/audioset_tagging_cnn)

This fine-tuning was tested on an NVIDIA GPU:: | NVIDIA-SMI 550.144.03             Driver Version: 550.144.03     CUDA Version: 12.4     |

Datasets:
- coswara:  https://zenodo.org/records/7188627
- coughvid v03:  https://zenodo.org/records/4048312



This project provides an application for fine-tuning a large-scale Pretrained Audio Neural Network (PANN) model on custom audio datasets for audio classification tasks.

## Overview

PANN models are pre-trained on the large-scale AudioSet dataset and have shown excellent performance on various audio classification tasks. This application allows you to leverage transfer learning by fine-tuning these pre-trained models on your own audio dataset.

## Features

- Fine-tune PANN models on custom audio datasets
- Automatic dataset splitting (train/eval/test)
- Support for various audio formats and sampling rates
- Configurable training parameters
- Model evaluation and performance metrics

## Directory Structure

Organize your audio files in labeled folders:

```
data/
  ├── label1/
  │   ├── snippet1.wav
  │   ├── snippet2.wav
  │   └── ...
  ├── label2/
  │   ├── snippet1.wav
  │   └── ...
  └── ...
```

## Dataset Handling

The `SplitAudioFolderDataset` class automatically loads all samples and splits them into train (70%), evaluation (20%), and test (10%) sets.

```python
from AudioFolderDataset import SplitAudioFolderDataset
from torch.utils.data import DataLoader

# Create datasets with automatic splitting
train_dataset = SplitAudioFolderDataset("./data", target_sr=16000, split_type='train')
eval_dataset = SplitAudioFolderDataset("./data", target_sr=16000, split_type='eval')
test_dataset = SplitAudioFolderDataset("./data", target_sr=16000, split_type='test')

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
```

## Fine-Tuning Process

The application uses a pre-trained PANN model and fine-tunes it on your custom dataset:

1. Loads a pre-trained PANN model (e.g., CNN14, ResNet38, etc.)
2. Replaces the final classification layer to match your number of classes
3. Freezes early layers (optional) to preserve learned features
4. Fine-tunes the model on your dataset
5. Evaluates performance on the validation set
6. Saves the fine-tuned model

The fine-tuned model is saved to `pod_finetuned_classifier.pth`

## Usage

### Training

To fine-tune the PANN model on your dataset:

```bash
source bin/activate
python train.py
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchaudio
- librosa
- numpy
- pandas
- tqdm

## References

- [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition](https://arxiv.org/abs/1912.10211)
- [Official PANN Implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn)
