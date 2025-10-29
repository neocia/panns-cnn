import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa # For loading and resampling audio
import os
from AudioFolderDataset import SplitAudioFolderDataset
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

# Add the path to the PANNs models directory to your Python path
import sys

panns_repo_path = '/workspace/audioset_tagging_cnn/pytorch' 

if os.path.exists(panns_repo_path) and panns_repo_path not in sys.path:
    sys.path.append(panns_repo_path)
    print(f"Added {panns_repo_path} to sys.path.")
else:
    print(f"Warning: {panns_repo_path} not found or already in sys.path. "
          "Please ensure the PANNs repository is cloned correctly.")

from models import Cnn14

# --- Your PANNs Transfer Learning Model ---
class PodFineTunedClassifier(nn.Module):
    def __init__(self, num_classes, panns_weights_path, freeze_panns=True,
                 unfreeze_last_layers=0, target_panns_sample_rate=16000): # PANNs Cnn14 expects 16kHz
        super().__init__()

        # Define the device for Apple Silicon (MPS)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using Nvidia GPU (CUDA) for training and inference.")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, falling back to CPU.")

        # 1. Load the pre-trained PANNs Cnn14 model
        # The parameters here (sample_rate, window_size, etc.) are what Cnn14 was trained with.
        # Your input audio will be RESAMPLED to this rate.
        self.target_panns_sample_rate = target_panns_sample_rate
        panns_window_size = 512 # FFT window size
        panns_fmin = 10
        panns_fmax = 4000
        panns_hop_size = 160
        panns_mel_bins = 64 # This is the number of Mel frequency bins
        panns_classes_num_audioset = 527 # Number of classes PANNs was trained on (AudioSet)

        try:
            self.panns_model = Cnn14(
                sample_rate=self.target_panns_sample_rate,
                window_size=panns_window_size,
                fmin=panns_fmin,
                fmax=panns_fmax,
                hop_size=panns_hop_size,
                mel_bins=panns_mel_bins,
                classes_num=panns_classes_num_audioset)
            print("Cnn14 model instantiated successfully.")
        except Exception as e:
            print(f"Error instantiating Cnn14: {e}")
            sys.exit(1)

        print("Loading weights from pretrained model...")
        try:
            # Load the pre-trained weights
            checkpoint = torch.load(panns_weights_path, map_location='cpu', weights_only=False) # Load to CPU first
            self.panns_model.load_state_dict(checkpoint['model'])
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)

        # Move the PANNs model to the determined device (MPS or CPU)
        self.panns_model.to(self.device)

        # First freeze all parameters
        for param in self.panns_model.parameters():
            param.requires_grad = False
            
        if not freeze_panns:
            # Unfreeze all layers if freeze_panns is False
            for param in self.panns_model.parameters():
                param.requires_grad = True
            print("All PANNs model layers unfrozen for fine-tuning.")
        elif unfreeze_last_layers > 0:
            # Selectively unfreeze the last few convolutional blocks
            # Cnn14 has 6 convolutional blocks (conv_block1 to conv_block6)
            layers_to_unfreeze = []
            
            if unfreeze_last_layers >= 1:
                layers_to_unfreeze.append('conv_block6')
            if unfreeze_last_layers >= 2:
                layers_to_unfreeze.append('conv_block5')
            if unfreeze_last_layers >= 3:
                layers_to_unfreeze.append('conv_block4')
                
            # Unfreeze the selected layers
            for name, param in self.panns_model.named_parameters():
                for layer in layers_to_unfreeze:
                    if layer in name:
                        param.requires_grad = True
            
            print(f"PANNs model layers partially unfrozen: {', '.join(layers_to_unfreeze)}")
        else:
            print("PANNs model layers completely frozen.")

        # 2. Determine the embedding size from the PANNs model
        # Cnn14's final classification layer is 'fc_audioset'
        panns_embedding_size = self.panns_model.fc_audioset.in_features # This should be 2048 for Cnn14

        # 3. Define your custom classification head
        self.fc1 = nn.Linear(panns_embedding_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        # Move custom head layers to the determined device
        self.fc1.to(self.device)
        self.dropout1.to(self.device)
        self.fc2.to(self.device)


    def forward(self, waveform):
        # waveform input is (batch_size, raw_audio_samples)
        # It's your 48kHz audio. The PANNs model will handle resampling internally.

        # Ensure input waveform is on the correct device
        waveform = waveform.to(self.device)

        # print(f"DEBUG: Input waveform shape to PANNs model: {waveform.shape}")
        # print(f"DEBUG: Input waveform dtype to PANNs model: {waveform.dtype}")

        # Pass waveform through the PANNs model
        # The forward method returns (clipwise_output, embedding)
        # We need the 'embedding' which is the 2048-dimensional feature vector per clip.
        # Remove torch.no_grad() to allow gradients to flow if layers are unfrozen
        model_output = self.panns_model(waveform)

        embedding = model_output["embedding"]  # Get the embedding (2048-dim vector)
        # Now, pass the embedding to your custom classification head
        x = F.relu(self.fc1(embedding))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# --- Audio Preprocessing Function (for raw WAV files) ---
def load_and_preprocess_raw_audio(audio_filepath, original_sr=48000, target_sr=32000):
    """
    Loads an audio file and resamples it to the target_sr for PANNs.
    PANNs Cnn14 expects 16000 Hz.
    """
    # Librosa loads audio as float32 in range [-1.0, 1.0]
    waveform, sr = librosa.load(audio_filepath, sr=original_sr, mono=True)

    # Resample if original_sr is different from target_sr
    if sr != target_sr:
        waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)

    # Convert to PyTorch tensor and add a batch dimension
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    # Shape: (1, num_samples_after_resampling)
    return waveform_tensor

# --- Modified Test Function for Raw WAV ---
def test_panns_raw_wav(audio_filepath,
                       trained_classifier_path='./panns_classifier_head.pth',
                       panns_base_weights_path='/workspace/checkpoints/Cnn14_16k_mAP=0.438.pth'):

    # IMPORTANT: Replace with YOUR actual labels
    # This should match the order you used during training.
    label_to_idx = {'healthy': 0, 'COVID': 1}
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # 1. Preprocess the raw audio (resampling handles by `load_and_preprocess_raw_audio`)
    # The PANNs Cnn14 model was trained with a sample rate of 16000 Hz.
    # We will resample your 48000 Hz audio to 16000 Hz.
    input_waveform_tensor = load_and_preprocess_raw_audio(audio_filepath,
                                                          original_sr=48000,
                                                          target_sr=16000)

    # 2. Instantiate your full transfer learning model
    num_classes = len(label_to_idx)
    # When loading for inference, we want to ensure panns_model is frozen (default)
    model = PannsAudioClassifier(num_classes=num_classes,
                                 panns_weights_path=panns_base_weights_path,
                                 freeze_panns=True)

    # Load the state_dict for your custom classification head (fc1, fc2)
    # Ensure this path points to the file where you saved ONLY your custom head's weights.
    # If you saved the entire model's state_dict, load it directly:
    # model.load_state_dict(torch.load(trained_classifier_path, map_location='cpu'))
    # If you saved just the head's weights (recommended for fine-tuning setups):
    model.load_state_dict(torch.load(trained_classifier_path, map_location=model.device))

    model.eval() # Set the entire model to evaluation mode

    # 3. Predict
    with torch.no_grad(): # Disable gradient calculations for inference
        output = model(input_waveform_tensor) # Pass the preprocessed waveform
        pred_idx = output.argmax(dim=1).item()
        pred_label = idx_to_label[pred_idx]
        
        # Optionally, get probabilities
        probabilities = F.softmax(output, dim=1)[0]
        confidence = probabilities[pred_idx].item()

        print(f"Predicted label: {pred_label} (index {pred_idx})")
        print(f"Confidence: {confidence:.4f}")

# --- Example Usage ---
if __name__ == "__main__":
    # Use a single data directory with labeled samples
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    data_dir = "./data"
    
    # Create datasets with automatic 70/20/10 splitting
    train_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='train')
    eval_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='eval')
    test_dataset = SplitAudioFolderDataset(data_dir, target_sr=16000, split_type='test')
    
    # Get label mappings from the dataset
    label_to_idx = train_dataset.label_to_idx
    idx_to_label = train_dataset.idx_to_label
    num_classes = len(label_to_idx)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Classes: {label_to_idx}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(eval_dataset) + len(test_dataset)}")
    print(f"Split ratio: {len(train_dataset):.1%}/{len(eval_dataset):.1%}/{len(test_dataset):.1%}")
    

    # Model, loss, optimizer
    model = PodFineTunedClassifier(
        num_classes=2,  #num_classes, 
        panns_weights_path='/workspace/checkpoints/Cnn14_16k_mAP=0.438.pth', 
        freeze_panns=True,
        unfreeze_last_layers=3  # Unfreeze the last 3 convolutional blocks (conv_block5, conv_block6, and conv_block7)
    )
    device = model.device
    # Single label vector output
    criterion = nn.CrossEntropyLoss()
    # Learning rates
    pretrained_lr = 5e-6  # Lower learning rate for pre-trained layers (was 1e-5)
    head_lr = 5e-4        # Higher learning rate for new layers (was 1e-4)
    
    # Use different learning rates for pre-trained layers and new layers
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'panns_model' in n and p.requires_grad], 'lr': pretrained_lr},
        {'params': [p for n, p in model.named_parameters() if 'panns_model' not in n], 'lr': head_lr}
    ])
    
    print(f"Learning rates: {pretrained_lr} for pre-trained layers, {head_lr} for classification head")

    # Training loop
    NUM_EPOCHS = 100  # Increased from 50 to 100 for better convergence
    best_eval_acc = 0.0
    best_model_state = None
    patience = 50  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * waveforms.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for waveforms, labels in eval_loader:
                waveforms = waveforms.to(device)
                labels = labels.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * waveforms.size(0)
                _, predicted = torch.max(outputs, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_total += labels.size(0)
        eval_loss = eval_loss / eval_total
        eval_acc = eval_correct / eval_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
        
        # Save the best model
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model with validation accuracy: {best_eval_acc:.4f}")
            
            # Save the best model so far
            torch.save(
            {
              'model': best_model_state,
              'label_to_idx': label_to_idx,
              'idx_to_label': idx_to_label,
              'epoch': epoch + 1,
              'best_eval_acc': best_eval_acc
            },
            "pod_finetuned_classifier_best.pth")
        else:
            patience_counter += 1
            
        # Early stopping
        if best_eval_acc == 1.0:
            print(f"Early stopping triggered after {epoch+1} epochs. as 100% accuracy reached.")
            break
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {patience} epochs.")
            break

    # Load the best model for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_eval_acc:.4f}")
    
    # Save the final model (which is now the best model)
    torch.save(
    {
      'model': model.state_dict(),
      'label_to_idx': label_to_idx,
      'idx_to_label': idx_to_label,
      'best_eval_acc': best_eval_acc
    },
    ##ajustar nome do checkpoint aqui ##########################################################
    "pod_finetuned_classifier.pth")
    print("Best model saved as pod_finetuned_classifier.pth")
    
    # Run test set evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    all_filenames = []
    
    # Get filenames from test dataset
    test_filenames = test_dataset.samples
    
    with torch.no_grad():
        for i, (waveforms, labels) in enumerate(test_loader):
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get filenames for this batch
            batch_size = labels.size(0)
            batch_start_idx = i * test_loader.batch_size
            batch_end_idx = min(batch_start_idx + batch_size, len(test_dataset))
            all_filenames.extend(test_filenames[batch_start_idx:batch_end_idx])
            
            test_correct += (predicted == labels).sum().item()
            test_total += batch_size
    
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    
    # Print detailed results
    print("\nDetailed Test Results:")
    print("-" * 80)
    print(f"{'Actual':<15}{'Predicted':<15}{'Correct?':<10}{'Filename':<40}")
    print("-" * 80)
    
    # Count misclassifications by class
    class_errors = {}
    for i in range(len(all_labels)):
        actual_label = idx_to_label[all_labels[i]]
        pred_label = idx_to_label[all_predictions[i]]
        is_correct = "✓" if all_labels[i] == all_predictions[i] else "✗"
        filename = os.path.basename(all_filenames[i])
        print(f"{actual_label:<15}{pred_label:<15}{is_correct:<10}{filename:<40}")
        
        # Track errors by class
        if all_labels[i] != all_predictions[i]:
            if actual_label not in class_errors:
                class_errors[actual_label] = 0
            class_errors[actual_label] += 1
    
    print("-" * 80)
    
    # Print error summary by class
    print("\nErrors by class:")
    for label, count in class_errors.items():
        total_in_class = sum(1 for l in all_labels if idx_to_label[l] == label)
        error_rate = count / total_in_class if total_in_class > 0 else 0
        print(f"{label}: {count}/{total_in_class} errors ({error_rate:.1%})")
    
    # Save predictions for further analysis
    np.savez('test_predictions.npz',
        predictions=np.array(all_predictions),
        labels=np.array(all_labels),
        filenames=np.array(all_filenames)
    )
    
    

