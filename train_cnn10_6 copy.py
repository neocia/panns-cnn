import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
from AudioFolderDataset import SplitAudioFolderDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import sys

# --- Add PANNs repo path ---
panns_repo_path = '/workspace/audioset_tagging_cnn/pytorch' 
if os.path.exists(panns_repo_path) and panns_repo_path not in sys.path:
    sys.path.append(panns_repo_path)
    print(f"Added {panns_repo_path} to sys.path.")
else:
    print(f"Warning: {panns_repo_path} not found or already in sys.path. Please ensure the PANNs repository is cloned correctly.")

from models import Cnn10, Cnn6  # Suporte aos dois modelos

# --- PANNs Transfer Learning Model ---
class PodFineTunedClassifier(nn.Module):
    def __init__(self, model_type='Cnn10', num_classes=2, panns_weights_path=None,
                 freeze_panns=True, unfreeze_last_layers=0, target_panns_sample_rate=16000):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.target_panns_sample_rate = target_panns_sample_rate

        # --- Escolhe arquitetura ---
        if model_type == 'Cnn10':
            self.panns_model = Cnn10(
                    sample_rate=16000,
                    window_size=1024,
                    hop_size=320,
                    mel_bins=64,
                    fmin=50,
                    fmax=14000,
                    classes_num=527
            )
        elif model_type == 'Cnn6':
            self.panns_model = Cnn6(
                sample_rate=16000,
                window_size=1024,   # n_fft
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527
                #kernel_size=5 
            )
        else:
            raise ValueError("model_type must be 'Cnn10' or 'Cnn6'")

        self.panns_model.to(self.device)

        # --- Carregar pesos pré-treinados ---
        if panns_weights_path:
            checkpoint = torch.load(panns_weights_path, map_location='cpu')
            self.panns_model.load_state_dict(checkpoint['model'])
            print(f"Weights loaded from {panns_weights_path}")

        # --- Freeze/unfreeze ---
        for param in self.panns_model.parameters():
            param.requires_grad = False

        if not freeze_panns:
            for param in self.panns_model.parameters():
                param.requires_grad = True
            print("All PANNs layers unfrozen.")
        elif unfreeze_last_layers > 0:
            layers_to_unfreeze = []
            if unfreeze_last_layers >= 1: layers_to_unfreeze.append('conv_block5')
            if unfreeze_last_layers >= 2: layers_to_unfreeze.append('conv_block4')
            if unfreeze_last_layers >= 3: layers_to_unfreeze.append('conv_block3')
            for name, param in self.panns_model.named_parameters():
                for layer in layers_to_unfreeze:
                    if layer in name:
                        param.requires_grad = True
            print(f"PANNs layers partially unfrozen: {', '.join(layers_to_unfreeze)}")
        else:
            print("PANNs layers completely frozen.")

        # --- Classification head ---
        panns_embedding_size = self.panns_model.fc_audioset.in_features
        self.fc1 = nn.Linear(panns_embedding_size, 256).to(self.device)
        self.dropout1 = nn.Dropout(0.5).to(self.device)
        self.fc2 = nn.Linear(256, num_classes).to(self.device)

    def forward(self, waveform):
        waveform = waveform.to(self.device)
        model_output = self.panns_model(waveform)
        embedding = model_output["embedding"]
        x = F.relu(self.fc1(embedding))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# --- Audio preprocessing ---
def load_and_preprocess_raw_audio(audio_filepath, original_sr=48000, target_sr=16000):
    waveform, sr = librosa.load(audio_filepath, sr=original_sr, mono=True)
    if sr != target_sr:
        waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
    return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)

# --- Main training and evaluation ---
if __name__ == "__main__":
    coughvid_dir = "./data/coughvid_processed"
    coswara_dir = "./data/coswara_processed"
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Datasets ---
    train_dataset = SplitAudioFolderDataset(coswara_dir, target_sr=16000, split_type='train')
    eval_dataset = SplitAudioFolderDataset(coswara_dir, target_sr=16000, split_type='eval')
    test_dataset = SplitAudioFolderDataset(coughvid_dir, target_sr=16000, split_type='test')

    label_to_idx = train_dataset.label_to_idx
    idx_to_label = train_dataset.idx_to_label
    
    # Garantir compatibilidade de labels Coswara -> CoughVid
    for i, label in enumerate(test_dataset.labels):
        if label not in label_to_idx.values():
            test_dataset.labels[i] = label_to_idx['healthy']

    num_classes = len(label_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(f"Classes: {label_to_idx}")
    print(f"Train/Eval/Test sizes: {len(train_dataset)}/{len(eval_dataset)}/{len(test_dataset)}")

    # --- Modelo ---
    # Troque 'Cnn10' para 'Cnn6' se quiser usar Cnn6
    model = PodFineTunedClassifier(
        model_type='Cnn6',
        num_classes=num_classes,
        panns_weights_path='/workspace/checkpoints/Cnn6_mAP=0.343.pth',
        freeze_panns=False,   # True para congelar todos, False para treinar todos
        unfreeze_last_layers=3
    )
    device = model.device

    # --- Loss e optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'panns_model' in n and p.requires_grad], 'lr': 5e-6},
        {'params': [p for n, p in model.named_parameters() if 'panns_model' not in n], 'lr': 5e-4}
    ])

    # --- Training loop ---
    NUM_EPOCHS = 100
    best_eval_acc = 0.0
    best_model_state = None
    patience = 50
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
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
        eval_loss, eval_correct, eval_total = 0.0, 0, 0
        with torch.no_grad():
            for waveforms, labels in eval_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * waveforms.size(0)
                _, predicted = torch.max(outputs, 1)
                eval_correct += (predicted == labels).sum().item()
                eval_total += labels.size(0)
        eval_loss /= eval_total
        eval_acc = eval_correct / eval_total

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save({'model': best_model_state, 
                        'label_to_idx': label_to_idx, 
                        'idx_to_label': idx_to_label, 
                        'epoch': epoch + 1, 
                        'best_eval_acc': best_eval_acc}, 
                       f"pod_finetuned_classifier_best_csxc_cnn06.pth")
        else:
            patience_counter += 1

        if best_eval_acc == 1.0 or patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- Load best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_eval_acc:.4f}")

    torch.save({'model': model.state_dict(), 'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label, 'best_eval_acc': best_eval_acc}, f"pod_finetuned_classifier_{model.__class__.__name__}.pth")
    print(f"Best model saved as pod_finetuned_classifier_{model.__class__.__name__}.pth")

    # --- Evaluation on test set ---
    model.eval()
    all_predictions, all_labels, all_filenames, all_probs = [], [], [], []
    test_filenames = [f[0] for f in test_dataset.samples]

    with torch.no_grad():
        for i, (waveforms, labels) in enumerate(test_loader):
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
            batch_size = labels.size(0)
            batch_start_idx = i * test_loader.batch_size
            batch_end_idx = min(batch_start_idx + batch_size, len(test_dataset))
            all_filenames.extend(test_filenames[batch_start_idx:batch_end_idx])

    # --- Metrics ---
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_probs = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape==(2,2) else (0,0,0,0)
    specificity = tn / (tn + fp) if (tn + fp)>0 else 0
    tpr = tp / (tp + fn) if (tp + fn)>0 else 0
    fpr = fp / (fp + tn) if (fp + tn)>0 else 0

    print(f"\nTest Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(f"TPR: {tpr:.4f} | FPR: {fpr:.4f} | Specificity: {specificity:.4f}")

    # --- Save detailed results ---
    existing_csvs = [f for f in os.listdir(results_dir) if f.startswith("test_predictions_") and f.endswith(".csv")]
    next_index = len(existing_csvs) + 1
    csv_filename = os.path.join(results_dir, f"test_predictions_{next_index}.csv")

    results_df = pd.DataFrame({
        "filename": [os.path.basename(f) for f in all_filenames],
        "true_label": [idx_to_label[l] for l in y_true],
        "predicted_label": [idx_to_label[p] for p in y_pred],
        "prob_COVID": y_probs
    })
    results_df.to_csv(csv_filename, index=False)
    print(f"Detailed test results saved to {csv_filename}")

    # --- Metrics & confusion matrix Excel ---
    excel_filename = os.path.join(results_dir, f'test_metrics_{next_index}.xlsx')
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        metrics_dict = {
            "Accuracy": [acc],
            "Precision": [precision],
            "Recall": [recall],
            "F1": [f1],
            "AUC": [auc],
            "TPR": [tpr],
            "FPR": [fpr],
            "Specificity": [specificity]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        cm_df = pd.DataFrame(cm,
                             index=[f'True_{idx_to_label[i]}' for i in range(cm.shape[0])],
                             columns=[f'Pred_{idx_to_label[i]}' for i in range(cm.shape[1])])
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
        print(f"Metrics and confusion matrix saved to {excel_filename}")

    # --- Save predictions for further analysis ---
    np.savez(os.path.join(results_dir, f'test_predictions_{next_index}.npz'),
             predictions=y_pred, labels=y_true, filenames=all_filenames, probs=y_probs)
