import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SleepDataset(Dataset):
    def __init__(self, folder_path, model_type):
        '''
        Uploads all .npz files from a specified folder and concatenates them into a single dataset.

        Parameters:
        - folder_path: path to the .npz files
        - model_type:
          - 0: returns Channels x Points (for CNN, TCN, MLP)
          - 1: returns Points x Channels (for LSTM, GRU)
        '''
        self.model_type = model_type
        if self.model_type not in [0, 1]:
            raise ValueError("model_type must be 0 or 1")

        # List all .npz files in the folder
        files = glob.glob(os.path.join(folder_path, '*.npz'))
        print(f"Load {len(files)} files from {folder_path}")

        total_epochs = 0
        for f in files:
            data = np.load(f)
            total_epochs += data['y'].shape[0]
            data.close()

        print(f"All epochs found: {total_epochs}. Memory alloc...")

        # Pre-allocate tensors for all data (Channels x Points)
        self.x_data = torch.empty((total_epochs, 2, 3000), dtype=torch.float32)
        self.y_data = torch.empty((total_epochs,), dtype=torch.long)

        # Load data into pre-allocated tensors
        current_idx = 0
        for f in files:
            data = np.load(f)
            x = data['x']  
            y = data['y']

            num_epochs = y.shape[0]

            # Load data into the pre-allocated tensors
            self.x_data[current_idx : current_idx + num_epochs] = torch.from_numpy(x).float()
            self.y_data[current_idx : current_idx + num_epochs] = torch.from_numpy(y).long()

            current_idx += num_epochs
            data.close()

        print(f"Done! X base shape: {self.x_data.shape}, Y shape: {self.y_data.shape}")

    def __len__(self):
        ''' Returns the total number of epochs in the dataset. '''
        return len(self.y_data)

    def __getitem__(self, idx):
        ''' Returns the (x, y) pair for the given index. '''
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Adapt the tensor shape based on the model type
        if self.model_type == 1:
            # LSTM/GRU expects (Points, Channels) -> (3000, 2)
            x = x.transpose(0, 1)
            
        return x, y


import time
import copy
import torch
import torch.nn as nn

def train_model_pytorch(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=5):
    '''
    Universal PyTorch training loop (CNN, LSTM, Transformer, etc.)
    Include: Early Stopping, Universal Safeguards, Gradient Clipping.
    '''
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    model = model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        
        # =========================
        # 1. FAZA DE ANTRENARE
        # =========================
        model.train()
        running_train_loss, running_train_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # --- SAFEGUARD UNIVERSAL (Ieșiri RNN/3D) ---
            if isinstance(outputs, tuple): outputs = outputs[0]
            if outputs.dim() == 3: outputs = outputs[:, -1, :]
                
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            
            # --- SAFEGUARD PENTRU EXPLODING GRADIENTS (Critic pentru LSTM/Transformers) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = running_train_corrects.double() / len(train_loader.dataset)
        
        # =========================
        # 2. FAZA DE VALIDARE
        # =========================
        model.eval()
        running_val_loss, running_val_corrects = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                
                # --- SAFEGUARD UNIVERSAL ---
                if isinstance(outputs, tuple): outputs = outputs[0]
                if outputs.dim() == 3: outputs = outputs[:, -1, :]
                    
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = running_val_corrects.double() / len(val_loader.dataset)
        
        # =========================
        # 3. AFIȘARE & EARLY STOPPING
        # =========================
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Time: {epoch_time:.0f}s | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping after {epoch+1} epochs.")
                break

    print(f"\nTrain done. The best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    
    return model
        
def evaluate_model(model, dataloader, device, class_names=['Wake', 'N1', 'N2', 'N3', 'REM']):
    model.eval()
    y_true, y_pred = [], []

    # Iterate through the dataloader and collect predictions and true labels
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Send inputs to the same device as the model
            outputs = model(inputs.to(device))

            # If the model returns a tuple (LSTM/GRU)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # If the output is 3D (CNN/MLP)
            if outputs.dim() == 3: 
                outputs = outputs[:, -1, :] 

            # Extragem clasa câștigătoare și o salvăm
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("--- Classification report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    # return y_true, y_pred

