import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from scipy.signal import medfilt, resample
from tensorflow.keras import layers
import keras
import tensorflow as tf

# -- Helper functions for dataset --
def build_sequences_per_subject(df, feature_cols, label_col='Label', subject_col='Subject_ID', seq_len=30, step=5):
    '''
    Build sequences of features and labels for each subject in the DataFrame.
    
    Parameters:
      - df (pd.DataFrame): Input DataFrame containing features, labels, and subject IDs
      - feature_cols (list): List of column names to be used as features.
      - label_col (str): Column name for the labels.
      - subject_col (str): Column name for the subject IDs.
      - seq_len (int): Length of the sequences to be created.
      - step (int): Step size for moving the window to create sequences.
    
    Returns:
      - X_seq (np.array): Array of feature sequences.
    '''
    X_seq, y_seq, subj_seq = [], [], []
    
    for subj, group in df.groupby(subject_col):
        group = group.sort_index()
        feats = group[feature_cols].values
        labels = group[label_col].values
        
        for i in range(0, len(feats) - seq_len, step):
            X_seq.append(feats[i:i+seq_len])
            y_seq.append(labels[i+seq_len-1])
            subj_seq.append(subj)
            
    return np.array(X_seq), np.array(y_seq), np.array(subj_seq)


class SleepDataGenerator(tf.keras.utils.PyDataset):
    def __init__(self, directory, is_training=True,batch_size=32,
        n_classes=5,window_size=3,use_context=False,
        use_lag_lead=False,downsample=False,target_length=1500,
                     sampling_rate=50,lag_seconds=0.5,select_channels=None,**kwargs):
        '''Custom data generator for sleep stage classification.'''
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.is_training = is_training
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.npz')]
        self.on_epoch_end()

        self.use_context = use_context
        self.use_lag_lead = use_lag_lead
        self.downsample = downsample
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.lag_steps = int(lag_seconds * sampling_rate)
        self.select_channels = select_channels
        if self.use_context:
            self.window_size = window_size
        else:
            self.window_size = 1

    def __len__(self):
        '''Return the number of batches per epoch.'''
        return len(self.file_list)

    def _normalize(self, X_batch):
        '''Normalize each epoch independently to zero mean and unit variance.'''
        mean = np.mean(X_batch, axis=1, keepdims=True)
        std  = np.std(X_batch, axis=1, keepdims=True)
        return (X_batch - mean) / (std + 1e-8)

    def _add_lag_lead(self, X_batch):
        '''Add lag and lead features to the input batch.'''
        if self.use_lag_lead:
            # lag/lead in time domain
            lag  = np.roll(X_batch,  self.lag_steps,  axis=1)
            lead = np.roll(X_batch, -self.lag_steps, axis=1)
            X_batch = np.concatenate([X_batch, lag, lead], axis=-1)
        return X_batch

    def __getitem__(self, index):
        '''Generate one batch of data.'''
        file_path = os.path.join(self.directory, self.file_list[index])
        with np.load(file_path, allow_pickle=True) as data:
            X_raw = data['x'] if 'x' in data else data['arr_0']
            y_raw = data['y'] if 'y' in data else data['arr_1']

        X = X_raw.copy()
        y = y_raw.copy()

        if len(X.shape) == 3 and X.shape[-1] == 3000 and X.shape[1] == 2:
            X = np.transpose(X, (0, 2, 1))

        if self.downsample:
            X_resampled = np.zeros((X.shape[0], self.target_length, X.shape[2]), dtype=X.dtype)
            for i in range(X.shape[0]):
                for ch in range(X.shape[2]):
                    X_resampled[i, :, ch] = resample(X[i, :, ch], self.target_length)
            X = X_resampled

        if self.use_context:
            return self._get_context_batch(X, y)
        else:
            return self._get_single_batch(X, y)

    def _get_single_batch(self, X, y):
        '''Get a batch of data without context.'''
        if self.is_training:
            idx = np.random.choice(X.shape[0], min(self.batch_size, X.shape[0]), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
        else:
            X_batch = X
            y_batch = y

        X_batch = self._normalize(X_batch)
        X_batch = self._add_lag_lead(X_batch)
        
        if self.select_channels is not None:
            X_batch = X_batch[..., self.select_channels]

        return X_batch, tf.keras.utils.to_categorical(y_batch, num_classes=self.n_classes)

    def _get_context_batch(self, X, y):
        '''Get a batch of data with context.'''
        num_epochs = X.shape[0]
        X_windows, y_windows = [], []

        for i in range(self.window_size - 1, num_epochs):
            window = X[i - self.window_size + 1 : i + 1]
            window_concat = np.concatenate(window, axis=0)
            X_windows.append(window_concat)
            y_windows.append(y[i])

        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)

        if self.is_training:
            idx = np.random.choice(X_windows.shape[0], min(self.batch_size, X_windows.shape[0]), replace=False)
            X_batch = X_windows[idx]
            y_batch = y_windows[idx]
        else:
            X_batch = X_windows
            y_batch = y_windows

        X_batch = self._normalize(X_batch)
        X_batch = self._add_lag_lead(X_batch)

        if self.select_channels is not None:
            X_batch = X_batch[..., self.select_channels]
        
        return X_batch, tf.keras.utils.to_categorical(y_batch, num_classes=self.n_classes)

    def on_epoch_end(self):
        '''Shuffle the file list at the end of each epoch if in training mode.'''
        if self.is_training:
            np.random.shuffle(self.file_list)

@keras.saving.register_keras_serializable(package="utils")
class Attention(layers.Layer):
    def __init__(self, units, **kwargs):
        # AICI e cheia: primește trainable, dtype, name etc.
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, inputs):
        score = self.V(keras.activations.tanh(self.W(inputs)))
        weights = keras.activations.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)


    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    
## --- Helpers for models --
def categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_pred = ops.clip(y_pred, 1e-7, 1.0)
        ce = -ops.sum(y_true * ops.log(y_pred), axis=-1)
        p_t = ops.sum(y_true * y_pred, axis=-1)
        modulating = (1.0 - p_t) ** gamma

        if alpha is not None:
            alpha_t = ops.sum(y_true * ops.convert_to_tensor(alpha), axis=-1)
            fl = alpha_t * modulating * ce
        else:
            fl = modulating * ce

        return ops.mean(fl)
    return loss

def se_feature_attention(inputs, reduction=8):
    '''
    Apply Squeeze-and-Excitation (SE) feature attention mechanism to the input tensor.
    
    Parameters:
      - inputs (tf.Tensor): Input tensor of shape (batch_size, time_steps, features).
      - reduction (int): Reduction ratio for the SE block.
    
    Returns:
      - tf.Tensor: Output tensor after applying SE attention, with the same shape as inputs.
    '''
    n = inputs.shape[-1]

    x = layers.Dense(n // reduction, activation='relu')(inputs)
    x = layers.Dense(n, activation='sigmoid')(x)

    return layers.Multiply()([inputs, x])

def compute_transition_matrix_adaptive(y, temps):
    A = np.ones((5, 5)) * 1e-3
    for i in range(len(y) - 1):
        A[y[i], y[i+1]] += 1

    A = A / A.sum(axis=1, keepdims=True)

    # Temperatures per clas
    for c in range(5):
        A[c] = A[c] ** temps[c]
        A[c] = A[c] / A[c].sum()

    return A

def viterbi(proba, A):
    T, C = proba.shape
    logA = np.log(A + 1e-12)
    logP = np.log(np.clip(proba, 1e-8, 1.0))

    delta = np.zeros((T, C))
    psi = np.zeros((T, C), dtype=int)

    delta[0] = logP[0]

    for t in range(1, T):
        scores = delta[t-1][:, None] + logA
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + logP[t]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states

def safe_medfilt(seq, center, k):
    half = k // 2
    start = max(0, center - half)
    end = min(len(seq), center + half + 1)
    window = seq[start:end]
    if len(window) < k:
        k = len(window) if len(window) % 2 == 1 else len(window) - 1
        if k < 1:
            return seq[center]
    return medfilt(window, kernel_size=k)[len(window)//2]

def adaptive_median(seq):
    out = seq.copy()
    for i in range(len(seq)):
        c = seq[i]
        if c == 1:      # N1
            out[i] = safe_medfilt(seq, i, 3)
        elif c == 2:    # N2
            out[i] = safe_medfilt(seq, i, 5)
        elif c == 3:    # N3
            out[i] = safe_medfilt(seq, i, 9)
        elif c == 4:    # REM
            out[i] = safe_medfilt(seq, i, 5)
        else:           # Wake
            out[i] = safe_medfilt(seq, i, 5)
    return out

def physiologic_rules(seq):
    out = seq.copy()
    for i in range(1, len(seq)-1):
        prev, cur, nxt = seq[i-1], seq[i], seq[i+1]

        # N1 -> N3 -> N1 => N3 become N2
        if prev == 1 and cur == 3 and nxt == 1:
            out[i] = 2

        # REM -> N2 -> REM => N2 become REM
        if prev == 4 and cur == 2 and nxt == 4:
            out[i] = 4

        # N1 -> N3 direct => N1 become N2
        if prev == 1 and nxt == 3 and cur == 1:
            out[i] = 2

        # N3 -> Wake -> N3 => Wake become N2
        if prev == 3 and cur == 0 and nxt == 3:
            out[i] = 2

    return out
    
def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            mode='max'
        )
    ]

def make_callbacks_seqsleepnet():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            mode='max'
        )
    ]

def train_model(model, X_train, y_train, X_val, sample_weight=None):
    '''
    Train and validate a machine learning model.

    Parameters:
      - model: The machine learning model to train.
      - X_train (array): Training features.
      - y_train (array): Training labels.
      - X_val (array): Validation features.
      - sample_weight (array, optional): Sample weights for training.

    Returns:
      - array: Predicted labels for the validation set.
    '''
    start_time_fit = time.time()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    print(f"Train time: {(time.time() - start_time_fit):.2f} seconds\n")

    return model.predict(X_val)

def create_evaluation_table(y_val, y_pred, df_val):
  '''
  Create a table to evaluate the model's performance.

  Parameters:
    - y_val (array): True labels.
    - y_pred (array): Predicted labels.
    - df_val (pd.DataFrame, optional): DataFrame containing subject IDs and labels.

  Returns:
    - pd.DataFrame: A table with subject IDs, true labels, and predicted labels.
  '''
  df_eval = pd.DataFrame({
      'Subject_ID': df_val['Subject_ID'],
      'Label': y_val,
      'Pred_Raw': y_pred
  })

  df_eval['Pred_Smooth'] = df_eval.groupby('Subject_ID')['Pred_Raw'].transform(
      lambda x: medfilt(x, kernel_size=3).astype(int)
      )

  return df_eval

def evaluate_model(
        y_true, 
        y_pred, 
        class_names=['Wake', 'N1', 'N2', 'N3', 'REM'],
        disable_view=True,
        save_folder=None,
        model_name=None
    ):

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    if disable_view:
        cm = confusion_matrix(y_true, y_pred)
    
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
    
        # Save BEFORE show/close
        save_path = None
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"confusion_matrix_{model_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved at: {save_path}")
        
        print(f"Accuracy: {acc:.4f}")
        print("--- Classification report ---")
        print(classification_report(y_true, y_pred, target_names=class_names))
        plt.show()
        print(f"Cohen's Kappa: {kappa:.4f}")

        return acc, kappa, save_path

    return acc, kappa
