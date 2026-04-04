import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GroupKFold
import time
import joblib
import os
from sklearn.base import clone
from scipy.signal import medfilt
from scipy.stats import zscore

# -- Helper functions for dataset --

def extract_basic_temporal_features(df):
    '''
    Adds temporal context (lag/lead) strictly isolated within each subject's data. This prevents 'data leakage' between different patients.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing multiple subjects.

    Returns:
    - pd.DataFrame: The DataFrame with patient-isolated temporal context.
    '''
    df_new = df.copy()
    exclude_cols = ['Subject_ID', 'Dataset_Source', 'Label']
    cols_to_shift = [c for c in df_new.columns if c not in exclude_cols]

    def shift_group(group):
        for col in cols_to_shift:
            group[f'{col}_lag2'] = group[col].shift(2)
            group[f'{col}_lag1'] = group[col].shift(1)
            group[f'{col}_lead1'] = group[col].shift(-1)
            group[f'{col}_lead2'] = group[col].shift(-2)
        group.bfill(inplace=True)
        group.ffill(inplace=True)
        return group

    return df_new.groupby('Subject_ID', group_keys=False).apply(shift_group, include_groups=False)

def normalize_by_subject(df, feature_cols):
    '''
    Apply Z-Score normalization per patient (Subject_ID).
    Brings signals from different electrodes (Fpz, C3, C4) to the same common denominator.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - feature_cols (list): List of column names

    Returns:
    - pd.DataFrame: The normalized DataFrame.
    '''
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = df_norm.groupby('Subject_ID')[col].transform(zscore)

    df_norm[feature_cols] = df_norm[feature_cols].fillna(0)
    return df_norm

def extract_enhanced_temporal_features(df):
    '''
    Extract Ratios, Lags, Leads, Diffs and Rolling Stats using ultra-fast vectorized operations, respecting per-patient isolation.

    Parameters:
      - df (pd.DataFrame): The input DataFrame.

    Returns:
      - pd.DataFrame: The enhanced DataFrame.
    '''
    df_new = df.copy()
    exclude_cols = ['Subject_ID', 'Dataset_Source', 'Label']

    # 1. Creare Ratios (Rapoarte)
    eps = 1e-6
    df_new['Ratio_DeepSleep'] = df_new['Abs_Delta'] / (df_new['Abs_Beta'] + eps)
    df_new['Ratio_Transition'] = df_new['Abs_Theta'] / (df_new['Abs_Alpha'] + eps)

    feature_cols = [c for c in df_new.columns if c not in exclude_cols]
    grouped = df_new.groupby('Subject_ID')[feature_cols]

    # 2. Extragere Lags, Leads și Diferențe vectorizate
    lag1_df = grouped.shift(1).add_suffix('_lag1')
    lag2_df = grouped.shift(2).add_suffix('_lag2')
    lead1_df = grouped.shift(-1).add_suffix('_lead1')
    lead2_df = grouped.shift(-2).add_suffix('_lead2')
    diff_df = df_new[feature_cols].sub(grouped.shift(1)).add_suffix('_diff')

    # 3. Statistici Glisante (Rolling Stats)
    roll_mean_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).mean()).add_suffix('_roll_mean')
    roll_std_df = grouped.transform(lambda x: x.rolling(window=5, min_periods=1, center=True).std()).add_suffix('_roll_std')

    # 4. Concatenare și curățare NaN-uri
    df_final = pd.concat([df_new, lag1_df, lag2_df, lead1_df, lead2_df, diff_df, roll_mean_df, roll_std_df], axis=1)
    df_final = df_final.groupby('Subject_ID', group_keys=False).apply(lambda g: g.bfill().ffill(), include_groups=False)

    return df_final

## --- Helpers for models --

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

def evaluete_model(y_true, y_pred, class_names=['Wake', 'N1', 'N2', 'N3', 'REM']):
    '''
    Evaluate the performance of a classification model.

    Parameters:
      - y_true (array-like): True labels.
      - y_pred (array-like): Predicted labels.
      - class_names (list): List of class names.
    '''
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy {acc:.4f}\n")

    print("--- Classification report ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def cross_validation_model(model, X, y, model_name, save_path = '/content', n_splits = 5, sample_weights=None):
    print(f"=== Cross validation for {model_name} ({n_splits} folds) ===")

    X_arr = np.array(X)
    y_arr = np.array(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    best_accuracy = 0.0
    best_model = None
    best_fold = -1

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        # Split
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Clone the model
        cloned_model = clone(model)

        # Train
        if sample_weights is None:
            cloned_model.fit(X_train, y_train)
        else:
            #For XGBoost because we have sample_weights
            fold_sample_weights = sample_weights[train_idx]
            cloned_model.fit(X_train, y_train, sample_weight=fold_sample_weights)

        # Make predictions
        y_pred = cloned_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = cloned_model
            best_fold = fold
            print(f"Fold {fold}: Accuracy = {acc:.4f} -> New best!\n")
        else:
            print(f"Fold {fold}: Accuracy = {acc:.4f}\n")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print("\n" + "="*50)
    print(f"Final results {model_name}:")
    print(f"Mean accuracy: {mean_acc:.4f} (± {std_acc:.4f})")

    full_save_path = os.path.join(save_path, f"best_{model_name}_model.pkl")
    joblib.dump(best_model, full_save_path)
    print(f"Best model save at: {full_save_path}")
    print("="*50 + "\n")

    return best_model, mean_acc