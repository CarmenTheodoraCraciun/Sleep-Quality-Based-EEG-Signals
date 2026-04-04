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