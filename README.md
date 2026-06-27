# Sleep Quality Assessment Using EEG Signals

This repository implements a complete machine learning and deep learning pipeline for **sleep stage classification and sleep quality assessment** using EEG signals.  
The project was developed as part of a thesis and integrates **data cleaning**, **feature extraction**, **model experimentation**, **evaluation**, and a **Streamlit dashboard** for visualizing results.

---

## Repository Structure

| Folder | Description |
| --- | --- |
| **notebooks/ELT_feature_extraction/** | Data cleaning and feature extraction notebooks for three public EEG datasets. |
| **notebooks/Models/** | Experimental notebooks for testing architectures and hyperparameters. |
| **notebooks/Cross_Validation.ipynb** | Performs 5‑fold cross‑validation on optimal models, saving accuracy and Cohen’s kappa. |
| **notebooks/Holdout_Test.ipynb** | Tests saved models on a completely new dataset and stores accuracy, kappa, and confusion matrices. |
| **notebooks/Hypnograms_Generation.ipynb** | Generates hypnograms for selected HMC recordings and computes accuracy. |
| **dashboard/** | Streamlit application for visualizing metrics, confusion matrices, and hypnograms. |
| **dashboard/results/** | Folder required by the dashboard — contains CSV summaries and generated images. |
| **feature_extraction_pipeline.py** | Core feature extraction logic for EEG epochs. |
| **utils.py** | Helper functions for preprocessing, evaluation, and post‑processing (HMM/Viterbi). |

---

## Datasets

The project uses three open‑access EEG datasets from [PhysioNet](https://physionet.org):

| Dataset | Abbreviation | EEG Channel | Description |
| --- | --- | --- | --- |
| [Sleep‑EDFX Database](https://www.physionet.org/content/sleep-edfx/1.0.0/) | Sleep‑EDF | Fpz‑Cz | Standard sleep staging dataset with healthy subjects. |
| [University College Dublin Sleep Apnea Database](https://physionet.org/content/ucddb/1.0.0/) | UCD | C3‑A2 | Contains recordings from subjects with sleep apnea. |
| [Haaglanden Medisch Centrum Sleep Staging Database](https://physionet.org/content/hmc-sleep-staging/1.1/recordings/#files-panel) | HMC | C4‑M1 | Clinical dataset used for hypnogram generation and model validation. |

---

## Workflow Overview

1. **Data Cleaning & ETL**  
   - Performed in `ELT_feature_extraction/` notebooks.  
   - Removes artifacts, synchronizes signals, and extracts epochs.  
   - Saves processed data as `.npz` files for model training.

2. **Feature Extraction**  
   - Computes handcrafted features (time‑domain, frequency‑domain, and spectral).

3. **Model Development**  
   - Experimental notebooks in `Models/` explore classical ML, probabilistic, and DL architectures.  
   - Includes LightGBM, XGBoost, Random Forest, CNN, LSTM, MLP, ResNet, and SeqSleepNet.

4. **Cross‑Validation & Holdout Testing**  
   - `Cross_Validation.ipynb` evaluates models across folds.  
   - `Holdout_Test.ipynb` tests on unseen data and saves metrics.

5. **Hypnogram Generation**  
   - `Hypnograms_Generation.ipynb` visualizes predicted sleep stages for HMC recordings.

6. **Dashboard Visualization**  
   - Streamlit app in `dashboard/` displays metrics, confusion matrices, and hypnograms interactively.

---

## Requirements
The folder `dashboard/results/` must exist and contain:
- `model_results.csv` — summary of metrics (accuracy, kappa, generation time)
- confusion matrix images
- hypnogram images

## Notebook Summary

| Notebook | Purpose | Environment Setup |
| --- | --- | --- |
| `1_Sleep_Quality_SlpEDF_visualization.ipynb` | Visualize raw EDF signals and annotations | Google Colab |
| `2_Sleep_Quality_SlpEDF_ETL.ipynb` | ETL pipeline for Sleep‑EDFX | Google Colab |
| `3_Sleep_Quality_UCD.ipynb` | ETL pipeline for UCD dataset | Google Colab |
| `4_Sleep_Quality_HMC.ipynb` | ETL pipeline for HMC dataset | Google Colab |
| `5_Feature_extraction.ipynb` | Feature engineering and saving NPZ files | Google Colab |
| `1D_CNN_Raw.ipynb` | CNN model for raw EEG | Google Colab |
| `DL_feature_extraction.ipynb` | Deep learning on extracted features | Google Colab |
| `Hybrid.ipynb` | SeqSleepNet architecture | Ubuntu VM |
| `Probabilistic_Models.ipynb` | Logistic Regression and LDA | Google Colab |
| `Tree_based_models.ipynb` | Random Forest, XGBoost, LightGBM experiments | Google Colab |
| `Cross_Validation.ipynb` | 5‑fold evaluation and metric aggregation | Ubuntu VM |
| `Holdout_Test.ipynb` | Testing on unseen data | Ubuntu VM |
| `Hypnograms_Generation.ipynb` | Hypnogram visualization and accuracy computation | Ubuntu VM |

---

## Dashboard Setup

The dashboard was developed in **VS Code** using **Streamlit**.  
To run it:

```bash
streamlit run dashboard/main.py
```

### Output
The dashboard displays:
- model comparison charts  
- confusion matrices  
- hypnogram visualizations  

---

## Requirements

Install dependencies:

```bash
pip install -r requirments.txt
```

---

## Notes

This project is intended for research and experimentation of a master thesis. 

The notebooks document the full development process, from data cleaning and feature extraction to model evaluation and visualization.