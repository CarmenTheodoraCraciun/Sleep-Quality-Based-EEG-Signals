# Machine Learning Systems for Sleep Quality Assessment Based on EEG Signals

**Domain:** Biomedical Signal Processing, Deep Learning in Healthcare, Medical Cybernetics, and MLOps  

---

## Project Overview

This repository implements a production-grade, end-to-end Machine Learning (ML) and Deep Learning (DL) engineering pipeline dedicated to **automated sleep stage classification and holistic sleep quality assessment using single-channel electroencephalogram (EEG) signals**. 

### Key Capabilities & Engineering Highlights
* Integrates traditional probabilistic estimators (Logistic Regression, LDA), robust tree-based ensembles (Random Forest, XGBoost, LightGBM), deep neural networks (1D-CNN on raw signals, MLP, Deep ResNet, LSTM), and sequence-to-sequence structures (SeqSleepNet).
* **Feature Engineering:** Implements mathematical extractions across the time domain, frequency domain, and spectral entropy distribution to represent macro- and micro-architectures of clinical sleep stages.
* **Interactive MLOps Dashboard:** A custom-built Streamlit analytical platform that displays comparative performance tables, multi-class confusion matrices, latency vs. accuracy Pareto optimal fronts, and dynamic patient hypnograms.

---

## Project Repository Structure

```text
├── dashboard/ # Streamlit Clinical Application Module
│   ├── main.py # Core application entry point for the dashboard UI
│   └── results/ # Persistent reporting directory (pre-compiled metrics & charts)
│       ├── model_results.csv # Aggregated validation & holdout test metrics matrix
│       ├── confusion_matrix_<model_name> # Generated multi-class confusion matrix image arrays
│       └── hypnogram_<model_name> # Serialized continuous true vs. predicted hypnogram plots
├── notebooks/ # Multi-stage Jupyter Research Workspaces
│   ├── ELT_feature_extraction/ # Extract, Load, Transform (ETL) and Signal Engineering
│   │   ├── 1_Sleep_Quality_SlpEDF_visualization.ipynb # Raw signal EDF EDA & manual annotation plots
│   │   ├── 2_Sleep_Quality_SlpEDF_ETL.ipynb # Ingestion & preprocessing for Sleep-EDFX database
│   │   ├── 3_Sleep_Quality_UCD.ipynb # Specialized cleaning for Sleep Apnea artifacts
│   │   ├── 4_Sleep_Quality_HMC.ipynb # Processing clinical sequences for validation
│   │   └── 5_Feature_extraction.ipynb # Vector engineering and .npz serialization
│   ├── Models/ # Hyperparameter Optimization & Core Architecture Training
│   │   ├── 1D_CNN_Raw.ipynb # End-to-End Deep CNN optimized for raw voltage inputs
│   │   ├── DL_feature_extraction.ipynb # MLP, LSTM, ResNet benchmarks
│   │   ├── Hybrid.ipynb # Sequence-to-Sequence SeqSleepNet execution framework
│   │   ├── Probabilistic_Models.ipynb # Linear & probabilistic statistical baseline configurations
│   │   └── Tree_based_models.ipynb # Optimized Gradient Boosting algorithms (XGBoost, LightGBM, RF)
│   ├── Cross_Validation.ipynb # Out-of-fold 5-fold evaluation & distribution auditor
│   ├── Holdout_Test.ipynb # Evaluation execution engine using untouched unseen data
│   └── Hypnograms_Generation.ipynb # Structural continuous sequence and HMM synthesis script
├── feature_extraction_pipeline.py # Production-grade mathematical extraction module
├── utils.py # System helpers, data structuring, and Viterbi sequence smoothing
└── requirements.txt # Unified environment dependency matrix configuration

```

---

## Dataset Demographics & Clinical Scope

To guarantee maximum robustness across diverse demographic cohorts and pathological deviations, the pipeline integrates three globally accepted, open-access medical databases sourced from **PhysioNet**:

| Dataset | Abbreviation | EEG Target Channel | Clinical Demographics & Pathology Focus | Role in Pipeline Structure |
| --- | --- | --- | --- | --- |
| **[Sleep-EDFX Database](https://www.physionet.org/content/sleep-edfx/1.0.0/)** | Sleep-EDF | `Fpz-Cz` | Standard healthy baseline subjects. Ideal for normal physiological sleep macrostructure modeling. | Primary Matrix for Training & 5-Fold Cross-Validation |
| **[UCD Sleep Apnea Database](https://physionet.org/content/ucddb/1.0.0/)** | UCD | `C3-A2` | Subjects exhibiting heterogeneous Sleep Apnea severities. High presence of physiological artifacts. | Pathological Generalization Auditing |
| **[HMC Sleep Staging Database](https://physionet.org/content/hmc-sleep-staging/1.1/)** | HMC | `C4-M1` | High-density real-world clinical records mapping continuous night monitoring sequences. | Full-Night Hypnogram Continuity Synthesis |

### Unified Data Partitioning Blueprint

* **Total Cohort Scope:** 372 unique subjects across the integrated databases.
* **Clinical Pathology Representation:** 175 subjects ($47.0\%$ of the entire dataset) display diagnosed sleep disorders (Apnea).
* **Model Training Allocation:** 260 subjects ($70.0\%$ training split) used for weight optimization.
* **Internal Validation Allocation:** 55 subjects ($15.0\%$ validation split) used for hyperparameter selection.
* **Strict Holdout Test Allocation:** 57 subjects ($15.0\%$ untouched split) completely isolated until final evaluation.

---

## Execution Ordering & Workflow Dependency

The system is built as a sequential dependency pipeline. To properly execute, reproduce, or audit the code, steps must follow this exact order:

```text
[Step 1: ETL Notebooks] ➔ [Step 2: Feature Extraction] ➔ [Step 3: Model Training]
                                                                  │
[Step 6: Dashboard Run] ◀ [Step 5: Hypnograms & HMM] ◀ [Step 4: Cross-Val & Holdout]

```

### Detailed Execution Manual

1. **Data Cleaning & Ingestion (`notebooks/ELT_feature_extraction/` Notebooks 1-4):**
Ingests raw European Data Format (`.edf`) signal streams, parses clinical annotations, filters high-frequency noise using bandpass filters, and splits continuous signals into distinct 30-second epochs. Output is written to local binary `.npz` storage tensors.
2. **Feature Extraction Pipeline (`notebooks/ELT_feature_extraction/5_Feature_extraction.ipynb`):**
Processes epochs using `feature_extraction_pipeline.py` to extract a multi-dimensional feature space covering:
* *Time Domain:* Mean, standard deviation, skewness, kurtosis, Hjorth mobility, and complexity.
* *Frequency Domain:* Power Spectral Density (PSD) metrics computed via Welch's method across traditional bands ($\delta$: 0.5-4Hz, $\theta$: 4-8Hz, $\alpha$: 8-12Hz, $\beta$: 12-30Hz, $\gamma$: 30-45Hz).
* *Spectral Metrics:* Spectral Entropy, spectral edge frequency, and relative power ratios.


3. **Standalone Model Training (`notebooks/Models/`):**
Can be run completely independently or concurrently. Notebooks optimize hyperparameter configurations for classical classifiers, tree boosting, and deep neural structures.
4. **Validation and Holdout Auditing (`Cross_Validation.ipynb` and `Holdout_Test.ipynb`):**
Runs a 5-fold cross-validation check on the training subset to record statistical error metrics. Next, `Holdout_Test.ipynb` must be executed to evaluate the locked model states against the untouched $15\%$ test matrix, generating final metrics and multi-class confusion matrices.
5. **Continuous Hypnogram Synthesis (`Hypnograms_Generation.ipynb`):**
Processes full-night clinical data sequentially. It passes local classifications into a Hidden Markov Model (HMM) running the Viterbi algorithm to smooth out biologically impossible transitions (such as jumping instantly from Deep Sleep N3 directly into a Wake state within a 30-second window).
6. **Dashboard UI Launch (`dashboard/main.py`):**
Launches the Streamlit framework. It pulls raw analytics dynamically from `dashboard/results/` to render all charts, tables, and comparative analyses.

---

## Analytical Benchmark Insights

The compiled dashboard provides deep engineering and scientific insights, establishing a rigorous framework for dissertation evaluation:

### 1. High-Level System Benchmarks

* **Fastest Inference Profile:** The end-to-end **1D-CNN** processes raw signals in **4.187 seconds** while maintaining a solid Test Accuracy of `0.770`.
* **Top Inter-Rater Statistical Concordance:** The Tabular Ensemble configuration (`ENS_Tab`) achieves peak structural alignment with human expert scoring, hitting a **0.706 Cohen's Kappa score**.
* **Peak Computational Efficiency:** The **1D-CNN** leads with an efficiency rating of **0.18** ($\text{Accuracy}/\text{second}$).
* **Maximum Out-of-Sample Test Accuracy:** The learned Tabular Ensemble (`ENS_Tab`) marks the highest classification performance at **0.788 accuracy**.
* **Optimal Full-Night Continuity:** **XGBoost (XGB)** shows excellent structural sequence alignment, reaching **0.823 accuracy** when evaluated across continuous full-night hypnograms.
* **Most Stable Training Variance:** The Recurrent **LSTM** network exhibits the lowest cross-validation variance, maintaining a strict standard deviation of **0.009**.

### 2. Generalization Capacity & Optimism Bias Audit

A critical focus of this thesis is checking for data leakage and overfitting:

* *Tree-Based Ensemble Robustness:* While deep models show great results during validation, tree-based boosting architectures demonstrate superior real-world generalization. On the untouched holdout dataset, **LightGBM and XGBoost outperform the rest, securing `0.781` and `0.780` test accuracy**, respectively.
* *The MLP Overfitting Case:* The multi-layer perceptron architecture shows a major **Optimism Bias**, exhibiting a drop of **0.07** in accuracy between its top cross-validation fold and the final holdout test set. Conversely, Deep ResNet and XGBoost show high generalization stability with a negligible drop ($<0.005$).

### 3. Inference Speed vs. Accuracy Front (Pareto Deployment Trade-off)

Evaluating computational latency against classification performance establishes clear clinical deployment paradigms:

* **Wearable Device Deployment Edge (<1s Budget):** The **1D-CNN** runs in **4.19s** across the entire batch, positioning it as the primary candidate for real-time edge computing on low-power wearable devices.
* **Server-Side Clinical Deployment (<10s Budget):** Tree-based models (XGB: 8.78s, LGBM: 8.85s, RF: 9.49s) and Deep ResNet (9.37s) provide an excellent balance of speed and high accuracy, fitting nicely within standard server infrastructure.
* **Batch Offline Clinical Processing:** Sequential Sleep Net (**SeqSleepNet / SSN**) requires **103.78 seconds** to compute. Due to this high temporal complexity, its use is restricted to offline clinical analysis post-recording.

### 4. Ensemble Weights

The top-performing framework (`ENS_Tab`) uses a learned gating network that distributes algorithmic importance across base models as follows:

* **LightGBM (LGBM):** $26.7\%$
* **Residual Network (ResNet):** $26.4\%$
* **XGBoost (XGB):** $15.1\%$
* **Random Forest (RF):** $11.8\%$
* **Linear Regression (LR):** $9.35\%$
* **Linear Discriminant Analysis (LDA):** $5.45\%$
* **Recurrent/Dense Stacks (LSTM, MLP):** Residual tracking components ($<5\%$)

### 5. Sleep Stage Multi-Class Confusion Analysis (Case Study: XGBoost)

An examination of individual stage classification behaviors highlights key clinical insights:

* **Wake (6,636 correct classifications) & N2 Stable Sleep (18,272 correct classifications):** These stages exhibit highly distinct electrical signatures, forming the most reliable clusters.
* **The Light Sleep N1 Challenge:** Stage N1 represents a classical medical classification problem. The model splits N1 predictions heavily across true N1 (1,463), N2 misclassifications (1,843), and Wake transitions (1,207), reflecting the highly volatile nature of transitional light sleep.
* **Deep Sleep N3 (5,058 correct) & REM Sleep (6,288 correct):** These stages demonstrate clear separation lines and minimal cross-contamination, satisfying strict clinical diagnostics criteria.

---

## Deployment & Installation Guide

### System Prerequisites

Ensure your host machine runs Python $\ge 3.9$ alongside an active environment isolation manager (`venv` or `conda`).

### Step 1: Clone Repository & Isolate Environment

### Step 2: Install Project Dependencies

```bash
pip install -r requirements.txt

```

### Step 3: Run the Streamlit Clinical Dashboard

To launch the interactive user interface using the pre-compiled evaluation artifacts, execute:

```bash
streamlit run dashboard/main.py

```

*Note: If you re-run or retrain the system using different configurations in the notebooks, the dashboard automatically updates its visualization metrics based on the newly saved CSV tables and image outputs.*

```