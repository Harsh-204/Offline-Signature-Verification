# Offline Signature Verification System

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

An interpretable, classical computer vision signature verification system built with explicit feature engineering (**Hu Moments**, **HOG**, **LBP**, and **Pixel-Density Grids**) paired with PCA/LDA dimensionality reduction and a Support Vector Machine (SVM) binary pair-classifier.

> [!NOTE]
> **No Deep Learning / No GPU**: Designed specifically for explainable biometric verification on CPU-only hardware.

---

## ⚡ Quick Demo

**No full CEDAR dataset download required.** This repository includes:
- A **pre-trained model** (`models/model.pkl`)
- A **bundled demo set** of 20 real CEDAR signature images (10 genuine + 10 forged, from 2 writers) in `demo/`

### 1. Install

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run the Demo

```bash
python run_demo.py
```

The demo loads the trained model, processes each demo signature through the **real** preprocessing → feature extraction → pair comparison → SVM classification pipeline, and prints per-pair verdicts:

```
============================================================
   Offline Signature Verification — Quick Demo
============================================================

[+] Loading model artifact : model.pkl
[+] Model loaded successfully.

[+] Loading demo dataset   : demo
    Writers   : 2
    Genuine   : 10
    Forged    : 10

------------------------------------------------------------
  #    Writer   Type       Verdict    Confidence   Image
------------------------------------------------------------
  1    1        genuine    GENUINE     92.14%      original_1_2.png
  2    1        forged     FORGED      87.30%      forgeries_1_1.png
  ...
------------------------------------------------------------

  Demo completed successfully.
```

---

## 🚀 Full Environment Setup

### 1. Prerequisites
- Python 3.9+ installed on your system.
- Git (optional, for cloning).

### 2. Virtual Environment & Installation
Clone or navigate to the repository directory, create a virtual environment, and install `sigverify` in editable mode:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify installation:
```bash
sigverify --help
```

---

## 📁 Dataset Placement (Full Training / Evaluation)

> [!IMPORTANT]
> The complete CEDAR dataset is **only** required for full training, benchmark evaluation, and ablation experiments. It is **not** needed for the quick demo above.

Download the **CEDAR Signature Dataset** (55 writers, 24 genuine + 24 forged signatures each) and place it under `data/cedar/` as described in [`data/README.md`](data/README.md):

```
data/
  cedar/
    full_org/           # genuine signatures (e.g., original_1_1.png)
    full_forg/          # forged signatures (e.g., forgeries_1_1.png)
```

The `data/cedar/` directory is excluded from Git via `.gitignore` to keep the repository lightweight.

---

## 💻 CLI Usage Guide

The `sigverify` package provides four primary subcommands: `train`, `evaluate`, `verify`, and `ablation`.

### 1. Train Model
Fits feature extraction, PCA dimensionality reduction, and the SVM pair-classifier. Persists the complete model artifact to `models/model.pkl`.

```bash
sigverify train --data data/cedar --protocol writer-independent --out models/model.pkl
```

### 2. Evaluate Model
Reproduces evaluation metrics (EER, AUC, Accuracy, Precision, Recall, F1) and generates visualization artifacts (`roc_curve.png`, `confusion_matrix.png`, `metrics.json`, `per_writer_accuracy.csv`).

```bash
sigverify evaluate --model models/model.pkl --data data/cedar --protocol writer-independent --out results/
```

### 3. Verify a Single Signature Pair
Performs inference on a single pair of images (one genuine reference and one query image to verify).

```bash
sigverify verify --model models/model.pkl --reference demo/full_org/original_1_1.png --query demo/full_forg/forgeries_1_1.png
```

#### Example Output:
```
=============================================
        SIGNATURE VERIFICATION VERDICT        
=============================================
 Reference Image : demo/full_org/original_1_1.png
 Query Image     : demo/full_forg/forgeries_1_1.png
 Verdict         : FORGED
 Confidence      : 94.20%
=============================================
```

### 4. Run Feature Ablation Study
Generates a comparative ablation table across individual feature groups (Hu-only, HOG-only, LBP-only, Grid-only, and Combined) saved as `results/ablation.csv`.

```bash
sigverify ablation --data data/cedar --protocol writer-independent --out results/
```

---

## 🧪 Running Automated Tests

Run the full pytest suite:

```bash
pytest -v
```

---

## 🏛️ Project Structure

```
Offline-Signature-Verification/
├── README.md
├── PROJECT_SPEC.md
├── pyproject.toml
├── requirements.txt
├── run_demo.py                  # ← Quick demo entry point
├── .gitignore
├── data/
│   └── README.md                # Dataset placement instructions
├── demo/                        # ← Bundled demo images (20 total)
│   ├── full_org/                #    10 genuine signatures
│   └── full_forg/               #    10 forged signatures
├── src/
│   └── sigverify/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── feature_selection.py
│       ├── pair_features.py
│       ├── dataset.py
│       ├── classifier.py
│       ├── evaluate.py
│       └── cli.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_feature_selection.py
│   ├── test_pair_features.py
│   ├── test_dataset.py
│   ├── test_classifier.py
│   └── test_evaluate.py
├── results/                     # Evaluation outputs (metrics, plots)
├── models/
│   └── model.pkl                # Pre-trained SVM model (~10 MB)
└── report/
    └── project_report.md
```
