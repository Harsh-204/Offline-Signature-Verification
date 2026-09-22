Overview

Offline signature verification system: given a reference (genuine) signature image and a query signature image, decide whether the query is genuine or forged, and output a similarity/confidence score. Built with classical feature engineering + a shallow classifier — deliberately not a CNN/deep-embedding approach, so every decision is explainable in a viva.

Hard constraints
Fully executable via CLI — no GUI, no notebook-only workflow as the deliverable (notebooks OK for exploration, not as the product).
CPU-only. No GPU dependency anywhere.
Modular Python package, not a single script. Minimum modules: preprocessing, feature extraction, feature selection/reduction, classifier, evaluation, CLI entrypoint.
Public GitHub repo, MIT or similar license, clean requirements.txt.
README must let someone with zero context set up and run everything from scratch.
Include automated tests (pytest) for each module, not just the end-to-end pipeline.
Dataset
CEDAR Signature Dataset (55 writers, 24 genuine + 24 forged signatures each). Publicly downloadable.
Expected local layout:
  data/
    cedar/
      full_org/          # genuine signatures, filenames like original_1_1.png
      full_forg/          # forged signatures, filenames like forgeries_1_1.png
Provide a data/README.md with the exact download source and instructions — do not assume the agent can silently fetch the dataset; the user will place it manually.
Implement both evaluation protocols, clearly separated in code and in the report:
Writer-dependent: train/test split within the same writers.
Writer-independent: some writers held out entirely for testing (the harder, more realistic setting — this should be the headline result).
Pipeline / modules
Module	Responsibility
preprocessing.py	Grayscale conversion, Otsu/adaptive binarization, noise removal, bounding-box crop to signature ink, resize/normalize to a fixed canvas, skeletonization (for structural features)
features.py	Extract: Hu moments (shape/asymmetry), HOG descriptors, Local Binary Pattern histograms (texture), pixel-density grid features (ink distribution). Combine into one feature vector per image.
feature_selection.py	PCA and/or LDA dimensionality reduction on the combined feature vector; must be fit only on training data and reused at inference time (save the fitted transform).
pair_features.py	Given a genuine-reference feature vector and a query feature vector, compute a pairwise distance/similarity feature vector (e.g., absolute difference, cosine similarity, Euclidean distance) — this is what the classifier actually sees, since verification is inherently a pair-comparison task, not single-image classification.
classifier.py	Train an SVM (with probability output) or logistic regression on pair-features to output genuine/forged + confidence. Persist the trained model + PCA/LDA transform to disk (joblib/pickle).
evaluate.py	Compute ROC curve, AUC, Equal Error Rate (EER), confusion matrix, per-writer accuracy breakdown. Save plots (ROC curve, PCA explained-variance) as PNG artifacts, not just printed numbers.
cli.py	argparse-based entrypoint with subcommands (see below).
CLI design (target commands)
bash
sigverify train --data data/cedar --protocol writer-independent --out models/model.pkl
sigverify evaluate --model models/model.pkl --data data/cedar --protocol writer-independent --out results/
sigverify verify --model models/model.pkl --reference ref.png --query query.png
train fits PCA/LDA + classifier, saves the artifact.
evaluate reproduces the full metrics report (ROC/EER/confusion matrix/per-writer table) from a saved model.
verify is the single-pair demo command for the viva — takes two images, prints a genuine/forged decision + similarity score.
Evaluation deliverables (what "done" looks like)
results/roc_curve.png, results/confusion_matrix.png
results/metrics.json — EER, AUC, accuracy, precision/recall for both protocols
results/per_writer_accuracy.csv
A short feature-ablation table: accuracy/EER with each feature group (Hu-only, HOG-only, LBP-only, all combined) — this is the single easiest thing to add that makes the project look like genuine engineering rather than "ran sklearn once."
Explicit non-goals
No pretrained CNN embeddings (e.g., no FaceNet-style siamese deep model) — the entire point is an interpretable, classical pipeline.
No web UI / Streamlit app as the primary deliverable (a small optional demo script is fine, but the CLI is the graded artifact).
signature-verification/
├── README.md
├── PROJECT_SPEC.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   └── sigverify/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── feature_selection.py
│       ├── pair_features.py
│       ├── classifier.py
│       ├── evaluate.py
│       └── cli.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_feature_selection.py
│   └── test_classifier.py
├── results/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── report/
    └── project_report.md