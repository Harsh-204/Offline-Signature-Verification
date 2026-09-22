#!/usr/bin/env python3
"""Offline Signature Verification — Quick Demo.

Runs the full verification pipeline on the bundled 20-image demo
dataset (10 genuine + 10 forged from 2 CEDAR writers) using the
pre-trained model.  No full CEDAR dataset download required.

Usage:
    python run_demo.py
"""

from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DEMO_DIR = PROJECT_ROOT / "demo"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

# Ensure ``src/`` is importable even without an editable install.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Imports from the real sigverify pipeline
# ---------------------------------------------------------------------------
from sigverify.classifier import load_model, predict        # noqa: E402
from sigverify.dataset import load_cedar_dataset             # noqa: E402
from sigverify.features import extract_features              # noqa: E402
from sigverify.feature_selection import transform            # noqa: E402
from sigverify.pair_features import compute_pair_features    # noqa: E402
from sigverify.preprocessing import preprocess               # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Pre-flight checks
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("   Offline Signature Verification — Quick Demo")
    print("=" * 60)
    print()

    if not DEMO_DIR.is_dir():
        sys.exit(f"[ERROR] Demo directory not found: {DEMO_DIR}")
    if not MODEL_PATH.is_file():
        sys.exit(f"[ERROR] Trained model not found: {MODEL_PATH}")

    # ------------------------------------------------------------------
    # 2. Load trained model artifact
    # ------------------------------------------------------------------
    print(f"[+] Loading model artifact : {MODEL_PATH.name}")
    artifact = load_model(str(MODEL_PATH))
    reducer = artifact.get("reducer")
    groups = artifact.get("feature_config", {}).get("groups")
    print("[+] Model loaded successfully.")
    print()

    # ------------------------------------------------------------------
    # 3. Load demo dataset (CEDAR layout: full_org / full_forg)
    # ------------------------------------------------------------------
    print(f"[+] Loading demo dataset   : {DEMO_DIR}")
    writers = load_cedar_dataset(str(DEMO_DIR))

    total_genuine = sum(len(w.genuine_paths) for w in writers)
    total_forged = sum(len(w.forged_paths) for w in writers)
    print(f"    Writers   : {len(writers)}")
    print(f"    Genuine   : {total_genuine}")
    print(f"    Forged    : {total_forged}")
    print()

    # ------------------------------------------------------------------
    # 4. Build reference–query pairs and run inference
    # ------------------------------------------------------------------
    # For each writer we pick the FIRST genuine sample as the reference
    # and verify every other genuine + every forged sample against it.
    # ------------------------------------------------------------------
    print("-" * 60)
    print(f"  {'#':<4} {'Writer':<8} {'Type':<10} {'Verdict':<10} {'Confidence':<12} {'Image'}")
    print("-" * 60)

    pair_num = 0
    correct = 0
    total = 0

    for wd in writers:
        if not wd.genuine_paths:
            continue

        # Reference = first genuine sample for this writer
        ref_path = wd.genuine_paths[0]
        img_ref = preprocess(ref_path)
        feat_ref = extract_features(img_ref, groups=groups)
        if reducer is not None:
            import numpy as np
            feat_ref = transform(feat_ref.reshape(1, -1), reducer).flatten()

        # --- Genuine queries (skip the reference itself) ---
        for gpath in wd.genuine_paths[1:]:
            pair_num += 1
            total += 1
            img_q = preprocess(gpath)
            feat_q = extract_features(img_q, groups=groups)
            if reducer is not None:
                feat_q = transform(feat_q.reshape(1, -1), reducer).flatten()

            pair_feat = compute_pair_features(feat_ref, feat_q)
            label, confidence = predict(artifact, pair_feat)
            verdict = "GENUINE" if label == 1 else "FORGED"
            if label == 1:
                correct += 1

            fname = pathlib.Path(gpath).name
            print(f"  {pair_num:<4} {wd.writer_id:<8} {'genuine':<10} {verdict:<10} {confidence*100:>6.2f}%      {fname}")

        # --- Forged queries ---
        for fpath in wd.forged_paths:
            pair_num += 1
            total += 1
            img_q = preprocess(fpath)
            feat_q = extract_features(img_q, groups=groups)
            if reducer is not None:
                feat_q = transform(feat_q.reshape(1, -1), reducer).flatten()

            pair_feat = compute_pair_features(feat_ref, feat_q)
            label, confidence = predict(artifact, pair_feat)
            verdict = "GENUINE" if label == 1 else "FORGED"
            if label == 0:
                correct += 1

            fname = pathlib.Path(fpath).name
            print(f"  {pair_num:<4} {wd.writer_id:<8} {'forged':<10} {verdict:<10} {confidence*100:>6.2f}%      {fname}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("-" * 60)
    print()
    accuracy = correct / total * 100 if total else 0
    print(f"  Total pairs verified : {total}")
    print(f"  Correct predictions  : {correct}/{total} ({accuracy:.1f}%)")
    print()
    print("=" * 60)
    print("   Demo completed successfully.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
