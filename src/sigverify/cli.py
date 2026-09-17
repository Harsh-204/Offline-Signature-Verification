"""Command Line Interface (CLI) for sigverify.

Supports:
- sigverify train --data DIR --protocol PROTO --out PATH
- sigverify evaluate --model PATH --data DIR --protocol PROTO --out DIR
- sigverify verify --model PATH --reference REF_IMG --query QUERY_IMG
- sigverify ablation --data DIR --protocol PROTO --out DIR
"""

from __future__ import annotations

import argparse
import sys
import pathlib

from sigverify.classifier import load_model, predict, save_model, train_model
from sigverify.dataset import create_pairs, load_cedar_dataset
from sigverify.evaluate import extract_pair_features_from_pairs, run_ablation, run_evaluation
from sigverify.feature_selection import fit_reducer, transform
from sigverify.features import extract_features
from sigverify.pair_features import compute_pair_features
from sigverify.preprocessing import preprocess


def cmd_train(args: argparse.Namespace) -> None:
    """Execute training pipeline and save model artifact."""
    print(f"[+] Loading dataset from: {args.data}")
    writers = load_cedar_dataset(args.data)
    print(f"[+] Generating pairs using protocol: {args.protocol}")
    train_pairs, _ = create_pairs(writers, protocol=args.protocol)
    print(f"[+] Total training pairs generated: {len(train_pairs)}")

    # Selected feature groups
    groups = args.features.split(",") if args.features else None

    # Step 1: Preprocess and extract single-image features for training pairs
    cache: dict[str, Any] = {}
    
    # Fit reducer if requested
    reducer = None
    if args.reducer != "none":
        print(f"[+] Extracting single-signature features for fitting {args.reducer.upper()}...")
        all_paths = list(set([p.ref_path for p in train_pairs] + [p.query_path for p in train_pairs]))
        X_singles = []
        for path in all_paths:
            img = preprocess(path)
            feat = extract_features(img, groups=groups)
            cache[path] = feat
            X_singles.append(feat)
        
        X_singles_arr = np.array(X_singles)
        # Dummy single labels if needed for LDA
        y_singles_arr = np.zeros(len(X_singles_arr))
        print(f"[+] Fitting reducer ({args.reducer.upper()}, n_components={args.n_components})...")
        reducer = fit_reducer(X_singles_arr, y_singles_arr, method=args.reducer, n_components=args.n_components)

    print("[+] Building pair feature vectors...")
    X_train, y_train = extract_pair_features_from_pairs(
        train_pairs, groups=groups, reducer=reducer, feature_cache=cache
    )

    print(f"[+] Training SVM classifier on {X_train.shape[0]} pairs (feature dimension: {X_train.shape[1]})...")
    feature_config = {
        "groups": groups,
        "reducer_type": args.reducer,
        "n_components": args.n_components if args.reducer != "none" else None,
    }
    artifact = train_model(X_train, y_train, reducer=reducer, feature_config=feature_config)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(artifact, str(out_path))
    print(f"[OK] Model artifact saved successfully to: {out_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute model evaluation and save metrics/plots."""
    print(f"[+] Evaluating model: {args.model}")
    print(f"[+] Dataset: {args.data} | Protocol: {args.protocol}")
    metrics = run_evaluation(args.model, args.data, protocol=args.protocol, out_dir=args.out)

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS BREAKDOWN           ")
    print("=" * 50)
    print(f" Protocol        : {metrics['protocol']}")
    print(f" Accuracy        : {metrics['accuracy']:.4f}")
    print(f" AUC             : {metrics['auc']:.4f}")
    print(f" Equal Error Rate: {metrics['eer']:.4f}")
    print(f" Precision       : {metrics['precision']:.4f}")
    print(f" Recall          : {metrics['recall']:.4f}")
    print(f" F1-Score        : {metrics['f1_score']:.4f}")
    print("=" * 50)
    print(f"[OK] All evaluation artifacts saved to directory: {args.out}")


def cmd_verify(args: argparse.Namespace) -> None:
    """Single-pair signature verification inference."""
    print(f"[+] Loading model artifact: {args.model}")
    artifact = load_model(args.model)

    reducer = artifact.get("reducer")
    groups = artifact.get("feature_config", {}).get("groups")

    print(f"[+] Processing reference signature: {args.reference}")
    img_ref = preprocess(args.reference)
    feat_ref = extract_features(img_ref, groups=groups)

    print(f"[+] Processing query signature: {args.query}")
    img_q = preprocess(args.query)
    feat_q = extract_features(img_q, groups=groups)

    if reducer is not None:
        feat_ref = transform(feat_ref.reshape(1, -1), reducer).flatten()
        feat_q = transform(feat_q.reshape(1, -1), reducer).flatten()

    pair_feat = compute_pair_features(feat_ref, feat_q)
    label, confidence = predict(artifact, pair_feat)

    verdict = "GENUINE" if label == 1 else "FORGED"

    print("\n" + "=" * 45)
    print("        SIGNATURE VERIFICATION VERDICT        ")
    print("=" * 45)
    print(f" Reference Image : {args.reference}")
    print(f" Query Image     : {args.query}")
    print(f" Verdict         : {verdict}")
    print(f" Confidence      : {confidence * 100:.2f}%")
    print("=" * 45)


def cmd_ablation(args: argparse.Namespace) -> None:
    """Execute feature ablation study."""
    print(f"[+] Running feature ablation study on: {args.data}")
    df = run_ablation(args.data, protocol=args.protocol, out_dir=args.out)
    print("\n" + "=" * 65)
    print("                 FEATURE ABLATION STUDY                   ")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)
    print(f"[OK] Ablation comparison table saved to: {pathlib.Path(args.out) / 'ablation.csv'}")


import numpy as np
from typing import Any

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sigverify",
        description="Offline Signature Verification using Classical Feature Engineering",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # train subcommand
    p_train = subparsers.add_parser("train", help="Train verification model")
    p_train.add_argument("--data", required=True, help="Path to CEDAR dataset directory")
    p_train.add_argument(
        "--protocol",
        choices=["writer-independent", "writer-dependent"],
        default="writer-independent",
        help="Evaluation protocol split",
    )
    p_train.add_argument("--out", default="models/model.pkl", help="Output path for saved model artifact")
    p_train.add_argument("--features", default=None, help="Comma-separated feature groups (hu,hog,lbp,grid)")
    p_train.add_argument("--reducer", choices=["pca", "lda", "none"], default="none", help="Dimensionality reducer")
    p_train.add_argument("--n-components", type=int, default=50, help="Number of reducer components")
    p_train.set_defaults(func=cmd_train)

    # evaluate subcommand
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model")
    p_eval.add_argument("--model", required=True, help="Path to saved model artifact")
    p_eval.add_argument("--data", required=True, help="Path to CEDAR dataset directory")
    p_eval.add_argument(
        "--protocol",
        choices=["writer-independent", "writer-dependent"],
        default="writer-independent",
        help="Evaluation protocol split",
    )
    p_eval.add_argument("--out", default="results", help="Output directory for results and plots")
    p_eval.set_defaults(func=cmd_evaluate)

    # verify subcommand
    p_verify = subparsers.add_parser("verify", help="Verify a single signature pair")
    p_verify.add_argument("--model", required=True, help="Path to saved model artifact")
    p_verify.add_argument("--reference", required=True, help="Path to genuine reference signature image")
    p_verify.add_argument("--query", required=True, help="Path to query signature image to verify")
    p_verify.set_defaults(func=cmd_verify)

    # ablation subcommand
    p_ablation = subparsers.add_parser("ablation", help="Run feature ablation comparison study")
    p_ablation.add_argument("--data", required=True, help="Path to CEDAR dataset directory")
    p_ablation.add_argument(
        "--protocol",
        choices=["writer-independent", "writer-dependent"],
        default="writer-independent",
        help="Evaluation protocol split",
    )
    p_ablation.add_argument("--out", default="results", help="Output directory for ablation table")
    p_ablation.set_defaults(func=cmd_ablation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
