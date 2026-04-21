"""
eval_dcg.py — Standalone evaluation of the pretrained DCG (Diffusion Classifier Guidance) model.

Computes per-class TP, TN, FP, FN and overall:
  Accuracy, Precision, Recall, F1, Sensitivity, Specificity, Kappa (quadratic),
  Balanced Accuracy — from the DCG guidance head only (no diffusion sampling).

Usage
-----
  python eval_dcg.py \
      --config  configs/aptos.yml \
      --ckpt    /path/to/aux_ckpt_best.pth \
      --testdata /path/to/aptos2019_test.pkl \
      [--traindata /path/to/aptos2019_train.pkl]  # optional, to eval train split too
      [--batch_size 32] [--device 0] [--output_dir ./dcg_eval_results]
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DCG-only evaluation for DiffMIC")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (e.g. configs/aptos.yml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to aux_ckpt_best.pth (saved as [state_dict, opt_state])")
    parser.add_argument("--testdata", type=str, default=None,
                        help="Override config testdata path (.pkl file)")
    parser.add_argument("--traindata", type=str, default=None,
                        help="Optional: also evaluate on train split")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device id (ignored if no GPU)")
    parser.add_argument("--output_dir", type=str, default="./dcg_eval_results",
                        help="Directory to save JSON / CSV results")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helper  (mirrors main.py dict2namespace)
# ---------------------------------------------------------------------------

def dict2namespace(cfg):
    ns = argparse.Namespace()
    for k, v in cfg.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, num_classes: int):
    """
    Parameters
    ----------
    y_true        : int array  (N,)
    y_pred_probs  : float array (N, C) — softmax probabilities or raw logits
    num_classes   : int

    Returns
    -------
    metrics : dict with scalar values
    detail  : str with confusion matrix + per-class TP/TN/FP/FN
    """
    y_pred = np.argmax(y_pred_probs, axis=1).astype(np.int64)
    y_true = np.asarray(y_true).astype(np.int64)

    # ---- global metrics ----
    labels = np.arange(num_classes)
    acc        = accuracy_score(y_true, y_pred)
    bal_acc    = balanced_accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro",
                                 zero_division=0, labels=labels)
    rec_macro  = recall_score(y_true, y_pred, average="macro",
                              zero_division=0, labels=labels)
    f1_macro   = f1_score(y_true, y_pred, average="macro",
                          zero_division=0, labels=labels)
    kappa      = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # per-class metrics via sklearn multilabel_confusion_matrix (one-vs-rest)
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    sens_list, spec_list = [], []
    for c in labels:
        tn, fp, fn, tp = int(mcm[c, 0, 0]), int(mcm[c, 0, 1]), int(mcm[c, 1, 0]), int(mcm[c, 1, 1])
        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # = recall for this class
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision_c  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c         = (2 * precision_c * sensitivity / (precision_c + sensitivity)
                        if (precision_c + sensitivity) > 0 else 0.0)
        per_class[int(c)] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Sensitivity": round(sensitivity, 6),
            "Specificity": round(specificity, 6),
            "Precision":   round(precision_c, 6),
            "F1":          round(f1_c,        6),
        }
        sens_list.append(sensitivity)
        spec_list.append(specificity)

    # macro sensitivity / specificity (mean over classes)
    sens_macro = float(np.mean(sens_list))
    spec_macro = float(np.mean(spec_list))

    metrics = {
        "accuracy":          round(float(acc),        6),
        "balanced_accuracy": round(float(bal_acc),    6),
        "precision_macro":   round(float(prec_macro), 6),
        "recall_macro":      round(float(rec_macro),  6),
        "f1_macro":          round(float(f1_macro),   6),
        "sensitivity_macro": round(float(sens_macro), 6),
        "specificity_macro": round(float(spec_macro), 6),
        "kappa_quadratic":   round(float(kappa),      6),
        "per_class":         per_class,
    }

    # ---- pretty detail string ----
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    lines = [
        "",
        "Confusion Matrix (rows=true, cols=pred):",
        np.array2string(cm),
        "",
        "Per-Class One-vs-Rest Metrics:",
        f"  {'Class':>6}  {'TP':>6}  {'TN':>6}  {'FP':>6}  {'FN':>6}"
        f"  {'Sens':>8}  {'Spec':>8}  {'Prec':>8}  {'F1':>8}",
    ]
    for c, m in per_class.items():
        lines.append(
            f"  {c:>6}  {m['TP']:>6}  {m['TN']:>6}  {m['FP']:>6}  {m['FN']:>6}"
            f"  {m['Sensitivity']:>8.4f}  {m['Specificity']:>8.4f}"
            f"  {m['Precision']:>8.4f}  {m['F1']:>8.4f}"
        )
    if num_classes == 2:
        lines.append(
            f"\n  Binary summary (positive=1): "
            f"TP={cm[1,1]}  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}"
        )

    detail = "\n".join(lines)
    return metrics, detail


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(model: nn.Module, loader: data.DataLoader, device, split_name: str):
    model.eval()
    all_probs  = []
    all_labels = []

    for images, labels in tqdm(loader, desc=f"Eval [{split_name}]", ncols=90):
        images = images.to(device)
        y_fusion, y_global, y_local = model(images)
        probs = y_fusion.softmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        if torch.is_tensor(labels):
            all_labels.append(labels.numpy())
        else:
            all_labels.append(np.asarray(labels))

    y_true  = np.concatenate(all_labels, axis=0).astype(np.int64)
    y_probs = np.concatenate(all_probs,  axis=0)
    return y_true, y_probs


# ---------------------------------------------------------------------------
# Result formatting & saving
# ---------------------------------------------------------------------------

SEP = "=" * 62

def format_result_block(split_name, metrics, detail_str):
    m = metrics
    block = (
        f"\n{SEP}\n"
        f"  DCG EVALUATION — {split_name.upper()}\n"
        f"{SEP}\n"
        f"  Accuracy          : {m['accuracy']*100:.2f}%\n"
        f"  Balanced Accuracy : {m['balanced_accuracy']*100:.2f}%\n"
        f"  Precision (macro) : {m['precision_macro']:.4f}\n"
        f"  Recall    (macro) : {m['recall_macro']:.4f}\n"
        f"  F1        (macro) : {m['f1_macro']:.4f}\n"
        f"  Sensitivity(macro): {m['sensitivity_macro']:.4f}\n"
        f"  Specificity(macro): {m['specificity_macro']:.4f}\n"
        f"  Kappa (quadratic) : {m['kappa_quadratic']:.4f}\n"
        f"{detail_str}\n"
        f"{SEP}"
    )
    return block


def save_results(output_dir, split_name, metrics, detail_str, block_str):
    os.makedirs(output_dir, exist_ok=True)
    tag = split_name.lower().replace(" ", "_")

    # JSON — full metrics including per-class
    json_path = os.path.join(output_dir, f"dcg_{tag}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # CSV — summary row
    csv_path = os.path.join(output_dir, f"dcg_{tag}_metrics.csv")
    header = ("split,accuracy,balanced_accuracy,precision_macro,recall_macro,"
               "f1_macro,sensitivity_macro,specificity_macro,kappa_quadratic")
    row = (
        f"{split_name},"
        f"{metrics['accuracy']:.8f},{metrics['balanced_accuracy']:.8f},"
        f"{metrics['precision_macro']:.8f},{metrics['recall_macro']:.8f},"
        f"{metrics['f1_macro']:.8f},{metrics['sensitivity_macro']:.8f},"
        f"{metrics['specificity_macro']:.8f},{metrics['kappa_quadratic']:.8f}"
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + row + "\n")

    # TXT — full human-readable report
    txt_path = os.path.join(output_dir, f"dcg_{tag}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(block_str + "\n")

    return json_path, csv_path, txt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ---- device ----
    device = (
        torch.device(f"cuda:{args.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logging.info("Using device: %s", device)

    # ---- config ----
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    config = dict2namespace(raw)

    num_classes = config.data.num_classes
    logging.info("Dataset: %s  |  num_classes: %d", config.data.dataset, num_classes)

    # ---- build DCG model ----
    # Lazy import to keep this script runnable from the repo root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pretraining.dcg import DCG

    model = DCG(config).to(device)

    # ---- load checkpoint ----
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)

    # aux_ckpt_best.pth is saved as a list [state_dict, opt_state_dict]
    if isinstance(ckpt, (list, tuple)):
        state_dict = ckpt[0]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Checkpoint loaded: %s  |  missing=%d  unexpected=%d",
        args.ckpt, len(missing), len(unexpected),
    )
    if missing:
        logging.warning("Missing keys: %s", missing[:10])
    if unexpected:
        logging.warning("Unexpected keys: %s", unexpected[:10])

    # ---- dataloader helper ----
    from dataloader.loading import APTOSDataset, BUDataset, ISICDataset

    _dataset_map = {
        "APTOS":     APTOSDataset,
        "PLACENTAL": BUDataset,
        "ISIC":      ISICDataset,
    }
    DatasetCls = _dataset_map.get(config.data.dataset.upper(), APTOSDataset)

    def make_loader(pkl_path, is_train=False):
        ds = DatasetCls(pkl_path, train=is_train)
        return data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ---- evaluate splits ----
    splits_to_eval = []

    # Test split
    test_pkl = args.testdata or getattr(config.data, "testdata", None)
    if not test_pkl or not os.path.isfile(test_pkl):
        logging.error(
            "Test data pickle not found: %s. "
            "Pass --testdata /path/to/aptos2019_test.pkl",
            test_pkl,
        )
        sys.exit(1)
    splits_to_eval.append(("test", test_pkl, False))

    # Train split (optional)
    train_pkl = args.traindata or getattr(config.data, "traindata", None)
    if args.traindata:
        if not os.path.isfile(train_pkl):
            logging.warning("Train data not found: %s — skipping.", train_pkl)
        else:
            splits_to_eval.append(("train", train_pkl, True))

    os.makedirs(args.output_dir, exist_ok=True)

    all_blocks = []
    for split_name, pkl_path, is_train in splits_to_eval:
        logging.info("\nEvaluating split: %s  (%s)", split_name, pkl_path)
        loader = make_loader(pkl_path, is_train=is_train)
        y_true, y_probs = run_evaluation(model, loader, device, split_name)

        metrics, detail_str = compute_all_metrics(y_true, y_probs, num_classes)
        block = format_result_block(split_name, metrics, detail_str)
        all_blocks.append(block)

        logging.info(block)

        j, c, t = save_results(args.output_dir, split_name, metrics, detail_str, block)
        logging.info("Saved JSON  → %s", j)
        logging.info("Saved CSV   → %s", c)
        logging.info("Saved TXT   → %s", t)

    # Combined summary file
    summary_path = os.path.join(args.output_dir, "dcg_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_blocks) + "\n")
    logging.info("\nFull summary saved → %s", summary_path)


if __name__ == "__main__":
    main()
