"""
FPGA Star Tracker - Geometric Feature Trainer
===============================================
Extracts hand-crafted spatial features from starfield images,
trains a classifier, and benchmarks it for FPGA comparison.

No CNN — features are computed directly from star pixel positions,
making them fast, interpretable, and FPGA-friendly.

Feature vector (12 values per image):
  [0]  top_ratio          - fraction of stars in top half
  [1]  bottom_ratio       - fraction of stars in bottom half
  [2]  left_ratio         - fraction of stars in left half
  [3]  right_ratio        - fraction of stars in right half
  [4]  centroid_x         - normalized horizontal centroid (0-1)
  [5]  centroid_y         - normalized vertical centroid (0-1)
  [6]  mean_radial_dist   - mean distance from center / half-diagonal
  [7]  star_density       - normalized star count
  [8]  center_density     - fraction of stars within 25% of half-diag
  [9]  edge_density       - fraction of stars beyond 75% of half-diag
  [10] std_x              - normalized horizontal spread
  [11] std_y              - normalized vertical spread

Usage:
    python train_classifier.py

Outputs:
    star_tracker_model.pkl     - trained model (joblib)
    feature_scaler.pkl         - fitted StandardScaler
    benchmark_report.txt       - accuracy + timing for FPGA comparison
    training_curves.png        - per-class accuracy breakdown
    confusion_matrix.png       - confusion matrix heatmap
"""

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR = "starfield_dataset"
IMG_W       = 320
IMG_H       = 240
CLASSES     = ["up", "down", "left", "right", "forward", "back"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

# Star detection
STAR_THRESHOLD = 100   # pixel brightness threshold (0-255); tune if needed

# Which classifier to use: "logreg" | "rf" | "gbm"
# Start with "logreg" — if accuracy < 90%, switch to "rf"
CLASSIFIER = "rf"


# ─────────────────────────────────────────────
# 1. STAR DETECTION
# ─────────────────────────────────────────────

def detect_stars(img_array):
    """
    Given a grayscale image array (H, W) with values 0-255,
    return an (N, 2) array of (x, y) pixel coordinates where
    brightness > STAR_THRESHOLD.

    This is the same logic you'd implement in FPGA HDL:
    just a brightness comparator scanning pixel-by-pixel.
    """
    ys, xs = np.where(img_array > STAR_THRESHOLD)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(img_array):
    """
    Compute the 12-element geometric feature vector for one image.
    All values are normalized to [0, 1] range.

    Args:
        img_array : np.ndarray shape (H, W), dtype uint8 or float

    Returns:
        np.ndarray shape (12,), dtype float32
    """
    stars = detect_stars(img_array)
    n = len(stars)

    # Edge case: blank or near-blank image
    if n == 0:
        return np.zeros(12, dtype=np.float32)

    xs = stars[:, 0]   # x coords (horizontal)
    ys = stars[:, 1]   # y coords (vertical)

    cx_img = IMG_W / 2.0
    cy_img = IMG_H / 2.0
    half_diag = np.sqrt(cx_img**2 + cy_img**2)

    # ── Directional density ──────────────────
    top_ratio    = np.sum(ys < cy_img) / n
    bottom_ratio = 1.0 - top_ratio
    left_ratio   = np.sum(xs < cx_img) / n
    right_ratio  = 1.0 - left_ratio

    # ── Centroid ────────────────────────────
    centroid_x = np.mean(xs) / IMG_W   # 0 = far left,  1 = far right
    centroid_y = np.mean(ys) / IMG_H   # 0 = top,       1 = bottom

    # ── Radial distribution ──────────────────
    dx = xs - cx_img
    dy = ys - cy_img
    radial_dists = np.sqrt(dx**2 + dy**2)
    mean_radial = np.mean(radial_dists) / half_diag   # 0=center, 1=corner

    # ── Density features (forward/back) ─────
    # Stars within inner 25% of half-diagonal → "back" signal
    # Stars beyond outer 75% of half-diagonal → "forward" signal
    inner_thresh = 0.25 * half_diag
    outer_thresh = 0.75 * half_diag
    center_density = np.sum(radial_dists < inner_thresh) / n
    edge_density   = np.sum(radial_dists > outer_thresh) / n

    # ── Star count ───────────────────────────
    # Normalize by expected max (tune MAX_STARS from generator config)
    MAX_STARS = 120
    star_density = min(n / MAX_STARS, 1.0)

    # ── Spread (std dev) ────────────────────
    std_x = np.std(xs) / IMG_W
    std_y = np.std(ys) / IMG_H

    return np.array([
        top_ratio,       # [0]
        bottom_ratio,    # [1]
        left_ratio,      # [2]
        right_ratio,     # [3]
        centroid_x,      # [4]
        centroid_y,      # [5]
        mean_radial,     # [6]
        star_density,    # [7]
        center_density,  # [8]
        edge_density,    # [9]
        std_x,           # [10]
        std_y,           # [11]
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# 3. LOAD DATASET + EXTRACT FEATURES
# ─────────────────────────────────────────────

def load_and_extract(dataset_dir):
    """
    Load all images, run feature extraction, return X (features) and y (labels).
    Images are loaded as grayscale uint8 — no normalization needed here
    since detect_stars uses raw brightness threshold.
    """
    X, y = [], []
    print("Loading images and extracting features...")

    for class_name in CLASSES:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  WARNING: folder not found: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        print(f"  {class_name:>8s}: {len(files)} images", end=" ", flush=True)

        class_features = []
        for fname in files:
            path = os.path.join(class_dir, fname)
            img = Image.open(path).convert("L")
            img = img.resize((IMG_W, IMG_H))
            arr = np.array(img, dtype=np.uint8)
            feat = extract_features(arr)
            class_features.append(feat)
            X.append(feat)
            y.append(CLASS_TO_ID[class_name])

        # Quick sanity: print mean feature values for this class
        cf = np.array(class_features)
        print(f"| top={cf[:,0].mean():.2f} bot={cf[:,1].mean():.2f} "
              f"lft={cf[:,2].mean():.2f} rgt={cf[:,3].mean():.2f} "
              f"rad={cf[:,6].mean():.2f}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nTotal: {len(X)} samples | Feature dim: {X.shape[1]}\n")
    return X, y


# ─────────────────────────────────────────────
# 4. BUILD CLASSIFIER
# ─────────────────────────────────────────────

def build_classifier(name):
    if name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "gbm":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (RF only)
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "top_ratio", "bottom_ratio", "left_ratio", "right_ratio",
    "centroid_x", "centroid_y", "mean_radial",
    "star_density", "center_density", "edge_density",
    "std_x", "std_y",
]

def plot_feature_importance(clf):
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)),
               [FEATURE_NAMES[i] for i in idx], rotation=45, ha="right")
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved: feature_importance.png")
    plt.show()


# ─────────────────────────────────────────────
# 6. BENCHMARK INFERENCE TIME
# ─────────────────────────────────────────────

def benchmark_inference(clf, scaler, X_test, num_runs=1000):
    """
    Time end-to-end: feature-extract one image → scale → predict.
    Uses a real test image array so timing is realistic.

    Returns avg latency (ms) and throughput (img/sec).
    """
    # Re-load one raw image for realistic feature extraction timing
    sample_class = CLASSES[0]
    sample_dir = os.path.join(DATASET_DIR, sample_class)
    sample_file = [f for f in os.listdir(sample_dir) if f.endswith(".png")][0]
    sample_img = np.array(
        Image.open(os.path.join(sample_dir, sample_file))
              .convert("L")
              .resize((IMG_W, IMG_H)),
        dtype=np.uint8
    )

    # Warm up
    feat = extract_features(sample_img).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    _ = clf.predict(feat_scaled)

    # Timed runs
    start = time.perf_counter()
    for _ in range(num_runs):
        feat = extract_features(sample_img).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        _ = clf.predict(feat_scaled)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    throughput = 1000.0 / avg_ms
    return avg_ms, throughput


# ─────────────────────────────────────────────
# 7. PLOT HELPERS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix — Geometric Feature Classifier")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")
    plt.show()


def plot_per_class_accuracy(report_dict):
    """Bar chart of per-class F1 scores."""
    classes = [c for c in CLASSES if c in report_dict]
    f1_scores = [report_dict[c]["f1-score"] for c in classes]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(classes, f1_scores, color="steelblue", edgecolor="white")
    plt.ylim(0, 1.05)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1, label="90% target")
    for bar, val in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.title("Per-Class F1 Score")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved: training_curves.png")
    plt.show()


# ─────────────────────────────────────────────
# 8. SAVE BENCHMARK REPORT
# ─────────────────────────────────────────────

def save_benchmark_report(test_acc, cv_mean, cv_std, avg_ms, throughput, report_str):
    lines = [
        "=" * 55,
        "  FPGA STAR TRACKER — GEOMETRIC FEATURE BENCHMARK",
        "=" * 55,
        "",
        f"  Classifier         : {CLASSIFIER.upper()}",
        f"  Feature Dim        : 12 (hand-crafted spatial features)",
        f"  Image Resolution   : {IMG_W} x {IMG_H} px (grayscale)",
        f"  Classes            : {', '.join(CLASSES)}",
        "",
        "  ── Accuracy ──",
        f"  Test Accuracy      : {test_acc * 100:.2f}%",
        f"  Cross-Val (5-fold) : {cv_mean * 100:.2f}% ± {cv_std * 100:.2f}%",
        "",
        "  ── Inference Timing (CPU, per image) ──",
        f"  Avg Latency        : {avg_ms:.4f} ms",
        f"  Throughput         : {throughput:.0f} images / sec",
        "",
        "  ── Per-Class Report ──",
        report_str,
        "",
        "  NOTE: The 12 feature values computed per image are the",
        "  exact values to replicate in FPGA HDL for inference.",
        "  See extract_features() in this file for the full spec.",
        "=" * 55,
    ]
    report = "\n".join(lines)
    print(report)

    with open("benchmark_report.txt", "w") as f:
        f.write(report)
    print("\nSaved: benchmark_report.txt")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # 1. Load + extract features
    X, y = load_and_extract(DATASET_DIR)

    # 2. Train / val / test split (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"Split — Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # 3. Scale features (important for logreg, harmless for RF)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # 4. Build + train
    print(f"Training {CLASSIFIER.upper()} classifier...")
    clf = build_classifier(CLASSIFIER)
    t0 = time.perf_counter()
    clf.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0
    print(f"Training done in {train_time:.2f}s\n")

    # 5. Validation check
    val_acc = accuracy_score(y_val, clf.predict(X_val_s))
    print(f"Validation accuracy: {val_acc * 100:.2f}%\n")

    # 6. Cross-validation (on full train set for stable estimate)
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, n_jobs=-1)
    print(f"CV: {cv_scores * 100} → mean {cv_scores.mean()*100:.2f}%\n")

    # 7. Test set evaluation
    print("Evaluating on held-out test set...")
    y_pred = clf.predict(X_test_s)
    test_acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=CLASSES)
    report_dict = classification_report(y_test, y_pred,
                                         target_names=CLASSES, output_dict=True)
    print(report_str)

    # 8. Benchmark inference
    print("Benchmarking inference time (1000 runs)...")
    avg_ms, throughput = benchmark_inference(clf, scaler, X_test)
    print(f"Avg latency: {avg_ms:.4f} ms | Throughput: {throughput:.0f} img/s\n")

    # 9. Save model + scaler
    joblib.dump(clf,    "star_tracker_model.pkl")
    joblib.dump(scaler, "feature_scaler.pkl")
    print("Saved: star_tracker_model.pkl")
    print("Saved: feature_scaler.pkl\n")

    # 10. Plots + report
    plot_confusion_matrix(y_test, y_pred)
    plot_per_class_accuracy(report_dict)
    plot_feature_importance(clf)
    save_benchmark_report(test_acc, cv_scores.mean(), cv_scores.std(),
                           avg_ms, throughput, report_str)


if __name__ == "__main__":
    main()