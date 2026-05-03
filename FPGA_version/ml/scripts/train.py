#!/usr/bin/env python3
"""
Train a compact star-attitude CNN using synthetic star images generated from 3D geometry.

High-level workflow:
1) Create a deterministic 3D star catalog (a fixed "universe") from a seed.
2) Render 2D camera images by rotating that universe with yaw/pitch/roll.
3) Add jitter and sensor noise to improve robustness.
4) Train a tiny CNN suitable for later HLS deployment.
5) Export quantized weights to a C header and save run metadata/artifacts.

This script is intentionally self-contained so a beginner can follow how dataset generation,
training, and export are connected.
"""

from datetime import datetime
from math import atan, tan
from pathlib import Path
import json
import os
import random
import time
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

NUM_CLASSES = 6

# Fixed "universe" (3D stars on unit sphere).
# Increased from 200 to 500 to provide more features for CNN to learn from.
NUM_STARS = 500
UNIVERSE_SEED = 1234

# Class map: 0=Up, 1=Down, 2=Left, 3=Right, 4=Forward, 5=Backward
CLASS_ANGLES_DEG = {
    0: (0.0, 90.0, 0.0),
    1: (0.0, -90.0, 0.0),
    2: (-90.0, 0.0, 0.0),
    3: (90.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0),
    5: (180.0, 0.0, 0.0),
}

CLASS_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "Forward",
    5: "Backward",
}


def rotation_matrix_to_quaternion(R):
    """Convert 3×3 rotation matrix to unit quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def quaternion_to_euler_deg(q):
    """Convert unit quaternion [w,x,y,z] to (yaw, pitch, roll) in degrees.
    Uses ZYX aerospace convention matching get_rotation_matrix in this file.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll  = np.degrees(np.arctan2(sinr, cosr))
    sinp  = 2.0 * (w * y - z * x)
    pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    siny  = 2.0 * (w * z + x * y)
    cosy  = 1.0 - 2.0 * (y * y + z * z)
    yaw   = np.degrees(np.arctan2(siny, cosy))
    return float(pitch), float(roll), float(yaw)


def angular_error_deg_batch(q_pred, q_true):
    """Per-sample angular error in degrees between two batches of unit quaternions.
    Handles double-cover: q and -q represent the same rotation.
    q_pred, q_true: torch tensors shape (B, 4)
    Returns: tensor shape (B,) with error in degrees.
    """
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot) * (180.0 / torch.pi)


def geodesic_loss(q_pred, q_true):
    """Quaternion geodesic loss: 0 when identical, 1 when orthogonal.
    Handles double-cover (q == -q) by taking abs of dot product.
    q_pred, q_true: (B, 4) unit quaternions
    """
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(0.0, 1.0)
    return (1.0 - dot).mean()


# CNN architecture — sized to fit Zynq-7020 BRAM comfortably using flip-flop buffers.
# 120x90 input + 12/24/48 channels + pool(3,5) to adapt to new size.
CNN_CONV1_OUT_CH = 12
CNN_CONV1_KERNEL = 3    # k=3, p=1, s=2: 120->60, 90->45
CNN_CONV1_PAD = 1
CNN_CONV2_OUT_CH = 24
CNN_CONV3_OUT_CH = 48
CNN_KERNEL = 3          # conv2 / conv3
CNN_STRIDE = 2
CNN_PAD = 1
# 120x90 input: conv1->60x45, conv2->30x23, conv3->15x12
# AdaptiveAvgPool2d((3,5)): 12/3=4, 15/5=3 -> integer bins, 48x3x5 = 720 features
CNN_POOL_H = 3
CNN_POOL_W = 5
CNN_FC1_OUT = 64


def make_star_catalog(num_stars=NUM_STARS, seed=UNIVERSE_SEED):
    """
    Create a reproducible random star catalog on the unit sphere.
    
    Each star has 4 components: (x, y, z, magnitude)
    - (x,y,z): Unit sphere position
    - magnitude: Brightness [0, 1], drawn from realistic distribution
    """
    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((num_stars, 3), dtype=np.float32)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    
    # Assign magnitudes: exponential distribution (faint stars are more common)
    magnitudes = np.exp(-rng.uniform(0, 2, size=num_stars)).astype(np.float32)
    
    catalog = np.column_stack([positions, magnitudes])
    return catalog


STAR_CATALOG = make_star_catalog()


def get_rotation_matrix(yaw, pitch, roll):
    """
    Build a camera rotation matrix from intuitive camera axes.

    Convention used here:
    - yaw:   left/right rotation about the vertical axis
    - pitch: up/down tilt about the horizontal axis
    - roll:  in-plane image rotation about the optical axis

    This matches how the interactive viewer sliders are labeled.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    ryaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rpitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rroll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return rroll @ rpitch @ ryaw


def render_star_camera(
    yaw,
    pitch,
    roll,
    star_catalog,
    img_width=640,
    img_height=480,
    fov_x_degrees=62.0,
    fov_y_degrees=None,
    psf_sigma=1.5,
):
    """
    Project 3D stars to a 2D image using a pinhole camera model with PSF blurring.

    Improvements for physical realism:
    - Uses star magnitudes to control brightness gradation
    - Applies Gaussian PSF (Point Spread Function) to blur stars via camera optics
    - Provides more texture information for CNN to learn discriminative features
    """
    img = np.zeros((img_height, img_width), dtype=np.float32)

    fov_x_rad = np.radians(fov_x_degrees)
    if fov_y_degrees is None:
        fov_y_rad = 2.0 * atan((img_height / img_width) * tan(fov_x_rad / 2.0))
    else:
        fov_y_rad = np.radians(fov_y_degrees)

    fx = (img_width / 2.0) / np.tan(fov_x_rad / 2.0)
    fy = (img_height / 2.0) / np.tan(fov_y_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0

    r = get_rotation_matrix(yaw, pitch, roll)
    # Catalog: [x, y, z, magnitude]
    rotated_xyz = star_catalog[:, :3] @ r.T
    in_front_mask = rotated_xyz[:, 2] > 0.0
    rotated_xyz = rotated_xyz[in_front_mask]
    magnitudes = star_catalog[in_front_mask, 3]

    # Vectorized PSF: compute all star projections at once
    u_float = (fx * rotated_xyz[:, 0] / rotated_xyz[:, 2]) + cx  # (N,)
    v_float = (fy * rotated_xyz[:, 1] / rotated_xyz[:, 2]) + cy  # (N,)

    psf_kernel_size = int(np.ceil(3 * psf_sigma))
    offsets = np.arange(-psf_kernel_size, psf_kernel_size + 1)
    du, dv = np.meshgrid(offsets, offsets)
    du = du.flatten()  # (K²,)
    dv = dv.flatten()  # (K²,)

    u_center = np.round(u_float).astype(np.int32)  # (N,)
    v_center = np.round(v_float).astype(np.int32)  # (N,)

    # (N, K²) pixel coordinates
    u_pix = u_center[:, None] + du[None, :]
    v_pix = v_center[:, None] + dv[None, :]

    dist_sq = (u_float[:, None] - u_pix) ** 2 + (v_float[:, None] - v_pix) ** 2
    psf_weight = np.exp(-dist_sq / (2 * psf_sigma ** 2))  # (N, K²)

    contributions = magnitudes[:, None] * psf_weight  # (N, K²)

    valid = (u_pix >= 0) & (u_pix < img_width) & (v_pix >= 0) & (v_pix < img_height)
    np.add.at(img, (v_pix[valid], u_pix[valid]), contributions[valid])

    return np.clip(img, 0.0, 1.0)


def generate_labeled_sample(
    class_id,
    star_catalog,
    jitter_degrees,
    camera_width,
    camera_height,
    model_width,
    model_height,
    fov_x_degrees,
    noise_prob,
    rng,
    psf_sigma=1.5,
):
    """
    Generate one labeled training sample with realistic magnitude and PSF effects.

    The class picks a base attitude, jitter perturbs it, and camera projection renders
    a realistic star image with magnitude gradation and PSF blurring. Gaussian noise
    (not binary bit-flips) better simulates real sensor noise on continuous intensity.
    """
    base_yaw, base_pitch, base_roll = CLASS_ANGLES_DEG[class_id]

    yaw_deg = base_yaw + rng.uniform(-jitter_degrees, jitter_degrees)
    pitch_deg = base_pitch + rng.uniform(-jitter_degrees, jitter_degrees)
    roll_deg = base_roll + rng.uniform(-jitter_degrees, jitter_degrees)

    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    img_np = render_star_camera(
        yaw,
        pitch,
        roll,
        star_catalog=star_catalog,
        img_width=camera_width,
        img_height=camera_height,
        fov_x_degrees=fov_x_degrees,
        psf_sigma=psf_sigma,
    )

    # Max-normalize: sparse star images have very low raw pixel values.
    # Without this, most of the [0,1] range is unused and gradients starve.
    max_val = img_np.max()
    if max_val > 1e-6:
        img_np = img_np / max_val

    img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    if (model_height, model_width) != (camera_height, camera_width):
        img = F.interpolate(
            img,
            size=(model_height, model_width),
            mode="bilinear",
            align_corners=False,
        )

    img = img.squeeze(0)

    # Apply Gaussian noise instead of bit-flips for continuous-valued images
    if noise_prob > 0.0:
        noise = torch.normal(0, noise_prob, size=img.shape)
        img = torch.clamp(img + noise, 0.0, 1.0)

    # Build ground-truth quaternion from the actual rotation applied
    R = get_rotation_matrix(yaw, pitch, roll)
    quaternion = rotation_matrix_to_quaternion(R)  # [w, x, y, z]

    metadata = {
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[int(class_id)],
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "quaternion": quaternion.tolist(),
    }
    return img, quaternion, metadata


class StarTrackerSyntheticDataset(Dataset):
    """
    PyTorch dataset that yields (image, class_id) pairs generated on-the-fly.

    Why on-the-fly generation?
    - Reduces storage needs (no giant image dataset on disk).
    - Keeps synthetic randomness reproducible via per-sample seeds.
    - Allows flexible parameter tuning without regenerating dataset.
    """
    def __init__(
        self,
        num_samples,
        star_catalog,
        camera_width=640,
        camera_height=480,
        model_width=160,
        model_height=120,
        jitter_degrees=10.0,
        noise_prob=0.01,
        fov_x_degrees=62.0,
        psf_sigma=1.5,
        cache_samples=True,
        seed=42,
    ):
        self.num_samples = num_samples
        self.star_catalog = star_catalog
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.model_width = model_width
        self.model_height = model_height
        self.jitter_degrees = jitter_degrees
        self.noise_prob = noise_prob
        self.fov_x_degrees = fov_x_degrees
        self.psf_sigma = psf_sigma
        self.cache_samples = cache_samples

        rng = np.random.default_rng(seed)
        self.class_ids = rng.integers(0, NUM_CLASSES, size=num_samples, endpoint=False)
        self.sample_seeds = rng.integers(0, 2**31 - 1, size=num_samples)

        if self.cache_samples:
            cached_images = []
            cached_quats = []
            for idx in range(self.num_samples):
                class_id = int(self.class_ids[idx])
                sample_rng = np.random.default_rng(int(self.sample_seeds[idx]))
                img, quat, _meta = generate_labeled_sample(
                    class_id=class_id,
                    star_catalog=self.star_catalog,
                    jitter_degrees=self.jitter_degrees,
                    camera_width=self.camera_width,
                    camera_height=self.camera_height,
                    model_width=self.model_width,
                    model_height=self.model_height,
                    fov_x_degrees=self.fov_x_degrees,
                    noise_prob=self.noise_prob,
                    rng=sample_rng,
                    psf_sigma=self.psf_sigma,
                )
                cached_images.append(img)
                cached_quats.append(torch.from_numpy(quat))
            self.cached_images = torch.stack(cached_images, dim=0)
            self.cached_quats = torch.stack(cached_quats, dim=0)
        else:
            self.cached_images = None
            self.cached_quats = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        class_id = int(self.class_ids[idx])
        if self.cached_images is not None:
            img = self.cached_images[idx]
            quat = self.cached_quats[idx]
        else:
            sample_rng = np.random.default_rng(int(self.sample_seeds[idx]))
            img, quat_np, _meta = generate_labeled_sample(
                class_id=class_id,
                star_catalog=self.star_catalog,
                jitter_degrees=self.jitter_degrees,
                camera_width=self.camera_width,
                camera_height=self.camera_height,
                model_width=self.model_width,
                model_height=self.model_height,
                fov_x_degrees=self.fov_x_degrees,
                noise_prob=self.noise_prob,
                rng=sample_rng,
                psf_sigma=self.psf_sigma,
            )
            quat = torch.from_numpy(quat_np)
        return img, class_id, quat


class StarTrackerTinyCNN(nn.Module):
    """
    Dual-head CNN: shared backbone → classification head + quaternion regression head.

    Star count injection:
      Mean pixel brightness of the input image is appended to the 960 pooled features
      before both FC heads. This lets the network learn that sparse-star frames carry
      less attitude information (differentiable proxy for visible star count).

    Classification head  → 6-class coarse direction (for FPGA / HLS export)
    Quaternion head      → unit quaternion [w,x,y,z] (for ARM PS / precise attitude)

    BatchNorm is trained here but folded into conv weights at HLS export time.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(1, CNN_CONV1_OUT_CH, kernel_size=CNN_CONV1_KERNEL,
                               stride=CNN_STRIDE, padding=CNN_CONV1_PAD)
        self.bn1 = nn.BatchNorm2d(CNN_CONV1_OUT_CH)
        self.conv2 = nn.Conv2d(CNN_CONV1_OUT_CH, CNN_CONV2_OUT_CH, kernel_size=CNN_KERNEL,
                               stride=CNN_STRIDE, padding=CNN_PAD)
        self.bn2 = nn.BatchNorm2d(CNN_CONV2_OUT_CH)
        self.conv3 = nn.Conv2d(CNN_CONV2_OUT_CH, CNN_CONV3_OUT_CH, kernel_size=CNN_KERNEL,
                               stride=CNN_STRIDE, padding=CNN_PAD)
        self.bn3 = nn.BatchNorm2d(CNN_CONV3_OUT_CH)
        self.pool = nn.AdaptiveAvgPool2d((CNN_POOL_H, CNN_POOL_W))

        feat_dim = CNN_CONV3_OUT_CH * CNN_POOL_H * CNN_POOL_W + 1  # +1 for star_density

        # Classification head
        self.drop_cls = nn.Dropout(0.3)
        self.fc_cls1 = nn.Linear(feat_dim, CNN_FC1_OUT)
        self.drop_cls2 = nn.Dropout(0.2)
        self.fc_cls2 = nn.Linear(CNN_FC1_OUT, num_classes)

        # Quaternion regression head
        self.drop_reg = nn.Dropout(0.3)
        self.fc_reg1 = nn.Linear(feat_dim, CNN_FC1_OUT)
        self.drop_reg2 = nn.Dropout(0.2)
        self.fc_reg2 = nn.Linear(CNN_FC1_OUT, 4)

    def forward(self, x):
        # Star density feature: mean brightness = differentiable proxy for star count
        star_density = x.mean(dim=[2, 3])  # [B, 1]

        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)               # [B, 960]

        # Inject star density
        x = torch.cat([x, star_density], dim=1)  # [B, 961]

        # Classification head
        cls = self.drop_cls(x)
        cls = F.relu(self.fc_cls1(cls))
        cls = self.drop_cls2(cls)
        cls_logits = self.fc_cls2(cls)          # [B, 6]

        # Quaternion head (unit-normalized output)
        reg = self.drop_reg(x)
        reg = F.relu(self.fc_reg1(reg))
        reg = self.drop_reg2(reg)
        quat = F.normalize(self.fc_reg2(reg), dim=1)  # [B, 4]

        return cls_logits, quat

    def forward_cls_only(self, x):
        """Classification-only path used at HLS export time (no regression overhead)."""
        cls_logits, _ = self.forward(x)
        return cls_logits


def evaluate(model, loader, criterion, device, reg_lambda=0.5):
    """Compute validation loss, classification accuracy, and mean angular error."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    angle_errors = []
    total_inf_time = 0.0

    # Force inference on CPU to serve as a proper software baseline
    model.to("cpu")

    with torch.no_grad():
        for images, labels, quats in loader:
            # Move inputs to CPU for inference timing
            images_cpu = images.to("cpu")

            start_t = time.perf_counter()
            cls_logits_cpu, q_pred_cpu = model(images_cpu)
            end_t = time.perf_counter()
            
            total_inf_time += (end_t - start_t)

            # Move results back to original device for loss/metric calculation
            cls_logits = cls_logits_cpu.to(device)
            q_pred = q_pred_cpu.to(device)
            labels = labels.to(device)
            quats  = quats.to(device)

            cls_loss = criterion(cls_logits, labels)
            reg_loss = geodesic_loss(q_pred, quats)
            loss = cls_loss + reg_lambda * reg_loss

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(cls_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            errs = angular_error_deg_batch(q_pred, quats)
            angle_errors.append(errs.cpu())

    # Restore model to original training device
    model.to(device)

    mean_angle_err = torch.cat(angle_errors).mean().item()
    avg_inf_time_ms = (total_inf_time / total) * 1000.0
    throughput_fps = total / total_inf_time
    return running_loss / total, 100.0 * correct / total, mean_angle_err, avg_inf_time_ms, throughput_fps


def calibrate_temperature(model, val_loader, device):
    """
    Find scalar temperature T that minimises NLL on the validation set.
    After calibration, softmax(logits / T) gives well-calibrated probabilities.
    Returns T as a Python float (stored in manifest for inference-time use).
    """
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels, _quats in val_loader:
            cls_logits, _ = model(images.to(device))
            logits_all.append(cls_logits.cpu())
            labels_all.append(labels)

    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)

    T = nn.Parameter(torch.ones(1))
    opt = optim.LBFGS([T], lr=0.01, max_iter=500)
    criterion = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = criterion(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T_val = float(T.item())
    print(f"Calibrated temperature T = {T_val:.4f}  "
          f"(>1 → model was overconfident, <1 → underconfident)")
    return T_val


def quantize_array(array, frac_bits):
    """Quantize float parameters into signed 8-bit fixed-point integers."""
    scale = 1 << frac_bits
    q = np.round(array * scale).astype(np.int32)
    return np.clip(q, -128, 127)


def write_flat_array(f, name, array, elements_per_line=24):
    """Write a flattened C array declaration for header export."""
    flat = array.flatten()
    f.write(f"const weight_storage_t {name}[{len(flat)}] = {{\n")
    for i, val in enumerate(flat):
        f.write(str(int(val)))
        if i < len(flat) - 1:
            f.write(", ")
        if (i + 1) % elements_per_line == 0 and i < len(flat) - 1:
            f.write("\n")
    f.write("\n};\n\n")


def fold_bn_into_conv(conv_w, conv_b, bn_mean, bn_var, bn_gamma, bn_beta, bn_eps=1e-5):
    """Fold BatchNorm parameters into the preceding conv layer for hardware export.

    After folding, the HLS kernel only needs conv+bias — no separate BN stage.
    Math: BN(W*x+b) = scale*(W*x + b - mean) + beta
          = (scale*W)*x + (scale*(b-mean) + beta)   where scale = gamma/sqrt(var+eps)
    """
    scale = bn_gamma / np.sqrt(bn_var + bn_eps)       # (out_ch,)
    folded_w = conv_w * scale[:, None, None, None]
    folded_b = (conv_b - bn_mean) * scale + bn_beta
    return folded_w, folded_b


def export_to_hls_header(model, output_header_path, frac_bits, model_width, model_height, catalog_seed):
    """
    Export trained CNN parameters to a C header consumed by HLS C++ inference.

    BatchNorm layers are folded into conv weights at export time so the hardware
    kernel implements only conv+bias. The generated header encodes all shape macros
    and the catalog seed so it can be traced back to the run that produced it.
    """
    model_cpu = model.cpu().eval()

    def get_np(param):
        return param.detach().numpy()

    # Fold BN into conv weights
    c1w, c1b = fold_bn_into_conv(
        get_np(model_cpu.conv1.weight), get_np(model_cpu.conv1.bias),
        get_np(model_cpu.bn1.running_mean), get_np(model_cpu.bn1.running_var),
        get_np(model_cpu.bn1.weight), get_np(model_cpu.bn1.bias),
    )
    c2w, c2b = fold_bn_into_conv(
        get_np(model_cpu.conv2.weight), get_np(model_cpu.conv2.bias),
        get_np(model_cpu.bn2.running_mean), get_np(model_cpu.bn2.running_var),
        get_np(model_cpu.bn2.weight), get_np(model_cpu.bn2.bias),
    )
    c3w, c3b = fold_bn_into_conv(
        get_np(model_cpu.conv3.weight), get_np(model_cpu.conv3.bias),
        get_np(model_cpu.bn3.running_mean), get_np(model_cpu.bn3.running_var),
        get_np(model_cpu.bn3.weight), get_np(model_cpu.bn3.bias),
    )
    fc1_w = get_np(model_cpu.fc_cls1.weight)
    fc1_b = get_np(model_cpu.fc_cls1.bias)
    fc2_w = get_np(model_cpu.fc_cls2.weight)
    fc2_b = get_np(model_cpu.fc_cls2.bias)

    c1w_q  = quantize_array(c1w,  frac_bits)
    c1b_q  = quantize_array(c1b,  frac_bits)
    c2w_q  = quantize_array(c2w,  frac_bits)
    c2b_q  = quantize_array(c2b,  frac_bits)
    c3w_q  = quantize_array(c3w,  frac_bits)
    c3b_q  = quantize_array(c3b,  frac_bits)
    fc1w_q = quantize_array(fc1_w, frac_bits)
    fc1b_q = quantize_array(fc1_b, frac_bits)
    fc2w_q = quantize_array(fc2_w, frac_bits)
    fc2b_q = quantize_array(fc2_b, frac_bits)

    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated header: Star Tracker CNN weights (BN folded into conv)\n")
        f.write(f"// Scale factor: {1 << frac_bits} (fixed-point)\n")
        f.write(f"// Star catalog seed: {catalog_seed}\n\n")

        f.write("#ifndef STAR_TRACKER_WEIGHTS_H\n")
        f.write("#define STAR_TRACKER_WEIGHTS_H\n\n")
        f.write("#if !STAR_TRACKER_USE_FLOAT\n")
        f.write("#include <ap_int.h>\n")
        f.write("typedef ap_int<8> weight_storage_t;\n")
        f.write("#else\n")
        f.write("typedef int weight_storage_t;\n")
        f.write("#endif\n\n")

        f.write(f"#define ST_INPUT_WIDTH {model_width}\n")
        f.write(f"#define ST_INPUT_HEIGHT {model_height}\n")
        f.write(f"#define ST_NUM_CLASSES {NUM_CLASSES}\n")
        f.write(f"#define ST_FRAC_BITS {frac_bits}\n")
        f.write(f"#define ST_CATALOG_SEED {catalog_seed}\n\n")

        f.write(f"#define ST_CONV1_IN_CH 1\n")
        f.write(f"#define ST_CONV1_OUT_CH {CNN_CONV1_OUT_CH}\n")
        f.write(f"#define ST_CONV1_K {CNN_CONV1_KERNEL}\n")
        f.write(f"#define ST_CONV1_STRIDE {CNN_STRIDE}\n")
        f.write(f"#define ST_CONV1_PAD {CNN_CONV1_PAD}\n\n")

        f.write(f"#define ST_CONV2_IN_CH {CNN_CONV1_OUT_CH}\n")
        f.write(f"#define ST_CONV2_OUT_CH {CNN_CONV2_OUT_CH}\n")
        f.write(f"#define ST_CONV2_K {CNN_KERNEL}\n")
        f.write(f"#define ST_CONV2_STRIDE {CNN_STRIDE}\n")
        f.write(f"#define ST_CONV2_PAD {CNN_PAD}\n\n")

        f.write(f"#define ST_CONV3_IN_CH {CNN_CONV2_OUT_CH}\n")
        f.write(f"#define ST_CONV3_OUT_CH {CNN_CONV3_OUT_CH}\n")
        f.write(f"#define ST_CONV3_K {CNN_KERNEL}\n")
        f.write(f"#define ST_CONV3_STRIDE {CNN_STRIDE}\n")
        f.write(f"#define ST_CONV3_PAD {CNN_PAD}\n\n")

        f.write(f"#define ST_POOL_H {CNN_POOL_H}\n")
        f.write(f"#define ST_POOL_W {CNN_POOL_W}\n\n")

        f.write(f"#define ST_FC1_IN  ({CNN_CONV3_OUT_CH} * {CNN_POOL_H} * {CNN_POOL_W} + 1)\n")
        f.write(f"#define ST_FC1_OUT {CNN_FC1_OUT}\n\n")

        write_flat_array(f, "conv1_w", c1w_q)
        write_flat_array(f, "conv1_b", c1b_q)
        write_flat_array(f, "conv2_w", c2w_q)
        write_flat_array(f, "conv2_b", c2b_q)
        write_flat_array(f, "conv3_w", c3w_q)
        write_flat_array(f, "conv3_b", c3b_q)
        write_flat_array(f, "fc1_w",   fc1w_q)
        write_flat_array(f, "fc1_b",   fc1b_q)
        write_flat_array(f, "fc2_w",   fc2w_q)
        write_flat_array(f, "fc2_b",   fc2b_q)

        f.write("#endif // STAR_TRACKER_WEIGHTS_H\n")


def train_model(
    num_samples=9000,
    num_epochs=100,
    batch_size=128,
    learning_rate=3e-3,
    camera_width=640,
    camera_height=480,
    model_width=80,
    model_height=60,
    fov_x_degrees=62.0,
    noise_prob=0.02,
    psf_sigma=1.5,
    reg_lambda=0.5,
    dataset_seed=42,
    catalog_seed=UNIVERSE_SEED,
    cache_samples=True,
    num_workers=None,
    device=None,
):
    """
    Train the dual-head CNN with curriculum learning.
    Combined loss: CrossEntropy(class) + reg_lambda * GeodesicLoss(quaternion)
    Returns: (model, star_catalog, val_loader) — val_loader used for temperature calibration.
    """
    CURRICULUM_JITTERS = [10.0, 17.0, 25.0]
    phase_len = num_epochs // 3

    star_catalog = make_star_catalog(num_stars=NUM_STARS, seed=catalog_seed)

    print("Building curriculum datasets (3 phases)...")
    datasets = []
    for jitter in CURRICULUM_JITTERS:
        print(f"  Generating dataset jitter={jitter}°  ({num_samples} samples)...")
        ds = StarTrackerSyntheticDataset(
            num_samples=num_samples,
            star_catalog=star_catalog,
            camera_width=camera_width,
            camera_height=camera_height,
            model_width=model_width,
            model_height=model_height,
            jitter_degrees=jitter,
            noise_prob=noise_prob,
            fov_x_degrees=fov_x_degrees,
            psf_sigma=psf_sigma,
            cache_samples=cache_samples,
            seed=dataset_seed + int(jitter * 10),
        )
        datasets.append(ds)

    if num_workers is None:
        _nw_candidate = min(4, os.cpu_count() or 1)
        # Probe whether the dataset class is picklable (required for num_workers > 0).
        # When train.py is exec()'d into a dynamic namespace the class can't be pickled.
        import pickle as _pickle
        try:
            _pickle.dumps(datasets[0])
            num_workers = _nw_candidate
        except Exception:
            num_workers = 0

    def make_loaders(ds):
        train_size = int(0.8 * len(ds))
        val_size   = len(ds) - train_size
        train_ds, val_ds = random_split(
            ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(7),
        )
        tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
        va = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
        return tr, va

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Probe MPS for AdaptiveAvgPool2d support with current spatial dims.
            # After 3 stride-2 convs: H_feat = ceil(model_height/8), W_feat = ceil(model_width/8).
            # MPS requires feat dims divisible by pool output dims.
            h_feat = (model_height + 7) // 8
            w_feat = (model_width + 7) // 8
            if h_feat % CNN_POOL_H == 0 and w_feat % CNN_POOL_W == 0:
                device = torch.device("mps")
            else:
                print("MPS AdaptiveAvgPool2d constraint not met "
                      f"(feat {w_feat}×{h_feat} vs pool {CNN_POOL_W}×{CNN_POOL_H}); "
                      "falling back to CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    model = StarTrackerTinyCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [], "train_ang": [],
        "val_loss": [], "val_acc": [], "val_ang": [], "val_inf_time": [], "val_fps": []
    }

    current_phase = 0
    train_loader, val_loader = make_loaders(datasets[0])
    print(f"\n--- Curriculum Phase 1/3  jitter={CURRICULUM_JITTERS[0]}° ---")

    for epoch in range(num_epochs):
        new_phase = min(epoch // phase_len, 2)
        if new_phase != current_phase:
            current_phase = new_phase
            train_loader, val_loader = make_loaders(datasets[current_phase])
            print(f"\n--- Curriculum Phase {current_phase + 1}/3  "
                  f"jitter={CURRICULUM_JITTERS[current_phase]}° ---")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        angle_errors = []

        for images, labels, quats in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            quats  = quats.to(device)

            optimizer.zero_grad()
            cls_logits, q_pred = model(images)

            cls_loss = criterion(cls_logits, labels)
            reg_loss = geodesic_loss(q_pred, quats)
            loss = cls_loss + reg_lambda * reg_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(cls_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            angle_errors.append(angular_error_deg_batch(q_pred.detach(), quats).cpu())

        scheduler.step()

        train_loss = running_loss / total
        train_acc  = 100.0 * correct / total
        train_ang  = torch.cat(angle_errors).mean().item()
        val_loss, val_acc, val_ang, val_inf_time, val_fps = evaluate(model, val_loader, criterion, device, reg_lambda)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_ang"].append(train_ang)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ang"].append(val_ang)
        history["val_inf_time"].append(val_inf_time)
        history["val_fps"].append(val_fps)

        print(
            f"Epoch [{epoch + 1:3d}/{num_epochs}] "
            f"Ph{current_phase + 1} "
            f"Loss:{train_loss:.4f} Acc:{train_acc:.1f}% AngErr:{train_ang:.1f}° | "
            f"Val Loss:{val_loss:.4f} Acc:{val_acc:.1f}% AngErr:{val_ang:.1f}° | "
            f"Val Throughput: {val_fps:.1f} FPS"
        )

    print("Training complete.")
    return model, star_catalog, val_loader, history


def save_run_artifacts(
    model,
    star_catalog,
    models_dir,
    run_id,
    frac_bits,
    model_width,
    model_height,
    camera_width,
    camera_height,
    catalog_seed,
    dataset_seed,
    fov_x_degrees,
    jitter_degrees,
    noise_prob,
    temperature=1.0,
    reg_lambda=0.5,
):
    """
    Persist all outputs from one training run.

    Two naming modes are saved:
    - Versioned files with run_id for reproducibility.
    - Canonical filenames for convenient "latest model" use.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    versioned_pth = models_dir / f"star_tracker_cnn_{run_id}.pth"
    versioned_hdr = models_dir / f"star_tracker_weights_{run_id}.h"
    versioned_catalog = models_dir / f"star_catalog_seed{catalog_seed}_{run_id}.npy"
    versioned_manifest = models_dir / f"star_tracker_manifest_{run_id}.json"

    canonical_pth = models_dir / "star_tracker_cnn.pth"
    canonical_hdr = models_dir / "star_tracker_weights.h"
    canonical_catalog = models_dir / f"star_catalog_seed{catalog_seed}.npy"

    torch.save(model.state_dict(), versioned_pth)
    torch.save(model.state_dict(), canonical_pth)

    np.save(versioned_catalog, star_catalog)
    np.save(canonical_catalog, star_catalog)

    export_to_hls_header(
        model=model,
        output_header_path=versioned_hdr,
        frac_bits=frac_bits,
        model_width=model_width,
        model_height=model_height,
        catalog_seed=catalog_seed,
    )
    export_to_hls_header(
        model=model,
        output_header_path=canonical_hdr,
        frac_bits=frac_bits,
        model_width=model_width,
        model_height=model_height,
        catalog_seed=catalog_seed,
    )

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "class_map": CLASS_NAMES,
        "class_angles_deg": CLASS_ANGLES_DEG,
        "catalog_seed": catalog_seed,
        "dataset_seed": dataset_seed,
        "num_stars": int(star_catalog.shape[0]),
        "camera_resolution": {"width": camera_width, "height": camera_height},
        "model_resolution": {"width": model_width, "height": model_height},
        "fov_x_degrees": fov_x_degrees,
        "jitter_degrees": jitter_degrees,
        "noise_prob": noise_prob,
        "temperature": temperature,
        "reg_lambda": reg_lambda,
        "architecture": "dual_head_quaternion_v2",
        "frac_bits": frac_bits,
        "artifacts": {
            "weights_pth": versioned_pth.name,
            "weights_header": versioned_hdr.name,
            "catalog_npy": versioned_catalog.name,
            "canonical_weights_pth": canonical_pth.name,
            "canonical_weights_header": canonical_hdr.name,
            "canonical_catalog_npy": canonical_catalog.name,
        },
    }

    with open(versioned_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved model checkpoint: {versioned_pth}")
    print(f"Saved HLS header:      {versioned_hdr}")
    print(f"Saved star catalog:    {versioned_catalog}")
    print(f"Saved manifest:        {versioned_manifest}")


def plot_metrics(history, save_path):
    """Generate high-quality charts for presentations comparing performance metrics."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Use a clean style if available, fallback gracefully
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        pass

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Star Tracker CNN Training & Inference Metrics', fontsize=18, fontweight='bold', y=0.96)

    # 1. Loss Plot
    axs[0, 0].plot(epochs, history["train_loss"], label='Train Loss', color='tab:blue', linewidth=2)
    axs[0, 0].plot(epochs, history["val_loss"], label='Val Loss', color='tab:orange', linewidth=2)
    axs[0, 0].set_title('Loss Over Epochs', fontsize=14)
    axs[0, 0].set_xlabel('Epoch', fontsize=12)
    axs[0, 0].set_ylabel('Combined Loss', fontsize=12)
    axs[0, 0].legend(fontsize=11)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 2. Accuracy Plot
    axs[0, 1].plot(epochs, history["train_acc"], label='Train Acc', color='tab:green', linewidth=2)
    axs[0, 1].plot(epochs, history["val_acc"], label='Val Acc', color='tab:red', linewidth=2)
    axs[0, 1].set_title('Classification Base Accuracy', fontsize=14)
    axs[0, 1].set_xlabel('Epoch', fontsize=12)
    axs[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axs[0, 1].legend(fontsize=11)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 3. Angular Error Plot
    axs[1, 0].plot(epochs, history["train_ang"], label='Train Err', color='tab:purple', linewidth=2)
    axs[1, 0].plot(epochs, history["val_ang"], label='Val Err', color='tab:brown', linewidth=2)
    axs[1, 0].set_title('Quaternion Angular Error', fontsize=14)
    axs[1, 0].set_xlabel('Epoch', fontsize=12)
    axs[1, 0].set_ylabel('Mean Error (Degrees)', fontsize=12)
    axs[1, 0].legend(fontsize=11)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 4. Inference Time Plot
    axs[1, 1].plot(epochs, history["val_fps"], label='Software Throughput', color='tab:pink', linewidth=2)
    
    # Calculate average time to show on the plot
    avg_fps = np.mean(history["val_fps"])
    axs[1, 1].axhline(y=avg_fps, color='black', linestyle=':', label=f'Avg: {avg_fps:.1f} FPS')
    
    axs[1, 1].set_title('Validation Throughput vs Epoch', fontsize=14)
    axs[1, 1].set_xlabel('Epoch', fontsize=12)
    axs[1, 1].set_ylabel('Throughput (Frames/Second)', fontsize=12)
    axs[1, 1].legend(fontsize=11)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Add a bit of space at the top for the title
    fig.subplots_adjust(top=0.90)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved high-res presentation plot: {save_path}")


if __name__ == "__main__":
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    camera_width  = 640
    camera_height = 480
    model_width   = 120
    model_height  = 90
    frac_bits     = 8
    catalog_seed  = UNIVERSE_SEED
    dataset_seed  = 42
    fov_x_degrees = 62.0
    noise_prob    = 0.02
    reg_lambda    = 0.5

    model, star_catalog, val_loader, history = train_model(
        num_samples=12000,
        num_epochs=80,
        batch_size=128,
        learning_rate=3e-3,
        camera_width=camera_width,
        camera_height=camera_height,
        model_width=model_width,
        model_height=model_height,
        fov_x_degrees=fov_x_degrees,
        noise_prob=noise_prob,
        reg_lambda=reg_lambda,
        dataset_seed=dataset_seed,
        catalog_seed=catalog_seed,
    )

    if torch.cuda.is_available():
        calib_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        calib_device = torch.device("mps")
    else:
        calib_device = torch.device("cpu")
    temperature = calibrate_temperature(model, val_loader, calib_device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{catalog_seed}"
    models_dir = Path(__file__).resolve().parent / "../models"

    save_run_artifacts(
        model=model,
        star_catalog=star_catalog,
        models_dir=models_dir,
        run_id=run_id,
        frac_bits=frac_bits,
        model_width=model_width,
        model_height=model_height,
        camera_width=camera_width,
        camera_height=camera_height,
        catalog_seed=catalog_seed,
        dataset_seed=dataset_seed,
        fov_x_degrees=fov_x_degrees,
        jitter_degrees=25.0,
        noise_prob=noise_prob,
        temperature=temperature,
        reg_lambda=reg_lambda,
    )

    metrics_path = models_dir / f"training_metrics_{run_id}.png"
    plot_metrics(history, metrics_path)
