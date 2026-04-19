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
import random

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


# Tiny CNN chosen to keep HLS header and compute footprint manageable.
CNN_CONV1_OUT_CH = 8
CNN_CONV2_OUT_CH = 16
CNN_KERNEL = 3
CNN_STRIDE = 2
CNN_PAD = 1


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

    # Project each star and apply PSF blurring
    for (x, y, z), magnitude in zip(rotated_xyz, magnitudes):
        u_float = (fx * x / z) + cx
        v_float = (fy * y / z) + cy
        
        # Apply Gaussian PSF by spreading magnitude over nearby pixels
        u_center = int(np.round(u_float))
        v_center = int(np.round(v_float))
        
        psf_kernel_size = int(np.ceil(3 * psf_sigma))
        for du in range(-psf_kernel_size, psf_kernel_size + 1):
            for dv in range(-psf_kernel_size, psf_kernel_size + 1):
                u = u_center + du
                v = v_center + dv
                
                if 0 <= u < img_width and 0 <= v < img_height:
                    # Gaussian PSF: weight decreases with distance from center
                    dist_sq = (u_float - u)**2 + (v_float - v)**2
                    psf_weight = np.exp(-dist_sq / (2 * psf_sigma**2))
                    img[v, u] += magnitude * psf_weight

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

    metadata = {
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[int(class_id)],
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
    }
    return img, metadata


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

        self.cached_images = None
        if self.cache_samples:
            cached_images = []
            for idx in range(self.num_samples):
                class_id = int(self.class_ids[idx])
                sample_rng = np.random.default_rng(int(self.sample_seeds[idx]))
                img, _meta = generate_labeled_sample(
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
            self.cached_images = torch.stack(cached_images, dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        class_id = int(self.class_ids[idx])
        if self.cached_images is not None:
            img = self.cached_images[idx]
        else:
            sample_rng = np.random.default_rng(int(self.sample_seeds[idx]))
            img, _meta = generate_labeled_sample(
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
        return img, class_id


class StarTrackerTinyCNN(nn.Module):
    """
    Minimal CNN designed to balance accuracy with hardware friendliness.

    Architecture: Conv -> ReLU -> Conv -> ReLU -> Global Average Pool -> Linear.
    This avoids a very large fully-connected layer on raw 640x480 pixels.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, CNN_CONV1_OUT_CH, kernel_size=CNN_KERNEL, stride=CNN_STRIDE, padding=CNN_PAD)
        self.conv2 = nn.Conv2d(CNN_CONV1_OUT_CH, CNN_CONV2_OUT_CH, kernel_size=CNN_KERNEL, stride=CNN_STRIDE, padding=CNN_PAD)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(CNN_CONV2_OUT_CH, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def evaluate(model, loader, criterion, device):
    """Compute validation loss/accuracy without gradient updates."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


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


def export_to_hls_header(model, output_header_path, frac_bits, model_width, model_height, catalog_seed):
    """
    Export trained CNN parameters to a C header consumed by HLS C++ inference.

    Export format intentionally includes network shape macros and seed metadata so the
    generated header can be traced back to the starscape that created the model.
    """
    model_cpu = model.cpu()

    conv1_w = model_cpu.conv1.weight.detach().numpy()
    conv1_b = model_cpu.conv1.bias.detach().numpy()
    conv2_w = model_cpu.conv2.weight.detach().numpy()
    conv2_b = model_cpu.conv2.bias.detach().numpy()
    fc_w = model_cpu.fc.weight.detach().numpy()
    fc_b = model_cpu.fc.bias.detach().numpy()

    conv1_w_q = quantize_array(conv1_w, frac_bits)
    conv1_b_q = quantize_array(conv1_b, frac_bits)
    conv2_w_q = quantize_array(conv2_w, frac_bits)
    conv2_b_q = quantize_array(conv2_b, frac_bits)
    fc_w_q = quantize_array(fc_w, frac_bits)
    fc_b_q = quantize_array(fc_b, frac_bits)

    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated header file for Star Tracker Tiny CNN weights\n")
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

        f.write("#define ST_INPUT_WIDTH {}\n".format(model_width))
        f.write("#define ST_INPUT_HEIGHT {}\n".format(model_height))
        f.write("#define ST_NUM_CLASSES {}\n".format(NUM_CLASSES))
        f.write("#define ST_FRAC_BITS {}\n".format(frac_bits))
        f.write("#define ST_CATALOG_SEED {}\n\n".format(catalog_seed))

        f.write("#define ST_CONV1_IN_CH 1\n")
        f.write("#define ST_CONV1_OUT_CH {}\n".format(CNN_CONV1_OUT_CH))
        f.write("#define ST_CONV1_K {}\n".format(CNN_KERNEL))
        f.write("#define ST_CONV1_STRIDE {}\n".format(CNN_STRIDE))
        f.write("#define ST_CONV1_PAD {}\n\n".format(CNN_PAD))

        f.write("#define ST_CONV2_IN_CH {}\n".format(CNN_CONV1_OUT_CH))
        f.write("#define ST_CONV2_OUT_CH {}\n".format(CNN_CONV2_OUT_CH))
        f.write("#define ST_CONV2_K {}\n".format(CNN_KERNEL))
        f.write("#define ST_CONV2_STRIDE {}\n".format(CNN_STRIDE))
        f.write("#define ST_CONV2_PAD {}\n\n".format(CNN_PAD))

        write_flat_array(f, "conv1_w", conv1_w_q)
        write_flat_array(f, "conv1_b", conv1_b_q)
        write_flat_array(f, "conv2_w", conv2_w_q)
        write_flat_array(f, "conv2_b", conv2_b_q)
        write_flat_array(f, "fc_w", fc_w_q)
        write_flat_array(f, "fc_b", fc_b_q)

        f.write("#endif // STAR_TRACKER_WEIGHTS_H\n")


def train_model(
    num_samples=4000,
    num_epochs=60,
    batch_size=64,
    learning_rate=5e-3,
    camera_width=640,
    camera_height=480,
    model_width=160,
    model_height=120,
    fov_x_degrees=62.0,
    jitter_degrees=10.0,
    noise_prob=0.02,
    psf_sigma=1.5,
    cache_samples=True,
    dataset_seed=42,
    catalog_seed=UNIVERSE_SEED,
):
    """
    Train the tiny CNN on synthetic star-camera images with improved realism.

    Improvements:
    - Uses magnitude-based star brightness for texture information
    - Applies PSF blurring to simulate camera optics
    - Larger batch size (64) for more stable gradient estimates and fewer steps per epoch
    - Higher learning rate (5e-3) with decay schedule
    - More training epochs (60) to allow convergence
    - Gaussian noise on continuous intensity values
    - Optional dataset caching to avoid re-rendering samples every epoch
    """
    star_catalog = make_star_catalog(num_stars=NUM_STARS, seed=catalog_seed)

    print("Generating synthetic 3D-projected star fields...")
    print(f"Camera resolution: {camera_width}x{camera_height}")
    print(f"Model resolution:  {model_width}x{model_height}")
    print(f"Star catalog size: {star_catalog.shape[0]} stars")
    print(f"PSF sigma: {psf_sigma}, Jitter: {jitter_degrees}°, Noise: {noise_prob}")

    dataset = StarTrackerSyntheticDataset(
        num_samples=num_samples,
        star_catalog=star_catalog,
        camera_width=camera_width,
        camera_height=camera_height,
        model_width=model_width,
        model_height=model_height,
        jitter_degrees=jitter_degrees,
        noise_prob=noise_prob,
        fov_x_degrees=fov_x_degrees,
        psf_sigma=psf_sigma,
        cache_samples=cache_samples,
        seed=dataset_seed,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarTrackerTinyCNN(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    print("Training complete.")
    return model, star_catalog


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


if __name__ == "__main__":
    # These constants are the practical defaults for your camera + training setup.
    # Change them here when you want a new experiment configuration.
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    camera_width = 640
    camera_height = 480
    model_width = 160
    model_height = 120
    frac_bits = 8
    catalog_seed = UNIVERSE_SEED
    dataset_seed = 42
    fov_x_degrees = 62.0
    jitter_degrees = 10.0
    noise_prob = 0.01

    model, star_catalog = train_model(
        num_samples=4000,
        num_epochs=60,
        batch_size=64,
        learning_rate=5e-3,
        camera_width=camera_width,
        camera_height=camera_height,
        model_width=model_width,
        model_height=model_height,
        fov_x_degrees=fov_x_degrees,
        jitter_degrees=jitter_degrees,
        noise_prob=noise_prob,
        cache_samples=True,
        dataset_seed=dataset_seed,
        catalog_seed=catalog_seed,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{catalog_seed}"
    models_dir = Path(__file__).resolve().parent / "../models"

    # run_id links all generated files (pth/header/catalog/manifest) from one run.
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
        jitter_degrees=jitter_degrees,
        noise_prob=noise_prob,
    )
