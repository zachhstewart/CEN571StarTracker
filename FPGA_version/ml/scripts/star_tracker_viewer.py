#!/usr/bin/env python3
"""
Visualize the starscape tied to a saved training run manifest.

What this script does:
1) Loads a run manifest and the exact star catalog (.npy) used for that run.
2) Re-renders six attitude classes (Up/Down/Left/Right/Forward/Backward).
3) Overlays yaw/pitch/roll values so orientation labels are easy to interpret.

Typical usage:
- Show latest run interactively:
    python3 star_tracker_viewer.py
- Save an image:
    python3 star_tracker_viewer.py --output viewer.png
- Visualize a specific run id:
    python3 star_tracker_viewer.py --run-id 20260418_123000_seed1234
"""

import argparse
import json
from math import atan, tan
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_ORDER = [0, 1, 2, 3, 4, 5]
CLASS_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "Forward",
    5: "Backward",
}
CLASS_ANGLES_DEG = {
    0: (0.0, 90.0, 0.0),
    1: (0.0, -90.0, 0.0),
    2: (-90.0, 0.0, 0.0),
    3: (90.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0),
    5: (180.0, 0.0, 0.0),
}


def get_rotation_matrix(yaw, pitch, roll):
    """
    Create a camera rotation matrix using intuitive camera axes.

    Convention used here:
    - yaw:   left/right rotation about the vertical axis
    - pitch: up/down tilt about the horizontal axis
    - roll:  in-plane image rotation about the optical axis
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    ryaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rpitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rroll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    return rroll @ rpitch @ ryaw


def quaternion_to_euler_deg(q):
    """Unit quaternion [w,x,y,z] → (yaw, pitch, roll) degrees. ZYX convention."""
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


def render_star_camera(star_catalog, yaw, pitch, roll, img_width, img_height, fov_x_degrees, psf_sigma=1.5):
    """
    Project 3D unit-sphere stars into 2D image pixels with physical realism.
    
    Improvements:
    - Uses star magnitudes to create brightness gradation
    - Applies Gaussian PSF (Point Spread Function) to simulate camera optics
    - Provides realistic sensor imagery for better model validation
    """
    img = np.zeros((img_height, img_width), dtype=np.float32)

    fov_x_rad = np.radians(fov_x_degrees)
    fov_y_rad = 2.0 * atan((img_height / img_width) * tan(fov_x_rad / 2.0))

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


def pick_manifest(models_dir, run_id=None):
    """Resolve either requested run_id manifest or the newest available manifest."""
    if run_id:
        target = models_dir / f"star_tracker_manifest_{run_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"Manifest not found: {target}")
        return target

    manifests = sorted(models_dir.glob("star_tracker_manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest files found in {models_dir}")
    return manifests[-1]


def load_run(manifest_path):
    """Load manifest metadata and the matching star catalog file."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models_dir = manifest_path.parent
    catalog_name = manifest["artifacts"]["catalog_npy"]
    catalog_path = models_dir / catalog_name
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file from manifest not found: {catalog_path}")

    star_catalog = np.load(catalog_path)
    return manifest, star_catalog


def render_view(manifest, star_catalog, output_path=None, psf_sigma=1.5):
    """Render a 2x3 grid of attitudes with realistic magnitudes and PSF effects."""
    cam_w = int(manifest["camera_resolution"]["width"])
    cam_h = int(manifest["camera_resolution"]["height"])
    fov_x = float(manifest.get("fov_x_degrees", 62.0))
    jitter = float(manifest.get("jitter_degrees", 10.0))
    dataset_seed = int(manifest.get("dataset_seed", 42))

    rng = np.random.default_rng(dataset_seed)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.flatten()

    for ax, class_id in zip(axes, CLASS_ORDER):
        base_yaw, base_pitch, base_roll = CLASS_ANGLES_DEG[class_id]
        yaw_deg = base_yaw + rng.uniform(-jitter, jitter)
        pitch_deg = base_pitch + rng.uniform(-jitter, jitter)
        roll_deg = base_roll + rng.uniform(-jitter, jitter)

        img = render_star_camera(
            star_catalog=star_catalog,
            yaw=np.radians(yaw_deg),
            pitch=np.radians(pitch_deg),
            roll=np.radians(roll_deg),
            img_width=cam_w,
            img_height=cam_h,
            fov_x_degrees=fov_x,
            psf_sigma=psf_sigma,
        )

        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{class_id}: {CLASS_NAMES[class_id]}\n"
            f"Yaw={yaw_deg:.2f}  Pitch={pitch_deg:.2f}  Roll={roll_deg:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(
        f"Star Tracker Viewer | run={manifest['run_id']} | catalog_seed={manifest['catalog_seed']}",
        fontsize=12,
    )

    if output_path:
        fig.savefig(output_path, dpi=130)
        print(f"Saved viewer image: {output_path}")
    else:
        plt.show()


class StarTrackerTinyCNN(nn.Module):
    """Dual-head CNN — must exactly mirror the architecture in train.py."""
    NUM_CLASSES = 6
    CONV1_OUT_CH = 12;  CONV2_OUT_CH = 24;  CONV3_OUT_CH = 48
    CONV1_K = 3;        KERNEL = 3;         STRIDE = 2
    CONV1_PAD = 1;      PAD = 1
    POOL_H = 3;         POOL_W = 5;         FC1_OUT = 64

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, self.CONV1_OUT_CH, self.CONV1_K, self.STRIDE, self.CONV1_PAD)
        self.bn1 = nn.BatchNorm2d(self.CONV1_OUT_CH)
        self.conv2 = nn.Conv2d(self.CONV1_OUT_CH, self.CONV2_OUT_CH, self.KERNEL, self.STRIDE, self.PAD)
        self.bn2 = nn.BatchNorm2d(self.CONV2_OUT_CH)
        self.conv3 = nn.Conv2d(self.CONV2_OUT_CH, self.CONV3_OUT_CH, self.KERNEL, self.STRIDE, self.PAD)
        self.bn3 = nn.BatchNorm2d(self.CONV3_OUT_CH)
        self.pool = nn.AdaptiveAvgPool2d((self.POOL_H, self.POOL_W))
        feat_dim = self.CONV3_OUT_CH * self.POOL_H * self.POOL_W + 1
        self.drop_cls  = nn.Dropout(0.3);  self.drop_cls2 = nn.Dropout(0.2)
        self.fc_cls1   = nn.Linear(feat_dim, self.FC1_OUT)
        self.fc_cls2   = nn.Linear(self.FC1_OUT, num_classes)
        self.drop_reg  = nn.Dropout(0.3);  self.drop_reg2 = nn.Dropout(0.2)
        self.fc_reg1   = nn.Linear(feat_dim, self.FC1_OUT)
        self.fc_reg2   = nn.Linear(self.FC1_OUT, 4)

    def forward(self, x):
        star_density = x.mean(dim=[2, 3])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, star_density], dim=1)
        cls = F.relu(self.fc_cls1(self.drop_cls(x)))
        cls_logits = self.fc_cls2(self.drop_cls2(cls))
        reg = F.relu(self.fc_reg1(self.drop_reg(x)))
        quat = F.normalize(self.fc_reg2(self.drop_reg2(reg)), dim=1)
        return cls_logits, quat


def load_model(manifest_path, device):
    """Load trained dual-head CNN and calibration temperature from manifest."""
    models_dir = Path(manifest_path).parent
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_name = manifest["artifacts"]["weights_pth"]
    model_path = models_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = StarTrackerTinyCNN(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    temperature = float(manifest.get("temperature", 1.0))
    return model, temperature


def render_interactive(manifest, star_catalog, model, temperature, device,
                       camera_width=640, camera_height=480,
                       model_width=120, model_height=90):
    """
    Interactive viewer with sliders for pitch/roll/yaw/fov and real-time CNN inference.
    
    Use this to validate the trained model:
    - Adjust sliders to rotate the camera view
    - See real-time star field rendering with magnitudes and PSF
    - Watch the CNN prediction update as you move the camera
    - Great for testing physical FPGA camera alignment!
    """
    fov_x_default = float(manifest.get("fov_x_degrees", 62.0))
    psf_sigma = 1.5
    
    # Give most vertical space to the star image; keep controls in a compact bottom band.
    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.22)
    
    # Main image display occupies most of the figure.
    ax_img = fig.add_axes([0.04, 0.28, 0.92, 0.64])
    ax_img.set_title("Star Field View (rotate with sliders below)", fontsize=12, fontweight="bold")
    ax_img.set_xlim(0, camera_width)
    ax_img.set_ylim(camera_height, 0)
    im = ax_img.imshow(np.zeros((camera_height, camera_width)), cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.axis("off")
    
    # Compact prediction text area at lower-right, away from the main image.
    ax_text = fig.add_axes([0.70, 0.04, 0.28, 0.22])
    ax_text.axis("off")
    text_display = ax_text.text(0.1, 0.9, "", transform=ax_text.transAxes, fontsize=11, verticalalignment="top",
                                 family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    # Slider axes in a tight bottom-left strip.
    ax_yaw = plt.axes([0.08, 0.17, 0.60, 0.025])
    ax_pitch = plt.axes([0.08, 0.13, 0.60, 0.025])
    ax_roll = plt.axes([0.08, 0.09, 0.60, 0.025])
    ax_fov = plt.axes([0.08, 0.05, 0.60, 0.025])
    
    slider_yaw = widgets.Slider(ax_yaw, "Yaw (°)", -180, 180, valinit=0, color="blue")
    slider_pitch = widgets.Slider(ax_pitch, "Pitch (°)", -90, 90, valinit=0, color="green")
    slider_roll = widgets.Slider(ax_roll, "Roll (°)", -180, 180, valinit=0, color="red")
    slider_fov = widgets.Slider(ax_fov, "FOV (°)", 30, 120, valinit=fov_x_default, color="purple")
    
    def update_view(val):
        yaw_deg   = slider_yaw.val
        pitch_deg = slider_pitch.val
        roll_deg  = slider_roll.val
        fov_x_deg = slider_fov.val

        img_full = render_star_camera(
            star_catalog=star_catalog,
            yaw=np.radians(yaw_deg), pitch=np.radians(pitch_deg),
            roll=np.radians(roll_deg), img_width=camera_width,
            img_height=camera_height, fov_x_degrees=fov_x_deg, psf_sigma=psf_sigma,
        )
        im.set_data(img_full)

        img_t = torch.from_numpy(img_full).unsqueeze(0).unsqueeze(0).to(device)
        img_r = F.interpolate(img_t, size=(model_height, model_width),
                               mode="bilinear", align_corners=False)

        with torch.no_grad():
            cls_logits, q_pred = model(img_r)
            probs = torch.softmax(cls_logits / temperature, dim=1)

        pred_cls  = int(torch.argmax(cls_logits, dim=1).item())
        pred_conf = float(probs[0, pred_cls].item()) * 100.0
        q_np      = q_pred[0].cpu().numpy()

        cnn_yaw, cnn_pitch, cnn_roll = quaternion_to_euler_deg(q_np)

        # Ground-truth quaternion from sliders for angular error
        def euler_to_quat(y_d, p_d, r_d):
            y, p, r = np.radians(y_d), np.radians(p_d), np.radians(r_d)
            cy, sy = np.cos(y / 2), np.sin(y / 2)
            cp, sp = np.cos(p / 2), np.sin(p / 2)
            cr, sr = np.cos(r / 2), np.sin(r / 2)
            return np.array([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ], dtype=np.float32)

        q_true  = euler_to_quat(yaw_deg, pitch_deg, roll_deg)
        dot     = abs(float(np.dot(q_np, q_true)))
        ang_err = float(np.degrees(2.0 * np.arccos(min(dot, 1.0))))

        star_count_est = int((img_full > 0.15).sum())

        lines = [
            f"━━  CLASSIFICATION  ━━\n",
            f"  Class:   {CLASS_NAMES[pred_cls]} ({pred_cls})\n",
            f"  Conf:    {pred_conf:5.1f}%  (T={temperature:.2f})\n\n",
            f"━━  QUATERNION  ━━\n",
            f"  w={q_np[0]:+.4f}\n",
            f"  x={q_np[1]:+.4f}\n",
            f"  y={q_np[2]:+.4f}\n",
            f"  z={q_np[3]:+.4f}\n\n",
            f"━━  CNN EULER (deg)  ━━\n",
            f"  Yaw:   {cnn_yaw:+7.2f}°\n",
            f"  Pitch: {cnn_pitch:+7.2f}°\n",
            f"  Roll:  {cnn_roll:+7.2f}°\n\n",
            f"━━  ERROR  ━━\n",
            f"  Angular: {ang_err:5.2f}°\n\n",
            f"━━  STAR COUNT  ━━\n",
            f"  Pixels>0.15: {star_count_est}\n",
        ]
        text_display.set_text("".join(lines))
        fig.canvas.draw_idle()
    
    # Connect slider callbacks
    slider_yaw.on_changed(update_view)
    slider_pitch.on_changed(update_view)
    slider_roll.on_changed(update_view)
    slider_fov.on_changed(update_view)
    
    # Initial render
    update_view(None)
    
    plt.suptitle("Star Tracker Interactive Viewer - Rotate Camera & Watch CNN Predictions", 
                 fontsize=13, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    # Command-line options are intentionally simple so this script is easy to use in lab demos.
    parser = argparse.ArgumentParser(description="Visualize starscape attitudes linked to a training run manifest.")
    parser.add_argument("--models-dir", default=str((Path(__file__).resolve().parent / "../models").resolve()))
    parser.add_argument("--run-id", default=None, help="Run id suffix from star_tracker_manifest_<run_id>.json")
    parser.add_argument("--output", default=None, help="Optional PNG output path instead of interactive show")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive slider viewer with live CNN predictions")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    manifest_path = pick_manifest(models_dir=models_dir, run_id=args.run_id)
    manifest, star_catalog = load_run(manifest_path)

    if args.interactive:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model, temperature = load_model(manifest_path, device)
        render_interactive(manifest, star_catalog, model, temperature, device)
    else:
        output_path = Path(args.output) if args.output else None
        render_view(manifest=manifest, star_catalog=star_catalog, output_path=output_path)
