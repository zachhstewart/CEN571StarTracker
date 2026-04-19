# Dual-Head Attitude ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the star tracker from a 6-class direction classifier into a full attitude determination system that outputs calibrated class confidence, a unit quaternion, derived Euler angles, angular error, and star count — all visible live in the interactive viewer.

**Architecture:** The CNN backbone (3 conv layers → spatial pool → 960 features) is unchanged. Star count is injected as one extra scalar making a 961-dim feature vector. Two FC heads branch from there: a classification head (961→128→6) and a quaternion regression head (961→128→4, L2-normalized). Training uses staged curriculum jitter (3°→8°→15° across 3 phases). Post-training temperature scaling calibrates confidence. The FPGA HLS export continues to use only the classification head; the regression head runs on ARM PS.

**Tech Stack:** Python 3, PyTorch, NumPy, Matplotlib (interactive viewer)

---

## File Map

| File | Role |
|---|---|
| `FPGA_version/ml/scripts/train.py` | Architecture, dataset, curriculum training, temperature calibration, export |
| `FPGA_version/ml/scripts/star_tracker_viewer.py` | Interactive viewer — rich attitude display |

---

## Task 1: Quaternion Utility Functions

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — add helpers near the top, after imports

- [ ] **Step 1: Add `rotation_matrix_to_quaternion` and `angular_error_deg` after the `CLASS_NAMES` dict (around line 53)**

```python
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
    Uses ZYX (aerospace) convention matching get_rotation_matrix.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # roll (x-axis)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    # yaw (z-axis)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return float(yaw), float(pitch), float(roll)


def angular_error_deg_batch(q_pred, q_true):
    """Mean angular error in degrees between two batches of unit quaternions.
    Handles double-cover: q and -q represent the same rotation.
    q_pred, q_true: torch tensors of shape (B, 4)
    Returns: tensor of shape (B,) with per-sample error in degrees.
    """
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot) * (180.0 / torch.pi)
```

- [ ] **Step 2: Verify helpers are reachable — quick smoke test in a Python shell**

```bash
cd /Users/ganapat0706/CEN571StarTracker
python3 -c "
import sys; sys.path.insert(0, 'FPGA_version/ml/scripts')
# import only the helpers without running __main__
import importlib.util, types
spec = importlib.util.spec_from_file_location('train', 'FPGA_version/ml/scripts/train.py')
# just parse-check
import ast
src = open('FPGA_version/ml/scripts/train.py').read()
ast.parse(src)
print('syntax OK')
"
```
Expected: `syntax OK`

---

## Task 2: Dataset Returns Quaternion Labels

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — `generate_labeled_sample`, `StarTrackerSyntheticDataset`

- [ ] **Step 1: In `generate_labeled_sample`, build quaternion from rotation matrix and add to return**

Find the line `metadata = {` and add quaternion computation just before it:

```python
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
```

- [ ] **Step 2: Update `StarTrackerSyntheticDataset.__init__` cache block to store quaternions**

In the `if self.cache_samples:` block, replace the inner loop with:

```python
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
```

- [ ] **Step 3: Update `StarTrackerSyntheticDataset.__getitem__` to return quaternion**

Replace the existing `__getitem__`:

```python
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
```

- [ ] **Step 4: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```
Expected: `OK`

---

## Task 3: Dual-Head Architecture with Star Count Injection

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — replace `StarTrackerTinyCNN`

- [ ] **Step 1: Replace the existing `StarTrackerTinyCNN` class entirely**

```python
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
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```
Expected: `OK`

---

## Task 4: Combined Loss + Updated `evaluate`

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — update `evaluate`, add `geodesic_loss`

- [ ] **Step 1: Add `geodesic_loss` helper after `angular_error_deg_batch`**

```python
def geodesic_loss(q_pred, q_true):
    """Quaternion geodesic loss: 0 when identical, 1 when orthogonal.
    Handles double-cover (q == -q) by taking abs of dot product.
    q_pred, q_true: (B, 4) unit quaternions
    """
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(0.0, 1.0)
    return (1.0 - dot).mean()
```

- [ ] **Step 2: Replace the existing `evaluate` function**

```python
def evaluate(model, loader, criterion, device, reg_lambda=0.5):
    """Compute validation loss, classification accuracy, and mean angular error."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    angle_errors = []

    with torch.no_grad():
        for images, labels, quats in loader:
            images = images.to(device)
            labels = labels.to(device)
            quats  = quats.to(device)

            cls_logits, q_pred = model(images)

            cls_loss = criterion(cls_logits, labels)
            reg_loss = geodesic_loss(q_pred, quats)
            loss = cls_loss + reg_lambda * reg_loss

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(cls_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            errs = angular_error_deg_batch(q_pred, quats)
            angle_errors.append(errs.cpu())

    mean_angle_err = torch.cat(angle_errors).mean().item()
    return running_loss / total, 100.0 * correct / total, mean_angle_err
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```

---

## Task 5: Curriculum Learning + Updated Training Loop

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — `train_model`

- [ ] **Step 1: Replace `train_model` signature and docstring**

```python
def train_model(
    num_samples=12000,
    num_epochs=100,
    batch_size=128,
    learning_rate=3e-3,
    camera_width=640,
    camera_height=480,
    model_width=160,
    model_height=120,
    fov_x_degrees=62.0,
    noise_prob=0.02,
    psf_sigma=1.5,
    reg_lambda=0.5,
    dataset_seed=42,
    catalog_seed=UNIVERSE_SEED,
):
    """
    Train the dual-head CNN with curriculum learning.

    Curriculum phases (split equally across num_epochs):
      Phase 1 — jitter=3°:  model sees tightly clustered classes, learns coarse features.
      Phase 2 — jitter=8°:  moderate overlap, refines decision boundaries.
      Phase 3 — jitter=15°: heavy overlap, forces robust quaternion regression.

    Combined loss: CrossEntropy(class) + reg_lambda * GeodesicLoss(quaternion)
    """
```

- [ ] **Step 2: Replace the body of `train_model` (everything from `star_catalog = ...` to `return model, star_catalog`)**

```python
    CURRICULUM_JITTERS = [3.0, 8.0, 15.0]
    phase_len = num_epochs // 3  # epochs per phase

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
            cache_samples=True,
            seed=dataset_seed + int(jitter * 10),
        )
        datasets.append(ds)

    num_workers = min(4, os.cpu_count() or 1)

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = StarTrackerTinyCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # Start with Phase 1 loaders
    current_phase = 0
    train_loader, val_loader = make_loaders(datasets[0])
    print(f"\n--- Curriculum Phase 1/3  jitter={CURRICULUM_JITTERS[0]}° ---")

    for epoch in range(num_epochs):
        # Switch curriculum phase
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
        val_loss, val_acc, val_ang = evaluate(model, val_loader, criterion, device, reg_lambda)

        print(
            f"Epoch [{epoch + 1:3d}/{num_epochs}] "
            f"Ph{current_phase + 1} "
            f"Loss:{train_loss:.4f} Acc:{train_acc:.1f}% AngErr:{train_ang:.1f}° | "
            f"Val Loss:{val_loss:.4f} Acc:{val_acc:.1f}% AngErr:{val_ang:.1f}°"
        )

    print("Training complete.")
    return model, star_catalog, val_loader
```

- [ ] **Step 3: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```

---

## Task 6: Temperature Scaling (Post-Training Calibration)

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — add `calibrate_temperature`, update `__main__`

- [ ] **Step 1: Add `calibrate_temperature` after `evaluate`**

```python
def calibrate_temperature(model, val_loader, device):
    """
    Find scalar temperature T that minimises NLL on the validation set.
    After calibration, softmax(logits / T) gives well-calibrated probabilities:
    a prediction reported as 90% confident should be correct ~90% of the time.
    Returns T as a Python float (store in manifest for use at inference time).
    """
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for images, labels, _quats in val_loader:
            cls_logits, _ = model(images.to(device))
            logits_all.append(cls_logits.cpu())
            labels_all.append(labels)

    logits = torch.cat(logits_all)   # (N, 6)
    labels = torch.cat(labels_all)   # (N,)

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
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```

---

## Task 7: Update `save_run_artifacts` and `__main__`

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py`

- [ ] **Step 1: Add `temperature` parameter to `save_run_artifacts` and write it to manifest**

Find the `manifest = {` dict inside `save_run_artifacts` and add after `"noise_prob"`:

```python
        "temperature": temperature,
        "reg_lambda": reg_lambda,
        "architecture": "dual_head_quaternion_v2",
```

Also add `temperature` and `reg_lambda` to the function signature:

```python
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
```

- [ ] **Step 2: Update `__main__` block — call new `train_model`, run calibration, pass T to artifacts**

Replace the `__main__` block:

```python
if __name__ == "__main__":
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    camera_width  = 640
    camera_height = 480
    model_width   = 160
    model_height  = 120
    frac_bits     = 8
    catalog_seed  = UNIVERSE_SEED
    dataset_seed  = 42
    fov_x_degrees = 62.0
    noise_prob    = 0.01
    reg_lambda    = 0.5

    model, star_catalog, val_loader = train_model(
        num_samples=12000,
        num_epochs=100,
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

    # Post-training calibration
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
        jitter_degrees=15.0,
        noise_prob=noise_prob,
        temperature=temperature,
        reg_lambda=reg_lambda,
    )
```

- [ ] **Step 3: Final syntax check of train.py**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('train.py OK')"
```
Expected: `train.py OK`

---

## Task 8: Update HLS Export for Dual-Head

**Files:**
- Modify: `FPGA_version/ml/scripts/train.py` — `export_to_hls_header`

The classification head's FC layers changed (now `fc_cls1` / `fc_cls2`). Update the export to use the correct attribute names and note the star-density feature is always 0 at export time (FPGA receives a pre-processed image, star density is not wired into the HLS kernel for now).

- [ ] **Step 1: Update `export_to_hls_header` to use new attribute names**

Replace the lines that extract `fc1_w`, `fc1_b`, `fc2_w`, `fc2_b`:

```python
    fc1_w = get_np(model_cpu.fc_cls1.weight)
    fc1_b = get_np(model_cpu.fc_cls1.bias)
    fc2_w = get_np(model_cpu.fc_cls2.weight)
    fc2_b = get_np(model_cpu.fc_cls2.bias)
```

Also update the `ST_FC1_IN` macro line since `feat_dim = 961` now (960 + 1 star density):

```python
        f.write(f"#define ST_FC1_IN  ({CNN_CONV3_OUT_CH} * {CNN_POOL_H} * {CNN_POOL_W} + 1)\n")
```

- [ ] **Step 2: Syntax check**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/train.py').read()); print('OK')"
```

---

## Task 9: Interactive Viewer — Rich Attitude Display

**Files:**
- Modify: `FPGA_version/ml/scripts/star_tracker_viewer.py`

The viewer needs to:
1. Load `temperature` from manifest (default 1.0 if absent)
2. Update `StarTrackerTinyCNN` to match new dual-head architecture (fc_cls1/fc_cls2 + fc_reg1/fc_reg2)
3. Display: calibrated confidence, quaternion (w,x,y,z), derived Euler angles, angular error vs slider truth, star count

- [ ] **Step 1: Replace `StarTrackerTinyCNN` in the viewer (must exactly mirror `train.py`)**

```python
class StarTrackerTinyCNN(nn.Module):
    """Dual-head CNN — must exactly mirror train.py architecture."""
    NUM_CLASSES = 6
    CONV1_OUT_CH = 16;  CONV2_OUT_CH = 32;  CONV3_OUT_CH = 64
    CONV1_K = 5;        KERNEL = 3;         STRIDE = 2
    CONV1_PAD = 2;      PAD = 1
    POOL_H = 3;         POOL_W = 5;         FC1_OUT = 128

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
```

- [ ] **Step 2: Update `load_model` to load temperature from manifest**

Replace the existing `load_model`:

```python
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
```

- [ ] **Step 3: Add `quaternion_to_euler_deg` helper to the viewer (copied from train.py)**

Add after the `get_rotation_matrix` function:

```python
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
    return float(yaw), float(pitch), float(roll)
```

- [ ] **Step 4: Replace `render_interactive` signature to accept temperature**

```python
def render_interactive(manifest, star_catalog, model, temperature, device,
                       camera_width=640, camera_height=480,
                       model_width=160, model_height=120):
```

- [ ] **Step 5: Replace the `update_view` callback and text display inside `render_interactive`**

Replace the inference block and text block inside `update_view`:

```python
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

        # Downscale to model resolution
        img_t = torch.from_numpy(img_full).unsqueeze(0).unsqueeze(0).to(device)
        img_r = F.interpolate(img_t, size=(model_height, model_width),
                               mode="bilinear", align_corners=False)

        # Inference
        with torch.no_grad():
            cls_logits, q_pred = model(img_r)
            # Calibrated probabilities
            probs = torch.softmax(cls_logits / temperature, dim=1)

        pred_cls  = int(torch.argmax(cls_logits, dim=1).item())
        pred_conf = float(probs[0, pred_cls].item()) * 100.0
        q_np      = q_pred[0].cpu().numpy()           # [w,x,y,z]

        # Euler angles from predicted quaternion
        cnn_yaw, cnn_pitch, cnn_roll = quaternion_to_euler_deg(q_np)

        # Angular error vs slider ground truth
        from math import cos, sin, sqrt
        def euler_to_quat(y_d, p_d, r_d):
            y, p, r = np.radians(y_d), np.radians(p_d), np.radians(r_d)
            cy, sy = cos(y/2), sin(y/2)
            cp, sp = cos(p/2), sin(p/2)
            cr, sr = cos(r/2), sin(r/2)
            return np.array([
                cr*cp*cy + sr*sp*sy,
                sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,
            ], dtype=np.float32)

        q_true    = euler_to_quat(yaw_deg, pitch_deg, roll_deg)
        dot       = abs(float(np.dot(q_np, q_true)))
        ang_err   = float(np.degrees(2 * np.arccos(min(dot, 1.0))))

        # Star count estimate from normalised image
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
```

- [ ] **Step 6: Update `__main__` in viewer to pass temperature to `render_interactive`**

```python
    if args.interactive:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model, temperature = load_model(manifest_path, device)
        render_interactive(manifest, star_catalog, model, temperature, device)
```

- [ ] **Step 7: Widen the text panel in `render_interactive` (more info needs space)**

Change:

```python
    ax_text = fig.add_axes([0.73, 0.06, 0.25, 0.15])
```
to:
```python
    ax_text = fig.add_axes([0.70, 0.04, 0.28, 0.22])
```

- [ ] **Step 8: Syntax check viewer**

```bash
python3 -c "import ast; ast.parse(open('FPGA_version/ml/scripts/star_tracker_viewer.py').read()); print('viewer OK')"
```
Expected: `viewer OK`

---

## Task 10: End-to-End Smoke Test (No GPU Required)

Run a tiny smoke-train (5 epochs, 120 samples) to verify all paths work before committing the full 100-epoch run.

- [ ] **Step 1: Run smoke test**

```bash
cd /Users/ganapat0706/CEN571StarTracker
python3 - <<'EOF'
import sys
sys.path.insert(0, 'FPGA_version/ml/scripts')

import random, numpy as np, torch
random.seed(1); np.random.seed(1); torch.manual_seed(1)

# Patch defaults for speed
import train as T
model, catalog, val_loader = T.train_model(
    num_samples=120, num_epochs=5, batch_size=32,
    learning_rate=1e-3, cache_samples=True,
)
temp = T.calibrate_temperature(model, val_loader, next(p.device for p in model.parameters()))
print(f"Smoke test passed. T={temp:.4f}")
EOF
```

Expected output contains: `Smoke test passed. T=`

- [ ] **Step 2: Verify quaternion output shape**

```bash
python3 - <<'EOF'
import sys; sys.path.insert(0, 'FPGA_version/ml/scripts')
import torch, train as T
model = T.StarTrackerTinyCNN()
model.eval()
x = torch.zeros(2, 1, 120, 160)
cls_logits, quat = model(x)
assert cls_logits.shape == (2, 6), f"Bad cls shape {cls_logits.shape}"
assert quat.shape == (2, 4), f"Bad quat shape {quat.shape}"
norms = (quat ** 2).sum(dim=1)
assert torch.allclose(norms, torch.ones(2), atol=1e-5), "Quaternion not unit!"
print("Shape + normalization check PASSED")
EOF
```

Expected: `Shape + normalization check PASSED`

- [ ] **Step 3: Commit**

```bash
cd /Users/ganapat0706/CEN571StarTracker
git add FPGA_version/ml/scripts/train.py \
        FPGA_version/ml/scripts/star_tracker_viewer.py \
        FPGA_version/ml/scripts/star_tracker.h \
        FPGA_version/ml/scripts/star_tracker.cpp \
        docs/superpowers/plans/
git commit -m "feat: dual-head attitude CNN — quaternion regression, curriculum learning, temperature calibration, rich viewer"
```

---

## Self-Review

**Spec coverage:**
- [x] Dual-head (classification + quaternion regression) — Task 3, 4, 5
- [x] Angular error metric — Task 4 (evaluate), Task 5 (training loop print), Task 9 (viewer)
- [x] Quaternion output — Task 1, 2, 3
- [x] Curriculum learning — Task 5
- [x] Star count injection — Task 3 (architecture forward pass)
- [x] Temperature scaling — Task 6, 7
- [x] Interactive viewer showing all values — Task 9
- [x] HLS export updated for new attribute names — Task 8

**Potential issues to watch:**
- `train_model` now returns 3 values `(model, star_catalog, val_loader)` — `__main__` updated accordingly in Task 7
- `load_model` in viewer now returns `(model, temperature)` — viewer `__main__` updated in Task 9, Step 6
- The smoke test in Task 10 imports `train` as a module; confirm no side effects run at import time (the `if __name__ == "__main__":` guard handles this)
