# Star Tracker Interactive Viewer Guide

## Overview
The interactive viewer lets you rotate a virtual camera in 3D space and watch the trained CNN predict spacecraft attitude in real-time. Perfect for validating the model and testing FPGA camera alignment!

## Launch

### Static 6-attitude grid (default)
Shows the six reference attitudes (Up, Down, Left, Right, Forward, Backward):
```bash
python ml/scripts/star_tracker_viewer.py
```

### Interactive viewer with sliders (new!)
Control pitch/roll/yaw/fov with real-time CNN predictions:
```bash
python ml/scripts/star_tracker_viewer.py --interactive
```

### Save static grid as image
```bash
python ml/scripts/star_tracker_viewer.py --output viewer.png
```

### Use specific training run
```bash
python ml/scripts/star_tracker_viewer.py --interactive --run-id 20260418_162634_seed1234
```

---

## Interactive Viewer Controls

### Sliders
- **Yaw (blue)**: ±180° rotation around Z-axis (left/right spin)
- **Pitch (green)**: ±90° rotation around Y-axis (up/down tilt)
- **Roll (red)**: ±180° rotation around X-axis (barrel roll)
- **FOV (purple)**: 30–120° field of view

### Display
- **Top**: Live star field rendering (640×480 camera resolution)
  - Shows realistic star magnitudes (brightness varies)
  - PSF blur simulates camera optics
- **Bottom right**: CNN prediction info
  - Current spacecraft attitude (Yaw/Pitch/Roll in degrees)
  - Predicted class (Up, Down, Left, Right, Forward, Backward)
  - Confidence % (how sure the model is)

---

## Use Case: FPGA Camera Validation

### Goal
Verify that your FPGA-mounted camera outputs the correct attitude when physically aligned with the laptop screen.

### Procedure
1. **Launch interactive viewer**:
   ```bash
   python ml/scripts/star_tracker_viewer.py --interactive
   ```

2. **Set laptop screen to known attitude** (e.g., Yaw=45°, Pitch=30°):
   - Use the sliders to configure the desired attitude
   - Note the predicted class and confidence
   - CNN should output the expected class with high confidence

3. **Physical alignment test**:
   - Hold the FPGA camera pointing at the laptop screen
   - Rotate the camera/screen to match the slider configuration
   - Read the FPGA serial output or LED display
   - **PASS**: FPGA output matches CNN prediction (same class ID)
   - **FAIL**: Outputs differ → debug camera orientation or model

4. **Iterate through attitudes**:
   - Test extreme angles (pitch=±90°, yaw=±180°)
   - Test intermediate poses (pitch=45°, yaw=60°, roll=30°)
   - Document which attitudes have high/low confidence

---

## Tips

### Star Rendering Quality
- **Magnitude**: Brighter stars are more prominent (exponential distribution)
- **PSF**: Stars appear blurred (σ=1.5 pixels), mimicking real optics
- Adjust FOV to see trade-offs (wider FOV = fewer bright stars, narrower FOV = denser star field)

### Model Confidence
- **High confidence (>80%)**: Model is certain → reliable for deployment
- **Medium confidence (50–80%)**: Model is learning this attitude
- **Low confidence (<50%)**: Model struggling → consider more training or jitter

### FPGA Integration
- The screen-based virtual camera can validate attitude estimation
- Compare CNN predictions with ground-truth FPGA measurements
- Use mismatches to debug camera calibration or quantization errors

---

## Example Workflow

```bash
# 1. Train with improved rendering
python ml/scripts/train.py

# 2. View static grid of 6 reference attitudes
python ml/scripts/star_tracker_viewer.py

# 3. Launch interactive viewer for detailed exploration
python ml/scripts/star_tracker_viewer.py --interactive

# 4. Adjust sliders to match your FPGA camera's current view
# 5. Verify CNN prediction matches your expected attitude
# 6. Move the physical FPGA camera to match screen angle
# 7. Check if FPGA output now matches CNN prediction
```

---

## Architecture Notes

- **Star Catalog**: 500 stars on unit sphere with magnitudes
- **Camera Model**: Pinhole projection (640×480, FOV ~62°)
- **Model**: Tiny CNN (Conv1→ReLU→Conv2→ReLU→GAP→Linear→6 classes)
- **Rendering**: Magnitude-based brightness + Gaussian PSF blur
- **Inference**: Real-time on CPU (takes ~10ms per frame)

