# CEN571StarTracker

This repository is split into two parts:

- `Baseline_CPU_version/` for the original CPU feature-based star tracker pipeline.
- `FPGA_version/` for the CNN-based FPGA-oriented pipeline, viewer, and HLS C++ sources.

## Python setup

Install the dependencies for the part you want to work on:

```bash
python -m pip install -r Baseline_CPU_version/requirements.txt
python -m pip install -r FPGA_version/requirements.txt
```

Use a local virtual environment if you want isolation, but do not commit `.venv/` to the repo.

## Running the FPGA setup
1) In ./FPGA_Version/ml, upload the jupyter notebook named "star_tracker_fpga.ipynb" and the "startracker.bit" and "startracker.hwh" files into a PYNQ Z2 FPGA.
2) Ensure that these files are within the same directory.
3) Plug in a Logitech C270 camera and run all the cells in order the jupyter notebook, and stop at cell 9, which should be the live camera output.
4) Run the command ```bash python startracker_viewer.py```
5) Point the camera at each one of the images to get an classification output onto the FPGA

## Notes

- `FPGA_version/ml/models/star_tracker_weights.h` is tracked because the HLS C++ sources include it directly.
- Generated training artifacts such as `.pth`, `.json`, and `.npy` files are ignored and can be regenerated locally.
- Notebook checkpoints, Python caches, and build outputs are ignored by default.
