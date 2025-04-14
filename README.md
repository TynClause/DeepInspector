# Deep Reinforcement Learning with CLIP‑GAN

> **Combine powerful vision‑language representations with generative environments to push the boundaries of autonomous agents.**

## Getting Started
### Prerequisites
* Windows 10/11 **or** Linux/macOS
* Python ≥ 3.12
* CUDA‑enabled GPU (optional but recommended)
* `git`, `pip`, and `virtualenv` or `conda`

### Quick Setup (Windows)
Clone the repo and run the batch script — it installs the Python environment and required packages:
```bat
> git clone https://github.com/TynClause/DeepInspector.git
> cd DeepInspector
> setup-user.bat
```

### Dataset Download
All datasets live under `./datasets`. Use the helper script to pull and prepare the **African Wildlife** collection (≈ 1.2 GB):
```powershell
> cd datasets
> python download_dataset.py --dataset african-wildlife --o .
```
The script automatically extracts archives and builds `train/`, `val/`, and `test/` splits.

---

