# Deep Reinforcement Learning with CLIP‑GAN

> **Combine powerful vision‑language representations with generative environments to push the boundaries of autonomous agents.**

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   * [Prerequisites](#prerequisites)
   * [Quick Setup (Windows)](#quick-setup-windows)
   * [Dataset Download](#dataset-download)
5. [Training](#training)
6. [Evaluation & Visualization](#evaluation--visualization)
7. [Contributing](#contributing)
8. [License](#license)
9. [Citation](#citation)

---

## Overview
**CLIP‑GAN** marries the contrastive vision‑language power of [CLIP](https://openai.com/research/clip) with the sample‑efficiency of modern **Deep Reinforcement Learning (DRL)** algorithms.  
Agents perceive the world through CLIP embeddings, generate candidate frames with a lightweight GAN, and learn behaviours with Proximal Policy Optimization (PPO). The result is a scalable framework for vision‑conditioned policy learning on complex image domains such as *African Wildlife*.

<p align="center">
  <img src="docs/architecture.svg" width="600" alt="High‑level architecture diagram"/>
</p>

---

## Features
- **Modular pipeline** — plug‑and‑play encoders, GAN generators and RL algorithms.
- **Dataset‑agnostic** — train on any image collection; a helper script handles downloads & preprocessing.
- **Mixed‑precision** training for faster iteration.
- **TensorBoard & Weights‑and‑Biases** logging baked in.
- **Unit‑tested** core components with >90 % coverage.

---

## Project Structure
```text
├── datasets/                # Raw & processed image sets
│   └── download_dataset.py  # Helper script (see below)
├── clip_gan/                # Library source code
│   ├── models/              # CLIP encoders, GAN, policy networks
│   ├── rl/                  # PPO, A2C, replay buffers, wrappers
│   └── utils/               # Common helpers & configs
├── configs/                 # YAML experiment configs
├── scripts/                 # Convenience shell & batch files
├── setup-user.bat           # One‑click environment setup for Windows
├── docs/                    # Diagrams & paper links
└── tests/                   # PyTest suites
```

---

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

## Training
In progress

## Evaluation & Visualization
In Progress

## Contributing
In Progress

## License
In Progress

## Citation
In Progress

