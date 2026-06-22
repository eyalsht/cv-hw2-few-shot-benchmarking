<div align="center">

# 🛰️ Few-Shot Classification with Frozen Foundation Models

### Benchmarking DINOv2 · CLIP · ConvNeXt across 3 classifier heads, 2 domains, and 4 shot settings

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.6%20%2B%20CUDA%2012.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.x-FFD21E?style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-executed-F37626?style=flat-square&logo=jupyter&logoColor=white" />
</p>
<p>
  <img src="https://img.shields.io/badge/experiments-72-22d3ee?style=flat-square" />
  <img src="https://img.shields.io/badge/episodes%20%2F%20config-10%2C000-f472b6?style=flat-square" />
  <img src="https://img.shields.io/badge/GPU-RTX%204070%20Super-a3e635?style=flat-square&logo=nvidia&logoColor=white" />
</p>

<br/>

<img src="assets/scaling.gif" width="92%" alt="Few-shot accuracy vs. number of shots — the backbone ranking flips between natural images and satellite imagery" />

</div>

---

## Overview

How well do **frozen** pre-trained vision backbones generalize to brand-new classes from just a
handful of labeled examples? This project answers that empirically. It runs a complete few-shot
benchmark — **3 backbones × 3 classifier heads × 2 datasets × 4 shot settings = 72 configurations**,
each averaged over **10,000 randomly sampled 5-way episodes** (`seed = 42`) — then adds a custom
**CLIP-Adapter** and tests whether a lightweight learned adapter can beat the classics.

<table>
<tr><td>

**Backbones** (frozen feature extractors)
- `DINOv2-small` — self-supervised ViT (Meta)
- `CLIP-RN50` — image-text contrastive (OpenAI)
- `ConvNeXt-Tiny` — ImageNet-supervised CNN

</td><td>

**Classifier heads** (fit per episode)
- Prototypical — nearest class-mean
- Ridge Regression — closed-form, regularized
- Linear Probing — `nn.Linear` + Adam

</td></tr>
</table>

**Datasets:** Stanford Cars (196 fine-grained classes, natural images) · EuroSAT (10 classes,
satellite imagery — a deliberate domain shift). **Episodes:** 5-way, K ∈ {1, 2, 4, 16} shots,
15 queries/class.

## 🔑 Key findings

> **1. Ridge Regression is the strongest head** — almost everywhere, and its lead *grows* with K.
> A well-regularized closed-form solver beats both the information-discarding Prototypical mean and
> the overfit-prone gradient-descent Linear probe in this small-data regime.

> **2. Backbone quality is domain-relative, not absolute.** CLIP-RN50 *dominates* natural-image
> Stanford Cars but is the **worst** backbone on satellite EuroSAT — pretraining distribution matters
> more than architecture for surviving domain shift. (This is the flip you see in the animation above.)

> **3. More parameters ≠ better few-shot.** A from-scratch per-episode CLIP-Adapter does **not** beat
> Ridge — an honest negative result: here the bottleneck is *data*, not model capacity.

#### Best accuracy per backbone (Ridge head, 16-shot)

| Backbone | Stanford Cars | EuroSAT | Δ (domain shift) |
|---|:---:|:---:|:---:|
| **CLIP-RN50** | **97.8%** 🥇 | 89.6% 🥉 | **−8.2 pts** |
| **DINOv2-small** | 96.1% | **92.2%** 🥇 | −3.9 pts |
| **ConvNeXt-Tiny** | 89.7% | 90.8% | **+1.1 pts** |

## How it works

```
Section 1 — Feature Extraction        Section 2 — Benchmarking
┌───────────────────────────┐         ┌────────────────────────────────────┐
│ images → frozen backbone  │   .pt   │ sample 5-way K-shot episode        │
│ → cache embeddings to disk│ ──────▶ │ → fit head on support              │
│   (6 cache files, ~350 MB)│         │ → score 15 queries/class           │
└───────────────────────────┘         │ → mean over 10,000 episodes        │
                                       └────────────────────────────────────┘
```

The 10,000-episode sweep is **vectorized across episodes** (a batched `(B, D, C)` solve instead of a
Python loop) — a ~50× speedup, verified bit-identical to the reference per-episode heads. The whole
benchmark runs in ~30–40 min on one GPU.

## Repo layout

```
cv_hw2_312359284_314884834.ipynb   self-contained, executed report (all results embedded)
benchmark_results.csv               72 baseline configs (mean / std accuracy)
adapter_results.csv                 24 CLIP-Adapter configs
make_hero_gif.py                    regenerates the animation above from the CSV
assets/scaling.gif                  hero animation
```

The six `{backbone}_{dataset}_features.pt` embedding caches (~350 MB) are **not** committed — the
notebook regenerates them on first run.

## Reproducing

```bash
pip install torch transformers datasets git+https://github.com/openai/CLIP.git \
            scikit-learn matplotlib seaborn pandas tqdm
jupyter nbconvert --to notebook --execute --inplace cv_hw2_312359284_314884834.ipynb
```

A restart-and-run-all reuses any cached `.pt`/`.csv` artifacts and just re-renders the analysis;
deleting them forces a full regeneration (`datasets` downloads Cars/EuroSAT automatically). One
Windows caveat: import `datasets` **before** `torch`/`clip` (OpenMP conflict) — documented in the
notebook's *Problems Encountered* log.

---

<div align="center">

**Computer Vision · Dr. Simon Korman · University of Haifa**
Imree Cohen · Eyal Shtinmetz

</div>
