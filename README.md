# CV Assignment 2 — Benchmarking Foundation Models for Few-Shot Classification

Computer Vision · Dr. Simon Korman · TA: Jerry Abu Ayoub · University of Haifa. Two-person submission.
**Authors:** Imree Cohen (312359284) · Eyal Shtinmetz (314884834).
**Deliverable:** one self-contained, executed [`cv_hw2_312359284_314884834.ipynb`](cv_hw2_312359284_314884834.ipynb)
(all code inline, all results embedded). Due 28/6/2026.

## What it does

Benchmarks **3 frozen backbones** (DINOv2-small, CLIP-RN50, ConvNeXt-Tiny) × **3 classifier heads**
(Prototypical, Ridge Regression, Linear Probing) × **2 datasets** (Stanford Cars, EuroSAT) ×
**K = 1, 2, 4, 16 shots** — 72 configurations, each averaged over **10,000** 5-way episodes
(`seed = 42`) — then adds a per-episode **CLIP-Adapter** (Task 5) under the identical protocol.

The notebook is structured as the assignment requires: **Section 1 — Feature Extraction** (cache
per-image embeddings to disk) and **Section 2 — Benchmarking** (sampler, heads, the 72-config sweep,
analysis, and the Task 5 extension), and reads as a self-contained report with every table/plot
embedded.

## Headline findings

- **Ridge Regression is the strongest head** almost everywhere, and its lead grows with K.
- **Backbone quality is domain-relative:** CLIP-RN50 wins Stanford Cars but is the *worst* backbone
  on satellite EuroSAT, where the less natural-photo-specialized DINOv2/ConvNeXt transfer better.
- **More parameters ≠ better few-shot:** the from-scratch per-episode CLIP-Adapter does not beat Ridge
  — an honest negative result; in this regime the bottleneck is data, not model capacity.

## Layout

```
cv_hw2_312359284_314884834.ipynb   the submission (executed, results embedded)
benchmark_results.csv               72 baseline configs (mean/std accuracy)
adapter_results.csv                 24 CLIP-Adapter configs (Task 5)
```

The six `{backbone}_{dataset}_features.pt` embedding caches (~350 MB) are **not** committed; the
notebook regenerates them on first run.

## Reproducing

A clean restart-and-run-all reuses any cached `.pt`/`.csv` artifacts and just re-renders the analysis.
Deleting them forces full regeneration (`datasets` downloads Cars/EuroSAT automatically; the full
sweep takes ~30–40 min on an RTX 4070 Super). One environment caveat on Windows: `datasets` must be
imported **before** `torch`/`clip` (OpenMP conflict) — see the *Problems Encountered* log in the
notebook.
