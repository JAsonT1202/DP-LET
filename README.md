# DP-LET

> **DP-LET: An Efficient Spatio-Temporal Network Traffic Prediction Framework**  
> 📄 *Under Review at GLOBECOM 2025*

DP-LET is a novel deep learning framework for accurate and efficient **spatio-temporal network traffic prediction**. It combines **denoising-based preprocessing**, **temporal convolution for local patterns**, and **Transformers for global modeling**. Designed for real-world cellular traffic, DP-LET achieves **state-of-the-art performance** while maintaining low computational overhead.

---

## 🚀 Highlights

- 📉 **TSVDR-Based Preprocessing**  
  Removes noise and decouples spatial dependencies with minimal memory usage.

- 🧠 **Local Feature Enhancement (TCN)**  
  Extracts fine-grained temporal features using multi-branch Temporal Convolutional Networks.

- 🔁 **Transformer Prediction**  
  Captures long-range dependencies with efficient self-attention.

- ⚡ **Lightweight & Accurate**  
  Reduces **MSE by 31.8%** and **MAE by 23.1%** compared to other SOTA models.

---

## 🏗️ Architecture Overview

![DP-LET Architecture](./ours.png)
---

## 📊 Performance (Milan CDR Dataset)

| Model         | MSE (T=72) | MAE (T=72) | MSE (T=144) | MAE (T=144) |
|---------------|------------|------------|-------------|-------------|
| **DP-LET**    | **0.225**  | **0.324**  | **0.238**   | **0.333**   |
| TimeMixer     | 0.230      | 0.329      | 0.255       | 0.347       |
| PatchTST      | 0.234      | 0.337      | 0.249       | 0.350       |
| iTransformer  | 0.237      | 0.333      | 0.256       | 0.346       |
| FEDformer     | 0.387      | 0.468      | 0.408       | 0.485       |
| Autoformer    | 0.429      | 0.506      | 0.443       | 0.513       |
| Informer      | 0.653      | 0.615      | 0.749       | 0.654       |

