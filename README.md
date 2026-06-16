# 🏗️ Critical Infrastructure Detection via RF-DETR & Synthetic Data Pipelines

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/Architecture-RF--DETR-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-4.0k%20Objects-blue.svg)]()
[![Overall F1](https://img.shields.io/badge/Overall%20F1-90.17%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade, data-centric computer vision pipeline leveraging **RF-DETR (Real-Time Detection Transformer)** to detect and localize **19 classes of critical infrastructure** in high-altitude aerial and drone imagery. 

This repository demonstrates an end-to-end engineering workflow designed to overcome real-world **data scarcity** and **geometric class confusion** using a hybrid synthetic data pipeline, strict annotation QA, and targeted error analysis.



<div align="center">
<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/f900634b-fb34-4934-bec3-2ceb1be9f891" alt="Satellite Ground Station Thermal Tracking"/>
    </td>
    <td width="50%">
      <!-- TODO: Convert your nuclear reactor video to a .gif file using FFmpeg and upload it here -->
      <img src="https://github.com/user-attachments/assets/573caae3-3bac-4706-ba1d-982900799da8" alt="Nuclear Reactor Aerial Detection"/>
    </td>
  </tr>
  <tr>
    <td align="center"><em>Thermal tracking and localization of Satellite Ground Stations</em></td>
    <td align="center"><em>Detection of Nuclear Reactors with high localized precision</em></td>
  </tr>
</table>
</div>
---



## 📌 Production Challenges & Architecture Selection

Deploying object detection models for aerial surveillance introduces distinct technical challenges:
1. **Severe Data Scarcity:** Real-world aerial imagery of restricted infrastructure (e.g., nuclear reactors, military-adjacent seaports) is heavily regulated or unavailable.
2. **High Scale Variance & Clutter:** Small, localized objects (e.g., mobile harbour cranes) must be detected within massive, high-resolution $1024 \times 1024$ views.
3. **High Perspective Distortions:** Nadir (top-down) and oblique (angled) drone perspectives distort classic structural features.

### Why RF-DETR?
While CNN-based architectures (like YOLOv8) perform well in standard localized settings, **RF-DETR** was selected for this production pipeline due to its transformer-based global attention mechanism. This allows the model to retain long-range contextual cues—crucial for distinguishing between structurally similar objects (e.g., Dams vs. Bridges) based on their environmental surroundings.

---

## 🛠️ Data Engineering & Curation Pipeline

The core value of this project lies in its **data-centric methodology**. Instead of relying purely on public data, a rigorous curation and synthesis workflow was built.
