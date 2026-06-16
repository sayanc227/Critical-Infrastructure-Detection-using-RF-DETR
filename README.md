# 🛰️ Critical Infrastructure Detection using RF-DETR

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/Architecture-RF--DETR-orange.svg)](https://github.com/roboflow/rf-detr)
[![Dataset](https://img.shields.io/badge/Dataset-19%20Classes-blue.svg)]()
[![Best mAP50-95](https://img.shields.io/badge/Best%20mAP50--95-0.531-brightgreen.svg)]()
[![Data Centric](https://img.shields.io/badge/Workflow-Data%20Centric-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 🎥 Model Detection Showcase

<div align="center">
<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/f900634b-fb34-4934-bec3-2ceb1be9f891" alt="Satellite Ground Station Detection"/>
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/573caae3-3bac-4706-ba1d-982900799da8" alt="Nuclear Reactor Detection"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>RF-DETR detection of Satellite Ground Stations from aerial imagery</em>
    </td>
    <td align="center">
      <em>RF-DETR detection of Nuclear Reactor infrastructure from aerial imagery</em>
    </td>
  </tr>
</table>
</div>

<p align="center">
  <i>All detections shown above were generated using the fine-tuned RF-DETR model developed in this project.</i>
</p>

---
## 🎯 Project Overview

This project explores aerial and drone-based detection of critical infrastructure using **RF-DETR (Roboflow Detection Transformer)**.

The objective was to build a robust object detection system capable of identifying multiple infrastructure categories from high-altitude imagery while addressing one of the biggest challenges in computer vision projects:

> **Data Scarcity**

Rather than focusing solely on model architecture, this project follows a **data-centric workflow**, emphasizing:

✅ Dataset Creation
✅ Annotation Quality Control
✅ Synthetic Data Generation
✅ Error Analysis
✅ Iterative Retraining
✅ Class-Specific Optimization

---

## 🏗️ Infrastructure Classes

The model was trained to detect **19 infrastructure categories**:

| Infrastructure                | Infrastructure                  |
| ----------------------------- | ------------------------------- |
| Airport Runway                | Oil Refinery                    |
| Bridge                        | Satellite Dish / Ground Station |
| Cargo Ship                    | Seaport                         |
| Cooling Tower                 | Shipping Containers             |
| Dam                           | Solar Power Plant               |
| Electrical Substation         | Thermal Power Plant             |
| Energy Storage Infrastructure | Transmission Tower              |
| Mobile Tower                  | Water Tower                     |
| Nuclear Reactor               | Wind Turbine                    |
| Mobile Harbour Cranes         |                                 |

---

## 📊 Dataset Development

### Hybrid Dataset Strategy

The dataset was created using a combination of:

* Real aerial imagery
* Satellite imagery
* Synthetic imagery

## 📊 Dataset Development

### Workflow Overview

```text
┌──────────────────────┐
│ Real Aerial Imagery  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Synthetic Imagery    │
│ (Nano Banana Pro,    │
│ Seedream 4)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Manual QA & Review   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Label Studio         │
│ Annotation           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ COCO Dataset Export  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ RF-DETR Training     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Evaluation &         │
│ Error Analysis       │
└──────────────────────┘
```

### Synthetic Data Workflow

To address limited availability of infrastructure imagery from rare aerial viewpoints, synthetic samples were generated using:

* Nano Banana Pro
* Seedream 4
* ChatGPT-assisted prompt engineering

Synthetic data was used to create:

* Oblique drone perspectives
* Complex environmental conditions
* Low-light scenarios
* Rare infrastructure viewpoints
* Additional examples for underrepresented classes

---

## 🧹 Data Cleaning & Quality Control

All images underwent manual review before training.

### Quality Assurance Checklist

✅ Removal of unrealistic synthetic outputs
✅ Visual inspection for generation artifacts
✅ Annotation consistency checks
✅ Duplicate image filtering
✅ Class balance verification
✅ Manual edge-case review

The goal was to maximize annotation quality before model training rather than relying solely on augmentation techniques.

---

## 🏷️ Annotation Pipeline

### Annotation Tool

**Label Studio**

### Annotation Format

**COCO JSON**

### Annotation Methodology

* Manual bounding-box annotation
* Class-level verification
* Edge-case review
* Annotation consistency validation

### High-Risk Confusion Pairs

| Category A            | Category B                    |
| --------------------- | ----------------------------- |
| Thermal Power Plant   | Nuclear Reactor               |
| Cargo Ship            | Mobile Harbour Cranes         |
| Shipping Containers   | Airport Runway Structures     |
| Electrical Substation | Energy Storage Infrastructure |

These categories received additional review due to visual similarity and contextual overlap.

---

## 🤖 Model Architecture

### RF-DETR (Roboflow Detection Transformer)

RF-DETR was selected because transformer-based detectors can leverage broader contextual information across aerial scenes.

This is particularly useful when distinguishing visually similar infrastructure categories where surrounding environmental context contributes to classification accuracy.

### Training Environment

* Google Colab
* PyTorch
* RF-DETR
* CUDA GPU Acceleration

### Training Strategy

✅ Transfer Learning
✅ Multi-Stage Fine-Tuning
✅ Early Stopping
✅ EMA Evaluation
✅ Targeted Retraining of Weak Classes

---

## 📈 Results

### Best Validation Performance

| Metric         | Value          |
| -------------- | -------------- |
| Best mAP50-95  | **0.531**      |
| Best Model     | EMA Checkpoint |
| Early Stopping | Enabled        |

### Performance Summary

The final model demonstrated strong performance on visually distinctive infrastructure classes while highlighting several challenging categories that remain active areas of improvement.

---

### 🌟 Strong Performing Categories

| Category                        |
| ------------------------------- |
| Oil Refinery                    |
| Water Tower                     |
| Satellite Dish / Ground Station |
| Mobile Tower                    |
| Transmission Tower              |

---

### ⚠️ Challenging Categories

| Category               |
| ---------------------- |
| Shipping Containers    |
| Electrical Substations |
| Solar Power Plants     |
| Dams                   |

Common challenges include:

* Small object size
* Dense object clustering
* High visual similarity
* Scale variation across aerial viewpoints

---

## 🔍 Key Findings

A major outcome of this project was confirming that:

> **Dataset quality, annotation consistency, and targeted data collection produced larger gains than additional training epochs alone.**

Performance improvements were primarily achieved through:

* Additional class-specific data collection
* Synthetic data augmentation
* Error analysis
* Annotation refinement
* Focused retraining of underperforming classes

This reinforced the importance of a **data-centric AI workflow** over purely model-centric optimization.

---

## 📂 Repository Structure

```text
Critical-Infrastructure-Detection-using-RF-DETR/
├── assets/
├── dataset_info/
├── notebooks/
├── weights/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 Resources

### Training Notebook

🔗 Google Colab

https://colab.research.google.com/drive/167xgsFcpFqzVfAbT88IFdUYeM_X0h1um

### Model Weights

Available through the repository Releases section.

---

## 🛣️ Future Work

Planned improvements include:

* Hard-negative mining
* Expansion of Shipping Container samples
* Improved Substation annotation consistency
* Video-based infrastructure tracking
* Integration with segmentation pipelines
* Multi-object tracking support

---

## 👨‍💻 Author

### Sayan Chakraborty

**Computer Vision Practitioner | Data Annotation Specialist**

📧 Open to collaboration on:

* Computer Vision
* Object Detection
* Dataset Development
* Annotation Workflows
* Synthetic Data Generation

**GitHub**
https://github.com/sayanc227

**LinkedIn**
https://www.linkedin.com/in/sayan-chakraborty-595a45382

---
