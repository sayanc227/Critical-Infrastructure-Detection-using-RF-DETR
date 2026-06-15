# 🏗️ Critical Infrastructure Detection using RF-DETR

Transformer-based RF-DETR model for drone & aerial surveillance detecting 19 classes of critical infrastructure (bridges, power plants, substations, towers, ports, etc.).

<div align="center">

<table>
  <tr>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/f2774356-3e9f-4079-af99-945f2cdccb0a" alt="Detection Example 1"/>
    </td>
    <td width="50%">
      <img src="https://github.com/user-attachments/assets/b080e1eb-a57c-46cc-bc8b-00a93e4fa179" alt="Detection Example 2"/>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Large-scale infrastructure detection from drone footage</em>
    </td>
    <td align="center">
      <em>Simultaneous multi-class detection in dense environments</em>
    </td>
  </tr>
</table>

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/Model-RF--DETR-orange.svg)]()
[![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-purple.svg)](https://roboflow.com/)
[![Label Studio](https://img.shields.io/badge/Label%20Studio-Annotation-ff6b6b.svg)](https://labelstud.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/167xgsFcpFqzVfAbT88IFdUYeM_X0h1um?usp=sharing)
[![Micro F1](https://img.shields.io/badge/Micro%20F1-92.4%25-brightgreen.svg)]()
[![Macro F1](https://img.shields.io/badge/Macro%20F1-83.5%25-yellowgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 Quick Start](#-how-to-run-the-project) • [📊 Results & Error Analysis](#-results--error-analysis) • [📂 Resources](#-resources--downloads)

</div>

---

## 📌 Project Overview

This project focuses on the **automated detection of 19 classes of critical infrastructure**—including **Bridges, Power Plants, Substations, Seaports, and Communication Towers**—from **aerial and drone imagery**.

Instead of relying solely on scarce or restricted public datasets, this work emphasizes a **Data-Centric AI approach** leveraging a **Synthetic Data Pipeline** with advanced **Generative AI models** to create rare and difficult-to-capture scenarios. All generated data was **manually reviewed and annotated** to ensure realism and label quality.

The detection model is built using **RF-DETR (Roboflow Detection Transformer)**, selected for its strong performance on small objects within large, cluttered aerial scenes and its suitability for near–real-time inference.

---

## 📊 Dataset Statistics

- **Total Images**: ~1,667 high-resolution aerial images
- **Classes**: 19 distinct infrastructure categories
- **Data Composition**: 60% Real-world / 40% Synthetic (Generative AI)
- **Annotation Format**: COCO JSON (Annotated via Label Studio)
- **Total Bounding Boxes**: ~4,500+ annotated instances

---

## 🛠️ Data Pipeline & Methodology

The primary value of this project lies in the **dataset design and curation strategy**, addressing the severe lack of diverse, multi-class aerial datasets in security-sensitive domains.

### 1️⃣ Data Gathering & Synthesis
* **Real Data (60%)**: Publicly accessible satellite imagery and permissible aerial/drone footage.
* **Synthetic Data (40%)**: Generated to simulate rare or operationally relevant viewpoints (oblique angles, low-light conditions, complex backgrounds).

**Generative AI tools used:**
* **Nano Banana Pro**: High-fidelity texture generation (e.g., rusted metal towers, cracked concrete bridges).
* **Seedream 4**: Complex environment generation (foggy terrain, dense vegetation, cluttered backgrounds).
* **ChatGPT**: Structured prompt engineering to enforce specific drone camera parameters (FOV, lighting, altitude, perspective).

### 2️⃣ Annotation Process
* **Tool:** Label Studio
* **Format:** COCO JSON
* **Target Classes:** 19 classes including *Bridge, Dam, Nuclear Reactor, Seaport, Mobile Harbour Cranes, Solar Power Plant*, etc.

---

## 🤖 Model Choice: RF-DETR

RF-DETR (Real-Time Detection Transformer) was selected over traditional CNN-based detectors (e.g., YOLO) due to its strengths in aerial surveillance scenarios:
* Transformer-based attention mechanisms help isolate infrastructure within large, cluttered scenes.
* Improved handling of small or distant objects.
* Modern architecture aligned with current research and deployment trends.

---

## 🧪 Training Configuration

* **Environment:** Google Colab Pro (NVIDIA T4 GPU)
* **Epochs:** 65
* **Batch Size:** 4
* **Optimizer:** AdamW

---

## 📊 Results & Error Analysis

### 🏆 Validation Performance Metrics
* **Micro-Average Precision**: **92.4%**
* **Micro-Average Recall**: **92.3%**
* **Macro-Average F1-Score**: **83.5%**

| Class | Precision | Recall | F1-Score | Primary Confusion |
| :--- | :---: | :---: | :---: | :--- |
| **Water Tower** | 0.991 | 0.957 | **0.974** | Satellite Dish |
| **Oil Refinery** | 0.966 | 0.955 | **0.960** | - |
| **Satellite Dish / Ground Station**| 0.937 | 0.974 | **0.955** | Water Tower |
| **Nuclear Reactor** | 0.916 | 0.936 | **0.926** | Energy Storage |
| **Mobile Tower** | 0.925 | 0.925 | **0.925** | Transmission Tower |
| **Bridge** | 0.888 | 0.946 | **0.916** | Dam |
| **Wind Turbine** | 0.907 | 0.910 | **0.909** | - |
| **Airport Runway** | 0.892 | 0.922 | **0.907** | Solar Power Plant |
| **Transmission Tower** | 0.878 | 0.892 | **0.885** | Thermal Power Plant |
| **Solar Power Plant** | 0.861 | 0.861 | **0.861** | Airport Runway |
| **Electrical Substation** | 0.857 | 0.842 | **0.850** | - |
| **Dam** | 0.835 | 0.862 | **0.848** | Bridge |
| **Energy Storage Infrastructure** | 0.856 | 0.837 | **0.846** | Nuclear Reactor |
| **Thermal Power Plant** | 0.848 | 0.838 | **0.843** | - |
| **Cooling Tower** | 0.680 | 0.944 | **0.791** | - |
| **Seaport** | 0.664 | 0.916 | **0.770** | Mobile Harbour Cranes |
| **Cargo Ship** | 0.757 | 0.750 | **0.753** | Seaport |
| **Mobile Harbour Cranes** | 0.745 | 0.708 | **0.726** | Seaport |
| **Shipping Containers** | 0.655 | 0.442 | **0.528** | Airport Runway |

### 🔍 Deep Dive: Confusion Matrix & Error Analysis
Analyzing the confusion matrix revealed critical insights into how aerial perspectives affect object detection:

1. **Seaport ↔ Mobile Harbour Cranes**: High mutual confusion. In top-down drone views, the structural footprint of mobile harbour cranes heavily overlaps with general seaport infrastructure, making boundary delineation difficult.
2. **Shipping Containers → Airport Runway**: False positives observed. Neatly arranged shipping container arrays in low-resolution aerial views can mimic the geometric lines and textures of airport runways.
3. **Energy Storage Infrastructure ↔ Nuclear Reactor**: Mutual confusion due to the extreme geometric similarity of large cylindrical storage tanks and nuclear containment domes when viewed from a nadir (top-down) perspective.
4. **Bridge ↔ Dam**: Confusion occurs when long concrete structures span across water bodies, sharing similar structural and shadow profiles from high altitudes.

---

## 🚀 How to Run the Project

This project is designed to be **fully reproducible via Google Colab**.

🔗 **Training & Inference Notebook:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/167xgsFcpFqzVfAbT88IFdUYeM_X0h1um?usp=sharing)

### Steps:
1. Open the Colab notebook using the link above.
2. Connect to a GPU runtime (T4 or better recommended).
3. The notebook handles installation of RF-DETR, dataset download, and setup.
4. Run the cells to train the model and visualize inference results.

---

## 📂 Resources & Downloads

* **Trained Weights:** Available via the Colab notebook.
* **Dataset:** Provided through Roboflow Universe (link inside notebook).

---

## 🎯 Use Cases

- 🏛️ **Defense & Security**: Critical infrastructure monitoring and threat detection.
- 🗺️ **Urban Planning**: Infrastructure mapping, zoning, and assessment.
- 🚨 **Disaster Response**: Rapid damage assessment after natural disasters.
- 📊 **Research**: Geospatial analysis and multi-modal infrastructure studies.

---

## 🤝 Acknowledgements

This project builds upon:
- [Original RF-DETR Paper](https://arxiv.org/abs/2303.10845) & [GitHub Repository](https://github.com/liming-ai/RF-DETR)
- [Roboflow](https://roboflow.com/) for dataset management and augmentation tools.
- [Supervision](https://github.com/roboflow/supervision) library for visualization.
- [Label Studio](https://labelstud.io/) for the annotation platform.
- [PyTorch](https://pytorch.org/) deep learning framework.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Sayan C** - [13sayanc@gmail.com](mailto:13sayanc@gmail.com)

Project Link: [https://github.com/sayanc227/Critical-Infrastructure-Detection-using-RF-DETR](https://github.com/sayanc227/Critical-Infrastructure-Detection-using-RF-DETR)

---

<div align="center">

⭐ **Star this repo if you find it useful!** ⭐

</div>
