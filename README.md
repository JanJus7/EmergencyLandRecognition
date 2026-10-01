# Emergency Aircraft Landing Zone Recognition (ResNet-18 / EuroSAT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

A deep learning Proof of Concept (PoC) designed to evaluate terrain suitability for emergency off-field aircraft landings from satellite imagery. Built using PyTorch, fine-tuned ResNet-18, and the multi-spectral EuroSAT benchmark.

> **Note on Documentation:** The formal Bachelor's thesis (`thesis/pracaLic.pdf`) is provided in Polish in accordance with University of Gdańsk degree regulations. All codebase components, inline documentation, and project architecture are maintained in English.

---

## Problem & Motivation
During forced off-field landing scenarios (e.g., dual engine failure), flight crews have mere seconds to analyze surroundings and select a touchdown site. High cognitive load and cockpit stress can result in misjudging terrain geometry and surface obstacles.

This project demonstrates an automated decision-support pipeline that categorizes satellite imagery into land cover classes and projects color-coded landing suitability heatmaps calibrated to specific aircraft kinetic energy profiles.

---

## Dataset & Spatial Ground Calibration
* **Dataset:** [EuroSAT](https://github.com/phelber/eurosat) (European Space Agency, Sentinel-2 optical imagery).
* **Data Volume:** 27,000 images uniformly balanced across 10 Land Use and Land Cover (LULC) classes.
* **Ground Sample Distance (GSD):** 10 meters per pixel. A raw 64×64 px matrix represents a 640 m × 640 m terrain patch.
* **Spatial Rescaling:** External evaluation imagery (e.g., Google Earth exports) was rescaled to match EuroSAT ground resolution using reference scale factor $P = \frac{224}{L_{px}} \cdot 100\%$, where $L_{px}$ is the pixel span representing 2240 m on the ground.
* **Data Split:** Strict 70% training, 15% validation, and 15% testing split (4,050 isolated test images).

---

## Model Pipeline & Fine-Tuning
* **Backbone:** `ResNet-18` initialized with pre-trained ImageNet weights.
* **Transfer Learning Strategy:** Early convolutional layers (extracting general Gabor filters and low-level textures) were frozen. Layers 3 and 4 were fine-tuned to capture domain-specific remote sensing features.
* **Classification Head:** Replaced final fully connected layer with a 10-class linear projection head.
* **Optimization:** Adam optimizer with conservative learning rate ($10^{-4}$) to prevent catastrophic forgetting.
* **Loss & Regularization:** Cross-entropy loss with early stopping (patience = 5 epochs), halting training at epoch 15 (min val loss: 0.055).

---

## Aircraft Kinetic Energy & Suitability Mapping
Touchdown survivability is strictly governed by kinetic energy:
$$E_k = \frac{1}{2} m v^2$$

* **Heavy Commercial Jet (e.g., Boeing 737: ~70 t, landing speed ~250 km/h):** Massive kinetic energy forces extreme penalization of soft ground (plowed agricultural fields risk nose gear collapse, structural rollover, and hull rupture).
* **Light General Aviation (e.g., Cessna 152: ~700 kg, landing speed ~120 km/h):** Low kinetic energy enables survivable landings on pastures, grass strips, and even soft agricultural ground.

Multi-class classification vectors are converted via heuristic cost matrices into localized landing heatmaps:
* 🟦 / 🟩 **High Suitability:** Pastures, flat low-friction fields.
* 🟥 **Critical Hazard:** Forests, residential zones, industrial plants, deep water bodies.

### Visual Comparison: Boeing 737 vs. Cessna 152
| Boeing 737 (High Risk / Soft Ground Penalized) | Cessna 152 (Wider Field Survivability) |
|:---:|:---:|
| ![B737 Gdańsk](thesis/V3/landing_heatmap_b737_kromeriz.png) | ![C152 Gdańsk](thesis/V3/landing_heatmap_c152_kromeriz.png) |

---

## Results & Proof of Concept Limitations

### 1. Benchmark Test Metrics
* **Isolated Test Accuracy:** **98.0%** across 4,050 unseen EuroSAT tiles.
* Near-zero false positives between high-risk classes (Forest, Water) and open landing areas (Pasture). Minor confusions occurred primarily between structurally identical agrarian classes (Annual Crop vs. Permanent Crop).

### 2. Real-World Inference Limitations (Sliding Window)
* **Linear Infrastructure Resolution:** Due to EuroSAT's 10 m/pixel resolution, narrow highways (typically 20–30 m wide) fail to register as viable strips and are often suppressed by surrounding terrain.
* **Tile Edge Artifacts:** Continuous sliding window analysis across large geographical rasters introduces mixed-class edge effects where boundary tiles contain combinations of forest, urban zones, and open fields.
* **Future Work:** Transitioning from a discrete classification sliding window to semantic segmentation (e.g., U-Net), integrating high-resolution aerial orthophotos, and feeding real-time meteorological data (wind vectors, surface friction).

---

## Repository Structure

```text
EmergencyLandRecognition/
├── model/
│   ├── eval.py                        # Computes test-set confusion matrix
│   ├── plotMetrics.py                 # Generates loss and accuracy curves
│   ├── predict.py                     # Sliding window inference & heatmap generation
│   ├── prepareDataset.py              # EuroSAT 70:15:15 splitting & transforms
│   ├── requirements.txt               # Dependencies (PyTorch, torchvision, matplotlib, etc.)
│   ├── saveAll.py                     # Model checkpoint & metric exporter
│   └── train.py                       # Training loop with Early Stopping
├── thesis/
│   ├── V3/                            # Output landing heatmaps (B737 vs C152 png files)
│   ├── confusion_matrix_V3.pdf        # Test set confusion matrix
│   ├── learning_curves_clean.pdf      # Training and validation loss/acc curves
│   ├── pracaLic.pdf                   # Complete Bachelor's thesis PDF (compiled)
│   └── pracaLic.tex                   # LaTeX documentation source
├── .gitignore
├── LICENSE
└── README.md
```
---

## Usage

### 1. Environment Setup
```bash
git clone https://github.com/JanJus7/EmergencyLandRecognition.git
cd EmergencyLandRecognition/model
pip install -r requirements.txt
```

### 2. Model Training
Train the ResNet-18 pipeline on EuroSAT with early stopping:
```bash
python train.py
```
or simply download the weights from Releases tab and place them in the ```model/``` folder.

### 3. Evaluation & Metrics
Generate the confusion matrix and loss/accuracy curves:
```bash
python eval.py
python plotMetrics.py
```

### 4. Terrain Inference & Heatmap Generation
Execute the sliding-window classification pipeline on satellite rasters:
```bash
python predict.py
```