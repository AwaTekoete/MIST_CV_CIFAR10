# MIST Computer Vision — CIFAR-10 Classification

## Projektübersicht
Bildklassifikation auf dem CIFAR-10 Datensatz im Rahmen des
MIST (Master School of IT) Computer Vision Kurses.

Zwei Ansätze werden verglichen:
- **Baseline CNN**: Einfaches CNN von Grund auf trainiert
- **ResNet50 Transfer Learning**: Vortrainiertes Modell + Custom Head

---

## Ergebnisse

| Modell | Accuracy | Macro F1 | Parameter |
|--------|---------|---------|-----------|
| Random Guess | 10.0% | 10.0% | 0 |
| Baseline CNN | **73.32%** | **73.15%** | 325k |
| ResNet50 TL | 73.85% | 73.69% | 24M |

**Fazit:** Kein statistisch signifikanter Unterschied (McNemar p=0.292).
Baseline CNN ist unter diesen Rahmenbedingungen vorzuziehen
(74× kleiner, stabiler gegen Overfitting).

---

## Projektstruktur
MIST_CV_CIFAR10/
├── notebooks/
│   ├── 01_setup_environment.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_transfer_learning.ipynb
│   └── 06_evaluation_final.ipynb
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
---

## Datensatz

**CIFAR-10** — 60.000 Bilder, 10 Klassen, 32×32 Pixel RGB

| Split | Samples | Verwendung |
|-------|---------|-----------|
| Training | 8.000 | Modell Training |
| Validation | 2.000 | Hyperparameter |
| Test | 10.000 | Finale Evaluation |

---

## Methodik

### Preprocessing
- Normalisierung: `preprocess_input()` (ImageNet-Standard)
- Augmentation: Horizontal Flip, Translation, Brightness, Contrast
- Pipeline: `tf.data` mit Cache, Shuffle, Prefetch

### Baseline CNN
- 3 Convolutional Blocks (32→64→128 Filter)
- BatchNormalization + Dropout + GlobalAveragePooling2D
- 325.418 Parameter — Training: 30 Epochen, Adam lr=0.001

### ResNet50 Transfer Learning
- Phase 1: Head Training (23 Epochen bis Konvergenz)
- Phase 2: Fine-Tuning (60 Layer aufgetaut, lr=1e-4)
- Custom Head: Dense(256) → Dense(128) → Dense(10)

---

## Wichtigste Erkenntnisse

1. **EDA ist fundamental** — alle Vorhersagen bestätigt
2. **Größeres Modell ≠ besseres Modell** — p=0.292 (nicht signifikant)
3. **Datenmenge dominiert** — 50k → +11% Accuracy
4. **Architektur-Mismatch** — ResNet50 für 224×224, nicht 32×32

---

## Abstand zu State-of-the-Art

| Modell | Accuracy | Delta |
|--------|---------|-------|
| Unser bestes Modell | 73.9% | — |
| ResNet50 (50k) | 85.0% | +11.1% |
| Mensch | 94.0% | +20.1% |
| ViT-H/14 (SOTA 2024) | 99.5% | +25.6% |

---

## Technologie
- Python 3.12 | TensorFlow 2.20.0 | scikit-learn 1.6.1
- NumPy 2.0.2 | OpenCV 4.13.0 | Google Colab (T4 GPU)

---

## Autor
**AwaTekoete** — MIST Term 09, 2026

*Lernfortschritt dokumentiert gesamten Entwicklungsprozess.*
