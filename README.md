# MIST Computer Vision — CIFAR-10 Bildklassifikation

**Student:** Erik Gerst | MIST Term 09
**GitHub:** https://github.com/AwaTekoete/MIST_CV_CIFAR10

---

## Projektübersicht

Bildklassifikation auf dem CIFAR-10 Datensatz (60.000 Bilder, 10 Klassen, 32×32px).
Zwei Projektphasen: Basis-Projekt (main) und Bonus-Projekt (bonus).

---

## Ergebnisse auf einen Blick

| Modell | Val | Test | Delta |
|--------|-----|------|-------|
| Basis CNN (10k) | 74.55% | 73.32% | — |
| ResNet50 TL Phase 1 | 65.90% | — | -7.42% |
| ResNet50 TL Phase 2 Original | 66.50% | — | -6.82% |
| ResNet50 TL Phase 2 Optimiert | 74.10% | 73.85% | +0.53% |
| Bonus CNN (10k, + Cutout) | 76.65% | 76.65% | +3.33% |
| Bonus CNN (50k, + alle Aug.) | 85.22% | 84.12% | +10.80% |
| **WideResNet-28-10 (50k)** | **90.06%** | **90.22%** | **+16.90%** |

---

## Projektstruktur
MIST_CV_CIFAR10/
├── notebooks/
│   ├── 01_setup_environment.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_evaluation_final.ipynb
│   ├── 07_bonus_augmentation.ipynb
│   ├── 08_bonus_50k_pipeline.ipynb
│   └── 09_bonus_wideresnet.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md

---

## Basis-Projekt (Branch: main)

### Notebooks

| Notebook | Inhalt | Ergebnis |
|----------|--------|---------|
| 01 | Setup, Drive, GPU | ✅ |
| 02 | EDA — Klassen, Pixel, Ähnlichkeit | Gini=0.90 |
| 03 | Preprocessing, Augmentation, Pipeline | ✅ |
| 04 | Baseline CNN | **73.32% Test** |
| 05 | ResNet50 Transfer Learning | 73.85% Test |
| 06 | Finale Evaluation, SOTA-Vergleich | ✅ |

### Wichtigste Erkenntnisse

**EDA:**
- airplane hellste Klasse (Mean=142.4), frog dunkelste (106.6)
- Klassenähnlichkeit: automobile/truck 0.885, cat/dog 0.786
- Gini-Koeffizient 0.90 → balancierter Datensatz

**Modellvergleich Basis:**
- Baseline CNN (325k Parameter): 73.32%
- ResNet50 (24.2M Parameter): 73.85%
- McNemar p=0.292 → statistisch kein Unterschied
- Baseline CNN vorzuziehen: 74× kleiner, stabiler

**ResNet50 Entwicklung:**
- Phase 1 (Head only, 2.3% trainable): 65.90%
- Phase 2 (30 Layer, lr=1e-5): 66.50% (+0.60%)
- Phase 2 (60 Layer, lr=1e-4): 74.10% (+7.60%)
- Problem: Architektur-Mismatch — ResNet50 designed für 224×224

---

## Bonus-Projekt (Branch: bonus)

### Notebooks

| Notebook | Inhalt | Ergebnis |
|----------|--------|---------|
| 07 | Cutout Augmentation (10k) | **76.65% Val** |
| 08 | 50k + Mixup + Label Smoothing | **84.12% Test** |
| 09 | WideResNet-28-10 | **90.22% Test** |

### Augmentation-Techniken

| Technik | Beschreibung | Gain |
|---------|-------------|------|
| Cutout | 8×8 Maske, 50% Wahrsch. | +3.33% |
| Mixup | Samples mischen (α=0.2) | Teil von +10.80% |
| Label Smoothing | Labels weicher (0.1) | Teil von +10.80% |

### Wichtiger Bug — gefunden und behoben
Bug: Normalisierung VOR Augmentation
→ RandomBrightness → images max=48.37 statt 1.0
→ Modell lernte nichts (10% = Random Guess)
Fix: Augmentieren → Normalisieren → Clippen

### WideResNet-28-10
Architektur: depth=28, width=10
Parameter  : 36.5M
Filter     : [16, 160, 320, 640]
Optimizer  : SGD + Nesterov + Cosine Annealing
Training   : 30 Epochen Colab + 14 Epochen Kaggle

**Warum WideResNet gewinnt:**
ResNet50 bei 32×32  : Output (1,1,2048) — kein Spatial Info
WideResNet bei 32×32: Output (4,4,640)  — Spatial erhalten
→ Architektur-Match mit Datensatz entscheidend
→ WideResNet designed für 32×32

### Statistische Signifikanz (McNemar Test)

| Vergleich | Chi² | p-value | Signifikant |
|-----------|------|---------|-------------|
| WideResNet vs Basis CNN | 1271.27 | ~0.000 | ✅ Ja |
| WideResNet vs Bonus CNN 50k | 317.54 | ~0.000 | ✅ Ja |

### Per-Class F1 WideResNet

| Klasse | F1 |
|--------|-----|
| automobile | 99% |
| ship | 98% |
| horse | 93% |
| frog / airplane / truck | 92% |
| deer | 88% |
| bird | 89% |
| cat | 82% |
| dog | 77% |

---

## SOTA-Vergleich

| Modell | Accuracy |
|--------|---------|
| Random Guess | 10.00% |
| **Basis CNN (unser)** | **73.32%** |
| **WideResNet (unser)** | **90.22%** |
| Mensch | ~94.00% |
| SOTA (ViT-H/14) | ~99.50% |

---

## Technische Details
Plattform  : Google Colab (T4 GPU) + Kaggle (2× T4)
Framework  : TensorFlow 2.20 / Keras
Datensatz  : CIFAR-10 (60k Bilder, 10 Klassen, 32×32px)
Normalisierung Basis    : preprocess_input() ResNet50
Normalisierung Bonus    : /255.0 + clip [0,1]

---

## Setup

```bash
git clone https://github.com/AwaTekoete/MIST_CV_CIFAR10
pip install -r requirements.txt
```

Notebooks in Google Colab ausführen — Drive Mount in Zelle 01.

