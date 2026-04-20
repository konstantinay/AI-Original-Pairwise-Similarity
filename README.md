# Audio Attribution Detection: Pairwise Similarity Modeling

A feature-based approach for detecting derivative musical relationships (covers, remixes, variations) using classical audio features and machine learning.

## Overview

This project implements pairwise similarity metrics to determine whether two audio tracks are related (e.g., cover songs, remixes, AI-generated derivatives). It uses classical audio features (MFCCs, chroma, spectral descriptors, tempo, DTW) and compares two approaches:

1. **Heuristic Baseline**: Weighted aggregation of similarity metrics with fixed weights
2. **XGBoost Approach**: Data-driven learning of feature weights using gradient boosting

## Repository Structure

```
AI-Original-Pairwise-Similarity/
├── song_attribution_score_clean/     # Refactored modular implementation
│   ├── audio_utils.py                # Audio preprocessing and segmentation
│   ├── feature_extraction.py         # Feature extraction (MFCC, chroma, etc.)
│   ├── compute_similarities.py       # Pairwise similarity metrics
│   ├── attribution_score.py          # Heuristic baseline detector
│   ├── attribution_score_xgboost.py  # XGBoost-based detector
│   ├── compare_tracks.py             # CLI for heuristic approach
│   ├── compare_tracks_xgboost.py     # CLI for XGBoost (training & validation)
│   └── data_exploration_final.ipynb  # Exploratory data analysis notebook
├── .gitignore
└── README.md
```

## Environment Setup

### Prerequisites
- Python 3.8+
- Conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/konstantinay/AI-Original-Pairwise-Similarity.git
cd AI-Original-Pairwise-Similarity
```

2. **Create conda environment**
```bash
conda create -n audio_attribution python=3.10
conda activate audio_attribution
```

3. **Install dependencies**
```bash
pip install librosa numpy scipy scikit-learn xgboost matplotlib seaborn pandas jupyter
```

**Required packages:**
- `librosa` - Audio processing and feature extraction
- `numpy` - Numerical operations
- `scipy` - Signal processing utilities
- `scikit-learn` - Machine learning utilities (ROC curves, metrics)
- `xgboost` - Gradient boosting classifier
- `matplotlib`, `seaborn` - Visualization
- `pandas` - Data manipulation
- `jupyter` - Notebook support (optional)

## Usage

### 1. Heuristic Baseline Approach

**Compare two audio tracks:**
```bash
cd song_attribution_score
python compare_tracks.py --track_a path/to/song1.wav --track_b path/to/song2.wav
```

**Run validation:**
```bash
python compare_tracks.py --validate
python compare_tracks.py --validate_sonics
```

**With extended features (HNR, phase continuity, spectral flux):**
```bash
python compare_tracks.py --validate --extended-features
```

### 2. XGBoost Approach

**Train the model (requires MIPPIA dataset):**
```bash
python compare_tracks_xgboost.py --train
```

**Validate on held-out data:**
```bash
python compare_tracks_xgboost.py --validate --model model.pkl
```

**Compare two tracks using trained model:**
```bash
python compare_tracks_xgboost.py --track_a path/to/song1.wav --track_b path/to/song2.wav --model model.pkl
```

## Dataset Requirements

The validation and training scripts expect the **MIPPIA/SMP & SONICS dataset** with the following structure, keeping only folders with valid pairs:

```
smp_dataset/
└── final_dataset_clean/
    ├── 1/
    │   ├── song1.wav
    │   └── song2.wav
    ├── 2/
    │   ├── song1.wav
    │   └── song2.wav
    └── ...
```

```
sonics_dataset/
└── fake_songs/
    ├── fake_songs/
    │   ├── song1.mp3
    │   └── song2.mp3
├── real_songs/
│   ├── song1.mp3
│   └── song2.mp3
└── ...
```
