# 🧠 Dyslexia Handwriting Detection using ResNet50

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-ResNet50-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A deep learning project that classifies handwritten characters (A–Z) from individuals with dyslexia using transfer learning with **ResNet50** pre-trained on ImageNet.

---

## 📌 Project Overview

Dyslexia affects handwriting in distinct ways — letter reversals, inconsistencies, and irregular spacing are common. This project uses a Convolutional Neural Network (ResNet50) fine-tuned on a labeled handwriting dataset to detect and classify such patterns across 26 alphabet characters.

---

## 📁 Project Structure

```
dyslexia-handwriting-detection/
│
├── notebooks/
│   └── dyslexia_detection.ipynb    # Main Jupyter notebook (full pipeline)
│
├── src/
│   ├── model.py                    # Model architecture definition
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation & metrics
│   └── predict.py                  # Inference / single image prediction
│
├── scripts/
│   └── download_dataset.sh         # Script to download dataset from Kaggle
│
├── dataset/
│   └── README.md                   # Dataset description and structure
│
├── docs/
│   └── project_report.md           # Detailed project report
│
├── results/
│   └── .gitkeep                    # Placeholder for model outputs & plots
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🗂️ Dataset

- **Source:** [Dyslexia Handwriting Dataset on Kaggle](https://www.kaggle.com/datasets/sumitaich/dyslexia-datasets)
- **Structure:** train / valid / test splits with folders for each letter (A–Z)
- **Classes:** 26 (one per alphabet character)

> ⚠️ Dataset is **not included** in this repository due to size constraints. See [`scripts/download_dataset.sh`](scripts/download_dataset.sh) to download it automatically.

---

## 🏗️ Model Architecture

| Component             | Details                          |
|-----------------------|----------------------------------|
| Base Model            | ResNet50 (ImageNet weights)      |
| Input Shape           | (224, 224, 3)                    |
| Custom Head           | GlobalAveragePooling2D → Dense(1024, ReLU) → Dense(26, Softmax) |
| Optimizer             | Adam                             |
| Loss Function         | Categorical Crossentropy         |
| Output Classes        | 26 (A–Z)                         |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/dyslexia-handwriting-detection.git
cd dyslexia-handwriting-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
```bash
bash scripts/download_dataset.sh
```
> You need a Kaggle API key (`kaggle.json`) placed at `~/.kaggle/kaggle.json`.

### 4. Run the Notebook
```bash
jupyter notebook notebooks/dyslexia_detection.ipynb
```

Or run training directly:
```bash
python src/train.py
```

---

## 🔁 Training Pipeline

1. **Data Augmentation** — rotation, zoom, flip, shift on training images
2. **Transfer Learning** — ResNet50 frozen base + custom classification head
3. **Callbacks** — ModelCheckpoint (saves best model) + EarlyStopping (patience=5)
4. **Epochs** — up to 100, early stopping based on `val_accuracy`

---

## 📊 Results

Training and validation accuracy/loss curves are generated after training.

Per-class metrics (Precision, Recall, F1-Score) are visualized using horizontal bar charts for all 26 character classes.

> 📁 Outputs saved to the `results/` directory.

---

## 🧰 Tech Stack

- Python 3.12
- TensorFlow / Keras
- ResNet50 (transfer learning)
- scikit-learn (metrics)
- matplotlib (visualization)
- NumPy

---

## 🚀 Future Work

- [ ] Fine-tune ResNet50 layers for better accuracy
- [ ] Experiment with EfficientNet / Vision Transformers
- [ ] Deploy as a web app using Flask or Streamlit
- [ ] Expand to detect full words, not just characters

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🙌 Acknowledgements

- Dataset by [Sumit Aich on Kaggle](https://www.kaggle.com/datasets/sumitaich/dyslexia-datasets)
- ResNet50 by He et al. (2015) — *Deep Residual Learning for Image Recognition*
