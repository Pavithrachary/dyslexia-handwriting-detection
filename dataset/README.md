# Dataset

## Source
**Dyslexia Handwriting Dataset**  
Available on Kaggle: https://www.kaggle.com/datasets/sumitaich/dyslexia-datasets

## Structure
After downloading, the dataset should be organized as follows:

```
dataset/
├── train/
│   ├── A/
│   ├── B/
│   └── ... (A–Z)
├── valid/
│   ├── A/
│   └── ... (A–Z)
└── test/
    ├── A/
    └── ... (A–Z)
```

## How to Download

### Option 1: Kaggle CLI (Recommended)
```bash
bash scripts/download_dataset.sh
```

### Option 2: Manual Download
1. Go to https://www.kaggle.com/datasets/sumitaich/dyslexia-datasets
2. Download and unzip into the `dataset/` folder

## Notes
- Images are RGB handwriting samples of individual letters (A–Z)
- Ensure the folder names match exactly (case-sensitive on Linux)
