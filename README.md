# Bangla Hate Speech Detection – Knowledge Distillation  
**Fast, Accurate, and Deployable Bengali Hate Speech Classifier using Distilled sahajBERT**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnnNaserNabil/Bangla-HateSpeech-Distillation/blob/main/notebooks/run_distillation.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/yourname/bangla-hate-distilled)

State-of-the-art **Bangla hate speech detection** using **knowledge distillation** from a large multilingual teacher (mBERT) to a lightweight student (`neuropark/sahajBERT`).  
Achieves **near full-sized BERT performance** with **~3× faster inference** and **smaller model size**.

---

### Features
- 5-fold stratified cross-validation
- Full training + validation metrics (Accuracy, Precision, Recall, F1, Macro F1, ROC-AUC for both classes)
- Early stopping + best model checkpointing
- Automatic saving of **Hugging Face-ready model** (`config.json`, `pytorch_model.bin`, tokenizer)
- MLflow tracking
- One-click deployment to Hugging Face Spaces

---

### Results (Example from 5-fold CV)
| Model                        | Val Macro F1 | Val F1 (Hate) | ROC-AUC | Model Size | Inference Speed |
|-----------------------------|--------------|---------------|---------|------------|-----------------|
| mBERT (Teacher)             | 0.872        | 0.859         | 0.941   | ~700 MB    | Slow            |
| **sahajBERT (Distilled)**   | **0.869**    | **0.853**     | **0.938** | **~270 MB** | **3× faster**   |

---

### Quick Start

#### Option 1: Google Colab (Recommended – Zero Setup)
Open in Colab → Run All:  
https://colab.research.google.com/github/AnnNaserNabil/Bangla-HateSpeech-Distillation/blob/main/notebooks/run_distillation.ipynb

Or manually:

```bash
# 1. Clone repo
!git clone https://github.com/AnnNaserNabil/Bangla-HateSpeech-Distillation
%cd Bangla-HateSpeech-Distillation

# 2. Install dependencies
!pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow

# 3. Upload your dataset (HateSpeech.csv with columns: Comments, HateSpeech)
#    → Drag and drop into Colab files panel
```

```bash
!git clone https://github.com/AnnNaserNabil/HateSpeechKD
%cd HateSpeechKD
!pip install -q torch transformers scikit-learn pandas numpy tqdm mlflow
```


```bash
python main_distill.py \
  --author_name "YourName" \
  --dataset_path "HateSpeech.csv" \
  --model_path "neuropark/sahajBERT" \
  --teacher "google-bert/bert-base-multilingual-cased" \
  --distill \
  --alpha 0.7 \
  --temperature 4.0 \
  --batch 32 \
  --lr 3e-5 \
  --epochs 12 \
  --num_folds 5 \
  --seed 42 \
  --dropout 0.1 \
  --early_stopping_patience 5
```


## Deploy to Hugging Face Spaces (2 minutes)After training finishes:

# Login once
huggingface-cli login

# Upload your best model
huggingface-cli upload yourname/bangla-hate-distilled ./best_distilled_model_YourName_sahajBERT .


