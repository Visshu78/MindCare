# 🧠 MindCare: Text-Based Mental Well-Being Analysis Using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DistilBERT-yellow)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)

> **MindCare** is an intelligent mental well-being journaling and analytics platform powered by a custom hybrid deep learning architecture (**DistilBERT + BiGRU + Additive Attention**). It detects emotional patterns, provides real-time empathetic feedback, tracks mood trends, highlights key emotional words via explainable AI, and includes an emergency safety triage system.

For the exhaustive academic and experimental report, please see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## 🚀 Key Features

- **Custom Hybrid Deep Learning Architecture:** Combines transformer contextual representations (`DistilBERT`), sequential bidirectional modeling (`BiGRU`), and token salience pooling (`Additive Attention`).
- **Dual-Benchmark Evaluated:** Benchmarked on both the **6-Class Emotion Dataset** (Sadness, Joy, Love, Anger, Fear, Surprise) and the **Reddit Mental Health Dataset** (Mental Distress vs Control).
- **Explainable AI (XAI):** Additive attention layer extracts and visualizes the exact words driving the emotional prediction (e.g. *hopeless*, *terrified*, *peaceful*).
- **Real-Time Interactive Journaling:** Dynamic sentiment analysis, mood sliders (1–10), and adaptive empathetic feedback.
- **Visual Analytics:** Interactive Plotly charts for mood tracking over time, emotional distribution breakdowns, and journaling streak counts.
- **Safety Triage Interceptor:** Automatic crisis detection for self-harm/suicide terms with immediate helpline intervention.
- **Export to PDF:** Download historical journal entries with ReportLab for personal review or sharing with healthcare providers.
- **Research Evaluation Dashboard:** Embedded ablation study matrix, confusion matrix heatmaps, classification report metrics, and live sentence tester.

---

## 🔬 Deep Learning Ablation & Comparison

### 1. Multi-Class Emotion Benchmark (6 Classes)
*Evaluated on 500 held-out test samples (`dair-ai/emotion`):*

| Architecture | Test Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Baseline** | 89.40% | 84.12% | 89.24% | 86.29% | 82.39% | 17.02 ms | 66.37M |
| **DistilBERT + BiGRU** | **89.80%** | 84.37% | **89.61%** | 87.83% | 83.13% | 18.35 ms | 67.05M |
| **DistilBERT + BiGRU + Attention (Proposed)** | 89.40% | **85.62%** | 89.44% | 87.25% | **85.28%** | 18.45 ms | 67.12M |
| **DistilBERT + CNN + BiGRU + Attention** | 89.60% | 84.73% | 89.35% | **88.62%** | 82.51% | 18.41 ms | 67.32M |

### 2. Reddit Mental Health Benchmark (Binary: Distress vs Control)
*Evaluated on 600 held-out test samples (`kamruzzaman-asif/reddit-mental-health-classification`):*

| Architecture | Test Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT Baseline** | 95.00% | 95.00% | 95.00% | 95.00% | 95.00% | 38.99 ms | 66.36M |
| **DistilBERT + BiGRU** | **95.50%** | **95.50%** | **95.50%** | **95.54%** | **95.50%** | 36.65 ms | 67.05M |
| **DistilBERT + BiGRU + Attention (Proposed)** | 94.00% | 94.00% | 94.00% | 94.05% | 94.00% | 37.12 ms | 67.12M |

---

## 🛠️ Installation & Setup

1. **Clone the repository and enter the directory:**
   ```bash
   cd MindCare_v2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run main.py
   ```

---

## 📂 Project Structure

```
MindCare/
├── model_experiments/                # Research Core & Experiments
│   ├── data/                         # Train/Val/Test CSV splits
│   ├── results/                      # summary_metrics.json, plots, ablation reports
│   ├── saved_models/                 # PyTorch model checkpoints (.pt)
│   ├── models.py                     # 4 neural model implementations
│   ├── data_loader.py                # Emotion dataset pipeline
│   ├── reddit_data_loader.py         # Reddit dataset pipeline
│   ├── trainer.py                    # Training & validation loop with metrics
│   ├── predictor.py                  # Standalone inference & attention extractor
│   ├── run_experiments.py            # Automated multi-class experiment runner
│   └── run_reddit_experiments.py     # Automated Reddit experiment runner
│
└── MindCare_v2/                      # Web Application Core
    ├── main.py                       # Streamlit entrypoint
    ├── auth.py                       # User authentication & session state
    ├── database.py                   # SQLite schema & queries
    ├── utils.py                      # Model loading, caching, PDF generation
    ├── journal.db                    # User data & journal entries
    ├── PROJECT_DOCUMENTATION.md      # Detailed academic research report
    └── Components/
        ├── journal_tab.py            # Daily journaling & AI inference
        ├── history_tab.py            # Historical entries & PDF download
        ├── analytics_tab.py          # Interactive Plotly mood analytics
        ├── evaluation_tab.py         # Research metrics & live attention inspector
        └── settings_tab.py           # Account settings & security
```

---

## 👥 Contributors & Academic Reference
- **Author:** Vishal Dhawal (Roll No: 2023BCS0153)
- **Project:** MindCare — Text-Based Mental Well-Being Analysis Using Deep Learning
- **Institution:** Academic Project / Mini Project

### Key Literature & References
1. **Hasan et al. (2022)** — *Advancing Mental Disorder Detection: A Comparative Evaluation of Transformer and LSTM Architectures on Social Media* (IEEE BIBM).
2. **Saravia et al. (2018)** — *CARER: Contextualized Affect Representations for Emotion Recognition* (EMNLP).
3. **Aina et al. (2024)** — *A Hybrid Learning-Architecture for Mental Disorder Detection Using Emotion Recognition* (IEEE Access).
4. **Sanh et al. (2019)** — *DistilBERT: A distilled version of BERT: smaller, faster, cheaper and lighter* (NeurIPS).
5. **Bahdanau et al. (2015)** — *Neural Machine Translation by Jointly Learning to Align and Translate* (Additive Attention, ICLR).
6. **Cho et al. (2014)** — *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation* (GRU, EMNLP).
