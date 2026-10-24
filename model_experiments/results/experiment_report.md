# MindCare Deep Learning Experiments: Comparative Ablation Report

Evaluation based on the architectures and ablation strategy specified in **`Deep Learning Minoe Project.pptx`**.

## 1. Quantitative Performance Comparison (Held-Out Test Set)

| Model Architecture | Test Accuracy | Macro F1 | Weighted F1 | Precision (Macro) | Recall (Macro) | Latency (ms) | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT Baseline** | 89.40% | 84.12% | 89.24% | 86.29% | 82.39% | 17.02 ms | 66,367,494 |
| **DistilBERT + BiGRU** | 89.80% | 84.37% | 89.61% | 87.83% | 83.13% | 18.35 ms | 67,054,086 |
| **DistilBERT + BiGRU + Attention (Proposed)** | 89.40% | 85.62% | 89.44% | 87.25% | 85.28% | 18.45 ms | 67,119,238 |
| **DistilBERT + CNN + BiGRU + Attention (Additional)** | 89.60% | 84.73% | 89.35% | 88.62% | 82.51% | 18.41 ms | 67,316,102 |

## 2. Key Findings & Ablation Insights

- **Baseline 1 vs Baseline 2**: Adding Bidirectional GRU captures forward and backward sequential dependencies across the contextual token embeddings.
- **Proposed Model (DistilBERT + BiGRU + Attention)**: The Additive Attention layer dynamically concentrates representation power on high-emotion and distress-bearing words (e.g. *exhausted, anxious, peaceful, terrified*), delivering superior classification robustness.
- **Additional Experiment (DistilBERT + CNN + BiGRU + Attention)**: Evaluates whether 1D convolutions add local phrase-level extraction before sequence modeling.

## 3. Visual Artifacts Generated

- `training_curves.png`: Epoch-by-epoch loss convergence and validation F1 progress.
- `ablation_comparison.png`: Side-by-side performance and inference latency metrics.
- `confusion_matrices.png`: Class-level confusion distribution.
- `attention_visualization.png`: Token-level attention weights demonstrating model explainability.
