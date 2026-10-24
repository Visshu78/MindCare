"""
Master Experiment Runner for Reddit Mental Health Disorder Detection.
Trains DistilBERT Baseline, DistilBERT + BiGRU, and Proposed DistilBERT + BiGRU + Attention
on the curated Reddit Mental Health benchmark dataset (Mental Distress vs Control).
Evaluates all models on identical held-out test split, computing Confusion Matrices,
Classification Report Matrices, and saves checkpoints & summary metrics.
"""

import os
import time
import json
import torch
from reddit_data_loader import prepare_reddit_data, REDDIT_LABEL_NAMES
from models import (
    DistilBERTBaseline,
    DistilBERT_BiGRU,
    DistilBERT_BiGRU_Attention
)
from trainer import ModelTrainer

def run_reddit_benchmarks(epochs=3, batch_size=32, lr=2e-5):
    print("=" * 70)
    print("MINDCARE RESEARCH BENCHMARK: REDDIT MENTAL HEALTH EXPERIMENTS")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    if device.type == 'cpu':
        threads = torch.get_num_threads()
        print(f"PyTorch parallel CPU threads: {threads}")

    # Prepare Reddit datasets
    train_loader, val_loader, test_loader, tokenizer, df_train, df_val, df_test = prepare_reddit_data(
        data_dir="model_experiments/data",
        train_sample=3000,
        val_sample=600,
        test_sample=600,
        max_len=128,
        batch_size=batch_size
    )

    models_to_train = [
        {
            "id": "reddit_baseline1",
            "name": "DistilBERT Baseline (Reddit)",
            "model_cls": DistilBERTBaseline,
            "save_name": "reddit_baseline1_best"
        },
        {
            "id": "reddit_baseline2",
            "name": "DistilBERT + BiGRU (Reddit)",
            "model_cls": DistilBERT_BiGRU,
            "save_name": "reddit_baseline2_best"
        },
        {
            "id": "reddit_proposed",
            "name": "DistilBERT + BiGRU + Attention (Proposed MindCare on Reddit)",
            "model_cls": DistilBERT_BiGRU_Attention,
            "save_name": "reddit_proposed_best"
        }
    ]

    all_results = {}
    results_dir = "model_experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "reddit_summary_metrics.json")

    for cfg in models_to_train:
        m_id = cfg["id"]
        m_name = cfg["name"]
        print("\n" + "#" * 60)
        print(f"STARTING EXPERIMENT: {m_name}")
        print("#" * 60)

        # Initialize model with 2 output classes (Control vs Mental Distress)
        model = cfg["model_cls"](num_classes=2)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

        trainer = ModelTrainer(
            model=model,
            device=device,
            save_dir="model_experiments/saved_models",
            model_name=cfg["save_name"]
        )

        start_time = time.time()
        history, training_time, best_checkpoint_path = trainer.fit(
            train_loader, val_loader, epochs=epochs, lr_backbone=lr, lr_head=1e-3
        )

        # Evaluate best model checkpoint on held-out test split
        test_metrics = trainer.evaluate(test_loader)

        test_loss = test_metrics['loss']
        test_acc = test_metrics['accuracy']
        test_macro_f1 = test_metrics['f1_macro']
        test_weighted_f1 = test_metrics['f1_weighted']
        test_prec = test_metrics['precision_macro']
        test_rec = test_metrics['recall_macro']
        test_cm = test_metrics['confusion_matrix']
        latency = test_metrics['latency_ms']

        print("\n" + "-" * 50)
        print(f"TEST RESULTS FOR {m_name}:")
        print(f"  Test Loss:        {test_loss:.4f}")
        print(f"  Test Accuracy:    {test_acc * 100:.2f}%")
        print(f"  Test Macro F1:    {test_macro_f1 * 100:.2f}%")
        print(f"  Test Weighted F1: {test_weighted_f1 * 100:.2f}%")
        print(f"  Macro Precision:  {test_prec * 100:.2f}%")
        print(f"  Macro Recall:     {test_rec * 100:.2f}%")
        print(f"  Avg Latency/Sample: {latency:.2f} ms")
        print(f"  Confusion Matrix:\n{test_cm}")
        print("-" * 50)

        all_results[m_id] = {
            "display_name": m_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "training_time_seconds": training_time,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_macro_f1),
            "test_weighted_f1": float(test_weighted_f1),
            "test_precision_macro": float(test_prec),
            "test_recall_macro": float(test_rec),
            "test_latency_ms": float(latency),
            "history": history,
            "test_confusion_matrix": test_cm
        }

        # Save incremental metrics
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved metrics to {summary_path}")

    print("\n" + "=" * 70)
    print("ALL REDDIT EXPERIMENTS COMPLETED!")
    print("=" * 70)
    return all_results

if __name__ == "__main__":
    run_reddit_benchmarks(epochs=3, batch_size=32, lr=2e-5)
