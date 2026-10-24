"""
Experiment Orchestrator for MindCare Deep Learning Models.
Trains all 4 models sequentially on identical splits, performs evaluation on the test set,
measures computational latency, and exports summary metrics.
"""

import os
import json
import time
import torch
try:
    from .data_loader import prepare_data, LABEL_NAMES
    from .models import get_model
    from .trainer import ModelTrainer
except ImportError:
    from data_loader import prepare_data, LABEL_NAMES
    from models import get_model
    from trainer import ModelTrainer

def run_all_experiments(epochs=3, batch_size=32, max_len=48, train_samples=2500, val_samples=500, test_samples=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"MindCare Deep Learning Comparative Experiments")
    print(f"Running on Device: {device} | CPU threads: {torch.get_num_threads()}")
    print("=" * 70)

    # 1. Prepare Data
    train_loader, val_loader, test_loader, tokenizer, num_classes = prepare_data(
        data_dir="model_experiments/data",
        train_sample=train_samples,
        val_sample=val_samples,
        test_sample=test_samples,
        max_len=max_len,
        batch_size=batch_size
    )

    models_to_run = [
        ('baseline1', 'DistilBERT Baseline'),
        ('baseline2', 'DistilBERT + BiGRU'),
        ('proposed', 'DistilBERT + BiGRU + Attention (Proposed)'),
        ('additional', 'DistilBERT + CNN + BiGRU + Attention (Additional)')
    ]

    all_results = {}
    saved_paths = {}

    for model_key, display_name in models_to_run:
        print("\n" + "=" * 70)
        print(f"STARTING EXPERIMENT: {display_name} ({model_key})")
        print("=" * 70)

        # Instantiate model
        model = get_model(model_key, num_classes=num_classes)
        trainer = ModelTrainer(model, device, save_dir="model_experiments/saved_models", model_name=model_key)

        # Fit model
        history, train_time, best_ckpt = trainer.fit(
            train_loader, val_loader, epochs=epochs, lr_backbone=2e-5, lr_head=1e-3
        )
        saved_paths[model_key] = best_ckpt

        # Evaluate on Held-Out Test Set
        print(f"\nEvaluating {display_name} on unseen held-out Test Set...")
        test_metrics = trainer.evaluate(test_loader)
        print(f"--> Test Accuracy: {test_metrics['accuracy']:.4f} | "
              f"Macro F1: {test_metrics['f1_macro']:.4f} | "
              f"Weighted F1: {test_metrics['f1_weighted']:.4f} | "
              f"Latency: {test_metrics['latency_ms']:.2f} ms/sample")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        all_results[model_key] = {
            'display_name': display_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_time_seconds': float(train_time),
            'test_accuracy': test_metrics['accuracy'],
            'test_macro_f1': test_metrics['f1_macro'],
            'test_weighted_f1': test_metrics['f1_weighted'],
            'test_precision_macro': test_metrics['precision_macro'],
            'test_recall_macro': test_metrics['recall_macro'],
            'test_latency_ms': test_metrics['latency_ms'],
            'history': history,
            'test_confusion_matrix': test_metrics['confusion_matrix']
        }

    # Save summary results
    os.makedirs("model_experiments/results", exist_ok=True)
    summary_path = "model_experiments/results/summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSummary metrics successfully saved to {summary_path}")

    # Display Comparative Table
    print("\n" + "=" * 90)
    print("FINAL COMPARATIVE ABLATION SUMMARY (Test Set Performance)")
    print("=" * 90)
    header = f"{'Model Architecture':<42} | {'Test Acc':<9} | {'Macro F1':<9} | {'Weight F1':<10} | {'Latency':<10} | {'Params'}"
    print(header)
    print("-" * 90)
    for k, v in all_results.items():
        row = f"{v['display_name']:<42} | {v['test_accuracy']*100:6.2f}%   | {v['test_macro_f1']*100:6.2f}%   | {v['test_weighted_f1']*100:6.2f}%    | {v['test_latency_ms']:5.2f} ms   | {v['total_parameters']:,}"
        print(row)
    print("=" * 90)

    return all_results

if __name__ == "__main__":
    run_all_experiments()
