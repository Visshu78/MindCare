"""
Trainer and Evaluator module for MindCare Model Experiments.
Computes comprehensive metrics: Accuracy, Macro F1, Weighted F1, Precision, Recall,
Inference Latency, and saves best model checkpoints.
"""

import os
import time
import json
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

class ModelTrainer:
    def __init__(self, model, device, save_dir="model_experiments/saved_models", model_name="model"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, optimizer, scheduler=None):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        _, _, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

        return avg_loss, acc, macro_f1

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        start_time = time.time()
        sample_count = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits, _ = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                sample_count += len(labels)

        eval_time = time.time() - start_time
        latency_ms_per_sample = (eval_time / max(sample_count, 1)) * 1000.0

        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds).tolist()

        metrics = {
            'loss': float(avg_loss),
            'accuracy': float(acc),
            'precision_macro': float(prec_macro),
            'recall_macro': float(rec_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(prec_weighted),
            'recall_weighted': float(rec_weighted),
            'f1_weighted': float(f1_weighted),
            'latency_ms': float(latency_ms_per_sample),
            'confusion_matrix': cm
        }
        return metrics

    def fit(self, train_loader, val_loader, epochs=3, lr_backbone=2e-5, lr_head=1e-3):
        # Differential learning rates: lower for pretrained transformer backbone, higher for custom head
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'distilbert' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ], weight_decay=0.01)

        best_val_f1 = -1.0
        best_save_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        print(f"\nTraining [{self.model.model_name}] for {epochs} epochs...")
        start_train_time = time.time()

        for epoch in range(1, epochs + 1):
            t_loss, t_acc, t_f1 = self.train_epoch(train_loader, optimizer)
            val_metrics = self.evaluate(val_loader)

            history['train_loss'].append(t_loss)
            history['train_acc'].append(t_acc)
            history['train_f1'].append(t_f1)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_macro'])

            print(f"  Epoch {epoch}/{epochs} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} F1: {t_f1:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f}")

            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                torch.save(self.model.state_dict(), best_save_path)
                print(f"    --> Saved new best checkpoint to {best_save_path} (Val F1: {best_val_f1:.4f})")

        total_training_time = time.time() - start_train_time
        print(f"Completed training for [{self.model.model_name}] in {total_training_time:.1f}s. Best Val F1: {best_val_f1:.4f}")

        # Load best checkpoint
        if os.path.exists(best_save_path):
            self.model.load_state_dict(torch.load(best_save_path, map_location=self.device))

        return history, total_training_time, best_save_path
