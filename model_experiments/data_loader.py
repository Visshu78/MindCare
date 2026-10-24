"""
Data loading and preprocessing module for MindCare Deep Learning Experiments.
Supports loading and splitting the benchmark emotion & mental well-being dataset,
tokenizing with DistilBERT, and preparing PyTorch DataLoaders.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split

LABEL_NAMES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
LABEL_MAP = {i: name for i, name in enumerate(LABEL_NAMES)}
SENTIMENT_MAP = {
    'sadness': 'NEGATIVE',
    'anger': 'NEGATIVE',
    'fear': 'NEGATIVE',
    'joy': 'POSITIVE',
    'love': 'POSITIVE',
    'surprise': 'NEUTRAL'
}

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }

def prepare_data(data_dir="model_experiments/data", train_sample=4000, val_sample=1000, test_sample=1000, max_len=64, batch_size=32):
    """
    Loads dataset, splits into deterministic train/val/test splits,
    saves them to data_dir for reproducibility, and returns DataLoaders.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"Loading existing datasets from {data_dir}...")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        if train_sample and len(df_train) > train_sample:
            df_train, _ = train_test_split(df_train, train_size=train_sample, stratify=df_train['label'], random_state=42)
        if val_sample and len(df_val) > val_sample:
            df_val, _ = train_test_split(df_val, train_size=val_sample, stratify=df_val['label'], random_state=42)
        if test_sample and len(df_test) > test_sample:
            df_test, _ = train_test_split(df_test, train_size=test_sample, stratify=df_test['label'], random_state=42)
    else:
        print("Curating and preparing dataset from dair-ai/emotion...")
        raw_ds = load_dataset("dair-ai/emotion")
        df_train_full = pd.DataFrame(raw_ds['train'])
        df_val_full = pd.DataFrame(raw_ds['validation'])
        df_test_full = pd.DataFrame(raw_ds['test'])

        # Stratified sampling for consistent, lightweight training
        if train_sample and train_sample < len(df_train_full):
            df_train, _ = train_test_split(
                df_train_full, train_size=train_sample, stratify=df_train_full['label'], random_state=42
            )
        else:
            df_train = df_train_full

        if val_sample and val_sample < len(df_val_full):
            df_val, _ = train_test_split(
                df_val_full, train_size=val_sample, stratify=df_val_full['label'], random_state=42
            )
        else:
            df_val = df_val_full

        if test_sample and test_sample < len(df_test_full):
            df_test, _ = train_test_split(
                df_test_full, train_size=test_sample, stratify=df_test_full['label'], random_state=42
            )
        else:
            df_test = df_test_full

        # Add label names
        df_train['label_name'] = df_train['label'].map(LABEL_MAP)
        df_val['label_name'] = df_val['label'].map(LABEL_MAP)
        df_test['label_name'] = df_test['label'].map(LABEL_MAP)

        # Save to disk
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)
        print(f"Saved dataset splits to {data_dir}: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_ds = TextDataset(df_train['text'], df_train['label'], tokenizer, max_len=max_len)
    val_ds = TextDataset(df_val['text'], df_val['label'], tokenizer, max_len=max_len)
    test_ds = TextDataset(df_test['text'], df_test['label'], tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer, len(LABEL_NAMES)

if __name__ == "__main__":
    t_loader, v_loader, test_loader, tok, n_classes = prepare_data()
    print(f"Data prepared successfully! Classes: {n_classes}, Train batches: {len(t_loader)}")
