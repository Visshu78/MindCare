"""
Reddit Mental Health Data Loader and Preprocessor.
Curation and splitting for Mental Health Disorder Detection
(Mental Distress vs Control) matching Hasan et al. (Advancing Mental Disorder Detection).
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from datasets import load_dataset

REDDIT_LABEL_NAMES = ["control", "mental_distress"]
REDDIT_LABEL_MAP = {0: "control", 1: "mental_distress"}

MENTAL_HEALTH_CATEGORIES = {'depression', 'healthanxiety', 'bipolarreddit', 'bpd', 'addiction', 'alcoholism'}
CONTROL_CATEGORIES = {'jokes', 'personalfinance', 'teaching', 'meditation', 'relationships'}

def clean_reddit_text(text):
    """Cleans common Reddit formatting artifacts."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
    text = re.sub(r'&gt;.*', '', text)          # Quotes
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
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

def prepare_reddit_data(
    data_dir="model_experiments/data",
    train_sample=3000,
    val_sample=600,
    test_sample=600,
    max_len=128,
    batch_size=32
):
    """
    Prepares balanced binary Reddit mental health dataset.
    Train: 3000 (1500 distress, 1500 control)
    Val: 600 (300 distress, 300 control)
    Test: 600 (300 distress, 300 control)
    """
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "reddit_train.csv")
    val_path = os.path.join(data_dir, "reddit_val.csv")
    test_path = os.path.join(data_dir, "reddit_test.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"Loading existing Reddit dataset from {data_dir}...")
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
    else:
        print("Curating balanced Reddit dataset from kamruzzaman-asif/reddit-mental-health-classification...")
        raw_ds = load_dataset('kamruzzaman-asif/reddit-mental-health-classification', split='train[:80000]')
        
        records = []
        for item in raw_ds:
            sub = item['label']
            raw_text = item['text']
            if not raw_text or len(raw_text.strip()) < 30:
                continue

            cleaned = clean_reddit_text(raw_text)
            if len(cleaned.split()) < 5:
                continue

            if sub in MENTAL_HEALTH_CATEGORIES:
                records.append({'text': cleaned, 'raw_category': sub, 'label': 1, 'label_name': 'mental_distress'})
            elif sub in CONTROL_CATEGORIES:
                records.append({'text': cleaned, 'raw_category': sub, 'label': 0, 'label_name': 'control'})

        df_all = pd.DataFrame(records)
        print(f"Collected total valid records: {len(df_all)}")
        print("Class distribution:\n", df_all['label_name'].value_counts())

        # Sample balanced pool
        target_per_class = (train_sample + val_sample + test_sample) // 2
        df_distress = df_all[df_all['label'] == 1].sample(n=min(target_per_class, sum(df_all['label'] == 1)), random_state=42)
        df_control = df_all[df_all['label'] == 0].sample(n=min(target_per_class, sum(df_all['label'] == 0)), random_state=42)

        df_balanced = pd.concat([df_distress, df_control]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        # Stratified splits
        train_ratio = train_sample / (train_sample + val_sample + test_sample)
        test_val_ratio = (val_sample + test_sample) / (train_sample + val_sample + test_sample)

        df_train, df_temp = train_test_split(
            df_balanced, train_size=train_sample, stratify=df_balanced['label'], random_state=42
        )
        df_val, df_test = train_test_split(
            df_temp, train_size=val_sample, stratify=df_temp['label'], random_state=42
        )

        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)
        print(f"Saved Reddit splits to {data_dir} (Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)})")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_ds = RedditDataset(df_train['text'], df_train['label'], tokenizer, max_len=max_len)
    val_ds = RedditDataset(df_val['text'], df_val['label'], tokenizer, max_len=max_len)
    test_ds = RedditDataset(df_test['text'], df_test['label'], tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer, df_train, df_val, df_test

if __name__ == "__main__":
    train_l, val_l, test_l, tok, tr, val, ts = prepare_reddit_data()
    print("Train batches:", len(train_l), "Val batches:", len(val_l), "Test batches:", len(test_l))
    print("Test label distribution:\n", ts['label_name'].value_counts())
