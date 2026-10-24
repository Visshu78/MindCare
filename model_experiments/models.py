"""
Neural Network Architectures for MindCare Mental Well-Being Analysis.
Implements the 4 models defined in Slide 10, 11, and 12 of Deep Learning Mini Project.pptx:
1. Baseline 1: DistilBERT -> Classifier
2. Baseline 2: DistilBERT -> BiGRU -> Classifier
3. Proposed Model: DistilBERT -> BiGRU -> Attention -> Classifier
4. Additional Experiment: DistilBERT -> CNN -> BiGRU -> Attention -> Classifier
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel

class AttentionLayer(nn.Module):
    """
    Additive Attention Mechanism (Slide 10 & 11)
    Assigns greater importance to informative emotional/mental-health tokens.
    """
    def __init__(self, hidden_dim, attention_dim=128):
        super(AttentionLayer, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, sequence_output, attention_mask=None):
        # sequence_output: [batch_size, seq_len, hidden_dim]
        # u: [batch_size, seq_len, attention_dim]
        u = self.projection(sequence_output)
        
        # scores: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        scores = self.context_vector(u).squeeze(-1)

        if attention_mask is not None:
            # Mask out padding tokens (where attention_mask is 0)
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # attention_weights: [batch_size, seq_len]
        weights = torch.softmax(scores, dim=-1)

        # weighted_representation: [batch_size, hidden_dim]
        weighted_output = torch.bmm(weights.unsqueeze(1), sequence_output).squeeze(1)

        return weighted_output, weights

class DistilBERTBaseline(nn.Module):
    """
    Baseline 1: DistilBERT -> Classifier (Slide 12)
    Uses the contextual representation of the [CLS] token.
    """
    def __init__(self, num_classes=6, dropout=0.2, pretrained_model='distilbert-base-uncased'):
        super(DistilBERTBaseline, self).__init__()
        self.model_name = "DistilBERT Baseline"
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (index 0)
        cls_rep = distilbert_out.last_hidden_state[:, 0, :]
        out = self.dropout(cls_rep)
        logits = self.classifier(out)
        return logits, None

class DistilBERT_BiGRU(nn.Module):
    """
    Baseline 2: DistilBERT -> BiGRU -> Classifier (Slide 12)
    BiGRU captures sequential forward + backward context relationships.
    """
    def __init__(self, num_classes=6, gru_hidden_dim=128, dropout=0.2, pretrained_model='distilbert-base-uncased'):
        super(DistilBERT_BiGRU, self).__init__()
        self.model_name = "DistilBERT + BiGRU"
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        hidden_size = self.distilbert.config.hidden_size

        self.bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(gru_hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_rep = distilbert_out.last_hidden_state  # [B, L, 768]

        gru_out, h_n = self.bigru(seq_rep)  # h_n: [2, B, gru_hidden_dim]
        # Concatenate final forward and backward states
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        context = torch.cat([forward_hidden, backward_hidden], dim=-1)  # [B, gru_hidden_dim * 2]

        out = self.dropout(context)
        logits = self.classifier(out)
        return logits, None

class DistilBERT_BiGRU_Attention(nn.Module):
    """
    Proposed Model: DistilBERT -> BiGRU -> Attention -> Classifier (Slide 10, 11, 12)
    Combines transformer contextual representations, bidirectional recurrent modeling,
    and additive attention weighting over informative text tokens.
    """
    def __init__(self, num_classes=6, gru_hidden_dim=128, attention_dim=128, dropout=0.2, pretrained_model='distilbert-base-uncased'):
        super(DistilBERT_BiGRU_Attention, self).__init__()
        self.model_name = "DistilBERT + BiGRU + Attention (Proposed)"
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        hidden_size = self.distilbert.config.hidden_size

        self.bigru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(
            hidden_dim=gru_hidden_dim * 2,
            attention_dim=attention_dim
        )
        self.dense = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_rep = distilbert_out.last_hidden_state  # [B, L, 768]

        gru_out, _ = self.bigru(seq_rep)  # [B, L, 2 * gru_hidden_dim]
        weighted_rep, attention_weights = self.attention(gru_out, attention_mask)  # [B, 2 * gru_hidden_dim]

        dense_out = self.dense(weighted_rep)
        logits = self.classifier(dense_out)
        return logits, attention_weights

class DistilBERT_CNN_BiGRU_Attention(nn.Module):
    """
    Additional Experiment: DistilBERT -> CNN -> BiGRU -> Attention -> Classifier (Slide 12)
    Adds 1D Convolution for extracting local n-gram features prior to recurrent & attention layers.
    """
    def __init__(self, num_classes=6, cnn_filters=256, gru_hidden_dim=128, attention_dim=128, dropout=0.2, pretrained_model='distilbert-base-uncased'):
        super(DistilBERT_CNN_BiGRU_Attention, self).__init__()
        self.model_name = "DistilBERT + CNN + BiGRU + Attention (Additional)"
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model)
        hidden_size = self.distilbert.config.hidden_size

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=cnn_filters, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.bigru = nn.GRU(
            input_size=cnn_filters,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionLayer(
            hidden_dim=gru_hidden_dim * 2,
            attention_dim=attention_dim
        )
        self.dense = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        distilbert_out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        seq_rep = distilbert_out.last_hidden_state  # [B, L, 768]

        # Conv1D expects [Batch, Channels, Length]
        conv_in = seq_rep.transpose(1, 2)
        conv_out = self.conv1d(conv_in)  # [B, cnn_filters, L]
        conv_out = conv_out.transpose(1, 2)  # [B, L, cnn_filters]

        gru_out, _ = self.bigru(conv_out)  # [B, L, 2 * gru_hidden_dim]
        weighted_rep, attention_weights = self.attention(gru_out, attention_mask)

        dense_out = self.dense(weighted_rep)
        logits = self.classifier(dense_out)
        return logits, attention_weights

def get_model(model_type, num_classes=6, **kwargs):
    """Factory function for instantiating model variants."""
    models = {
        'baseline1': DistilBERTBaseline,
        'baseline2': DistilBERT_BiGRU,
        'proposed': DistilBERT_BiGRU_Attention,
        'additional': DistilBERT_CNN_BiGRU_Attention
    }
    if model_type not in models:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(models.keys())}")
    return models[model_type](num_classes=num_classes, **kwargs)
