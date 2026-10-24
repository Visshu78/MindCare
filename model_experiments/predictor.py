"""
MindCare Standalone Predictor Module.
Provides an easy-to-use inference interface using the trained Proposed Model
(DistilBERT + BiGRU + Attention), outputting Emotion, Sentiment, Confidence,
and Key Attended Words for explainability.
Can be integrated into the MindCare application without altering existing application code.
"""

import os
import torch
import numpy as np
from transformers import DistilBertTokenizerFast
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

try:
    from .models import DistilBERT_BiGRU_Attention
except (ImportError, ModuleNotFoundError):
    try:
        from model_experiments.models import DistilBERT_BiGRU_Attention
    except (ImportError, ModuleNotFoundError):
        from models import DistilBERT_BiGRU_Attention

class MindCarePredictor:
    def __init__(self, checkpoint_path=None, model_dir=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(base_dir, "saved_models", "proposed_best.pt")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBERT_BiGRU_Attention(num_classes=len(LABEL_NAMES))

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained MindCare Proposed Model from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text):
        """Pipeline-compatible interface returning a list with prediction dictionary."""
        res = self.predict(text)
        return [{
            'label': res['emotion'],
            'score': res['confidence'],
            'sentiment': res['sentiment'],
            'top_attended_words': res['top_attended_words'],
            'all_probabilities': res['all_probabilities']
        }]

    def predict(self, text):
        """
        Analyzes a journal entry or text input.
        Returns:
            dict containing emotion, sentiment, confidence, and salient tokens.
        """
        enc = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64, padding=True).to(self.device)

        with torch.no_grad():
            logits, weights = self.model(enc['input_ids'], enc['attention_mask'])
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = LABEL_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

        sentiment = SENTIMENT_MAP.get(pred_label, 'NEUTRAL')

        # Extract top attended tokens (ignoring special tokens)
        tokens = self.tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
        attn = weights[0].cpu().numpy()
        token_attentions = []
        for t, a in zip(tokens, attn):
            if t not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attentions.append((t, float(a)))

        token_attentions.sort(key=lambda x: x[1], reverse=True)
        top_tokens = [t[0] for t in token_attentions[:5]]

        return {
            'text': text,
            'emotion': pred_label,
            'sentiment': sentiment,
            'confidence': confidence,
            'all_probabilities': {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))},
            'top_attended_words': top_tokens
        }

if __name__ == "__main__":
    predictor = MindCarePredictor()
    test_samples = [
        "I feel completely overwhelmed and hopeless with everything going on.",
        "Had a wonderful time with friends and feeling very relaxed!",
        "I'm really terrified about failing my upcoming test tomorrow."
    ]
    for s in test_samples:
        res = predictor.predict(s)
        print("\nInput:", res['text'])
        print(f"  --> Emotion: {res['emotion'].upper()} | Sentiment: {res['sentiment']} ({res['confidence']*100:.1f}%)")
        print("  --> Top Attended Words:", res['top_attended_words'])
