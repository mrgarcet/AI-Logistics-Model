# src/nlp/infer.py
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# path relative to *project root*  (no drive‑letter!)
MODEL_DIR = Path("models/bert_intent_classifier")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
)
model.eval()
id2label = model.config.id2label


def predict_intent(text: str) -> str:
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return id2label[int(logits.argmax())]


if __name__ == "__main__":
    print(predict_intent("How many of Item A do we have?"))
    print(predict_intent("Update stock of Item B to 10"))
