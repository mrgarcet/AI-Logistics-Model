#----------------------------------------------------
        # Path: src/nlp/train_intents.py
#----------------------------------------------------
'''
Fine‑tunes DistilBERT on inventory intents.
Usage:  python -m src.nlp.train_intents
'''


#----------------------------------------------------
        # Import necessary libraries and models
#----------------------------------------------------
import os, random, torch, pandas as pd
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


#----------------------------------------------------
                # Model configuration
#----------------------------------------------------
DATA_PATH = "data/intents.csv"
MODEL_DIR = "models/bert_intent_classifier"
BASE_MODEL = "distilbert-base-uncased"
NUM_EPOCHS = 3    # Epochs value is how many times the model will run through the dataset
BATCH = 16        # Number of samples process before updating model weights
SEED = 42         # Seed for reproducibility to ensure consistent random results


#----------------------------------------------------
          # Load CSV and create Label mapping
#----------------------------------------------------
df = pd.read_csv(DATA_PATH)                            # Load path from variable DATA_PATH
labels = sorted(df["intent"].unique())                 # Create sorted list of unique intent
label2id = {l: i for i, l in enumerate(labels)}        # Map each unique attempt to a numerical ID
id2label = {i: l for l, i in label2id.items()}         # Map numerical ID back to corresponding label
df["label"] = df["intent"].map(label2id)               # Convent intent label to numerical IDs in dataframe


#----------------------------------------------------
     # Splitting data for Training and Validation
#----------------------------------------------------
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)        # Randomizes the data
split = int(len(df) * 0.85)                                             # Divide data with 85% Training 15% Testing
train_ds = Dataset.from_pandas(df.iloc[:split][["text", "label"]])      # Set training data to variable train_ds
val_ds   = Dataset.from_pandas(df.iloc[split:][["text", "label"]])      # Set training data to variable val_ds


#----------------------------------------------------
                 # Tokenisation
#----------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE_MODEL)         # Initialise tokenizer

def tokenize(batch):                                    # Helper → encode each text
    return tok(batch["text"],
               truncation=True,
               padding="max_length",
               max_length=64)

train_ds = train_ds.map(tokenize, batched=True)         # Encode training split
val_ds   = val_ds.map(tokenize, batched=True)           # Encode validation split
train_ds = train_ds.rename_column("label", "labels")    # PyTorch Trainer expects 'labels'
val_ds   = val_ds.rename_column("label", "labels")
train_ds.set_format("torch")                            # Returns PyTorch tensors
val_ds.set_format("torch")


#----------------------------------------------------
                # Model definition
#----------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


#----------------------------------------------------
                # Evaluation metric
#----------------------------------------------------
metric = evaluate.load("accuracy")                      # Simple classification accuracy

def compute_metrics(p):                                 # Hugging‑Face Trainer callback
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds,
                          references=p.label_ids)


#----------------------------------------------------
            # Training hyper‑parameters
#----------------------------------------------------
args = TrainingArguments(
    output_dir="models/tmp_run",                        # Where checkpoints/logs go
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    logging_steps=50,
    seed=SEED,
    load_best_model_at_end=False                        # Simpler config for now
)

#----------------------------------------------------
                    # Training loop
#----------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

#----------------------------------------------------
                # Save artefacts
#----------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)                        # Model weights/config
tok.save_pretrained(MODEL_DIR)                          # Tokeniser files
print(f"Model saved to {MODEL_DIR}")
