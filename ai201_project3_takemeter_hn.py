# -*- coding: utf-8 -*-
"""TakeMeter starter script customized for a Hacker News comment dataset.

This is a Python-exported version of the starter Colab notebook with the two
main student TODOs already filled in:

1. LABEL_MAP
2. SYSTEM_PROMPT

Run the original notebook in Colab for the actual project workflow.
"""

import json
import subprocess
import sys
import time
import warnings

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "groq", "python-dotenv"],
    check=True,
)
print("Dependencies ready")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

print("Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

"""---
## Section 1: Load Your Dataset

Upload your labeled CSV and define your label map.
Your CSV must have at least two columns: `text` and `label`.
"""

LABEL_MAP = {
    "analysis": 0,
    "experience": 1,
    "reaction": 2,
}

ID_TO_LABEL = {value: key for key, value in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
print(f"Labels: {LABEL_MAP}")
print(f"Number of labels: {NUM_LABELS}")

from google.colab import files

print("Select your labeled dataset CSV file...")
uploaded = files.upload()
CSV_PATH = next(iter(uploaded))
print(f"Uploaded: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print(f"Columns: {df.columns.tolist()}")
print(f"Total examples: {len(df)}")
print()
print("Label distribution:")
print(df["label"].value_counts())

unknown = set(df["label"].unique()) - set(LABEL_MAP.keys())
if unknown:
    print(f"Labels in CSV not found in LABEL_MAP: {unknown}")
    print("Update your LABEL_MAP above to include all labels.")
else:
    print("All labels match your LABEL_MAP")

df["label_id"] = df["label"].map(LABEL_MAP)
df = df.dropna(subset=["label_id"])
df["label_id"] = df["label_id"].astype(int)

"""---
## Section 2: Prepare Data for Training
"""

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_id"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
)

print(f"Train: {len(train_df)} examples")
print(f"Validation: {len(val_df)} examples")
print(f"Test: {len(test_df)} examples")
print()
print("Train label distribution:")
print(train_df["label"].value_counts())
print()
print("Test label distribution:")
print(test_df["label"].value_counts())

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)


def make_dataset(df_split):
    dataset = Dataset.from_pandas(
        df_split[["text", "label_id"]].rename(columns={"label_id": "labels"})
    )
    return dataset.map(tokenize, batched=True)


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)
test_dataset = make_dataset(test_df)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenization complete")
print(f"Sample keys: {list(train_dataset[0].keys())}")

"""---
## Section 3: Fine-Tune Your Model
"""

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID_TO_LABEL,
    label2id=LABEL_MAP,
)
print(f"Model loaded: {MODEL_NAME}")
print(f"Output labels: {NUM_LABELS}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


training_args = TrainingArguments(
    output_dir="./takemeter-model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning... (5-15 minutes on T4 GPU)")
trainer.train()
print("Fine-tuning complete")

"""---
## Section 4: Evaluate Fine-Tuned Model on Test Set
"""

print("Running inference on test set...")
ft_output = trainer.predict(test_dataset)
ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)
ft_true_ids = ft_output.label_ids

ft_probs = torch.nn.functional.softmax(
    torch.tensor(ft_output.predictions), dim=-1
).numpy()

ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)
print(f"Fine-tuned model accuracy: {ft_accuracy:.3f}")

label_names = [ID_TO_LABEL[index] for index in range(NUM_LABELS)]
print("Per-class metrics (fine-tuned model):")
print(
    classification_report(
        ft_true_ids,
        ft_pred_ids,
        target_names=label_names,
        zero_division=0,
    )
)

cm = confusion_matrix(ft_true_ids, ft_pred_ids)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fine-Tuned Model - Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

wrong_idx = np.nonzero(ft_pred_ids != ft_true_ids)[0]
print(f"Wrong predictions: {len(wrong_idx)} / {len(ft_true_ids)}")

for i, idx in enumerate(wrong_idx[:15]):
    text = test_df.iloc[idx]["text"]
    true_label = ID_TO_LABEL[ft_true_ids[idx]]
    pred_label = ID_TO_LABEL[ft_pred_ids[idx]]
    confidence = ft_probs[idx][ft_pred_ids[idx]]
    print(f"--- #{i + 1} ---")
    print(f"Text:      {text[:200]}{'...' if len(text) > 200 else ''}")
    print(f"True:      {true_label}")
    print(f"Predicted: {pred_label}  (confidence: {confidence:.2f})")
    print()

"""---
## Section 5: Baseline Classifier (Groq)
"""

from groq import Groq
from google.colab import userdata

GROQ_API_KEY = userdata.get("GROQ_API_KEY")

assert GROQ_API_KEY, (
    "GROQ_API_KEY not set. Add it in Colab Secrets and enable notebook access, "
    "or paste it directly in the notebook if needed."
)

client = Groq(api_key=GROQ_API_KEY)
print("Groq client initialized")

SYSTEM_PROMPT = """
You are classifying Hacker News comments by discourse style.
Assign each comment to exactly one label.

analysis: The comment mainly explains, argues, or evaluates with concrete reasoning, mechanism, evidence, or comparison.
Example: "He actively campaigned against any regulation of derivatives. There is an infamous lunch that he had with Brooksley Born ... where she attempted to regulate them."

experience: The comment mainly relies on the writer's own experience, project, job, or personal example to make its point.
Example: "Nice job! I made something similar, mostly for myself: https://www.vexling.com Plan to keep it forever free :)"

reaction: The comment is primarily a quick reaction, joke, praise, dismissal, or unsupported opinion rather than developed reasoning.
Example: "Yeah, that irked me a bit but maybe it was intentional?"

Respond with ONLY the label name.
Do not explain your reasoning.

Valid labels:
analysis
experience
reaction
"""

print("Prompt length:", len(SYSTEM_PROMPT), "characters")


def classify_with_groq(text):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this comment:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        for label in sorted(LABEL_MAP, key=len, reverse=True):
            if raw == label or label in raw:
                return label
        return None
    except Exception as error:
        print(f"API error: {error}")
        return None


print(f"Running baseline on {len(test_df)} examples...")
baseline_preds = []
for i, (_, row) in enumerate(test_df.iterrows()):
    pred = classify_with_groq(row["text"])
    baseline_preds.append(pred)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(test_df)} complete...")
    time.sleep(0.1)

none_count = baseline_preds.count(None)
if none_count > 0:
    print(f"{none_count} responses could not be parsed.")

valid = [(pred, true_id) for pred, true_id in zip(baseline_preds, test_df["label_id"]) if pred is not None]
bl_pred_ids = [LABEL_MAP[pred] for pred, _ in valid]
bl_true_ids = [true_id for _, true_id in valid]

bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
print(
    f"Baseline accuracy: {bl_accuracy:.3f} "
    f"(evaluated on {len(valid)}/{len(test_df)} parseable responses)"
)
print("Per-class metrics (baseline):")
print(
    classification_report(
        bl_true_ids,
        bl_pred_ids,
        target_names=label_names,
        zero_division=0,
    )
)

"""---
## Section 6: Compare Results and Export
"""

print("=" * 50)
print("RESULTS COMPARISON")
print("=" * 50)
print(f"{'Model':<35} {'Accuracy':>8}")
print("-" * 45)
print(f"{'Zero-shot baseline (Groq)':<35} {bl_accuracy:>8.3f}")
print(f"{'Fine-tuned DistilBERT':<35} {ft_accuracy:>8.3f}")
print("-" * 45)
delta = ft_accuracy - bl_accuracy
direction = "improvement" if delta >= 0 else "regression"
print(f"Fine-tuning {direction}: {abs(delta):.3f}")

results = {
    "baseline_accuracy": round(bl_accuracy, 4),
    "finetuned_accuracy": round(ft_accuracy, 4),
    "improvement": round(ft_accuracy - bl_accuracy, 4),
    "test_set_size": len(test_df),
    "label_map": LABEL_MAP,
    "model": MODEL_NAME,
}
with open("evaluation_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)

print("Files ready to download:")
print("  evaluation_results.json")
print("  confusion_matrix.png")