import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import torch

# HuggingFace
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("sms_fraud_dataset.csv")  # replace with your dataset file
print("Dataset Shape:", df.shape)
print(df['label'].value_counts())

# Encode labels
label_map = {"genuine": 0, "suspected": 1, "confirmed_spam": 2}
df["label_id"] = df["label"].map(label_map)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label_id"], test_size=0.2, random_state=42, stratify=df["label_id"]
)

# ==============================
# 2. Classical ML Models
# ==============================
print("\n=== Classical ML Models ===")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

classical_results = {}
best_classical = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, train_labels)
    preds = model.predict(X_test)

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="weighted")
    prec = precision_score(test_labels, preds, average="weighted")
    rec = recall_score(test_labels, preds, average="weighted")

    print(f"\n{name} Results:")
    print(classification_report(test_labels, preds, target_names=label_map.keys()))

    classical_results[name] = {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

    if f1 > best_f1:
        best_f1 = f1
        best_classical = model

# Save best TF-IDF model
joblib.dump((vectorizer, best_classical), "best_classical_model.pkl")
print("\nBest Classical Model Saved:", type(best_classical).__name__)

# ==============================
# 3. Transformer Models
# ==============================
print("\n=== Transformer Models ===")

def train_transformer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": train_texts, "label": train_labels
    }))
    test_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": test_texts, "label": test_labels
    }))

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.replace('/', '_')}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True
    )

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(pred.label_ids, preds)
        f1 = f1_score(pred.label_ids, preds, average="weighted")
        prec = precision_score(pred.label_ids, preds, average="weighted")
        rec = recall_score(pred.label_ids, preds, average="weighted")
        return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"\n{model_name} Evaluation:", eval_results)

    # Save fine-tuned transformer
    model.save_pretrained(f"best_{model_name.replace('/', '_')}")
    tokenizer.save_pretrained(f"best_{model_name.replace('/', '_')}")
    return eval_results

transformer_models = [
    "distilbert-base-multilingual-cased",
    "ai4bharat/IndicBERTv2-MLM",
    "xlm-roberta-base"
]

transformer_results = {}
for tm in transformer_models:
    transformer_results[tm] = train_transformer(tm)

# ==============================
# 4. Leaderboard
# ==============================
print("\n=== Model Leaderboard ===")

def print_results(name, res):
    print(f"{name:35s} | Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f}")

for name, res in classical_results.items():
    print_results(name, res)

for name, res in transformer_results.items():
    print_results(name, res)

print("\n✅ Training completed. Best models saved.")
