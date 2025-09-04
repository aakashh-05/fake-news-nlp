
import argparse
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

from utils import load_config, ensure_dir, set_seed, get_logger

LOGGER = get_logger("bert")

def plot_confusion(cm, out_path, classes=("FAKE","REAL")):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("DistilBERT Confusion Matrix")
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        import torch as _torch
        item = {k: _torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = _torch.tensor(int(self.labels[idx]))
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])

    model_name = args.model_name or cfg["bert"]["model_name"]
    epochs = args.epochs or cfg["bert"]["num_train_epochs"]
    max_length = cfg["bert"]["max_length"]

    train_df = pd.read_csv(os.path.join(cfg["paths"]["processed_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(cfg["paths"]["processed_dir"], "val.csv"))
    test_df = pd.read_csv(os.path.join(cfg["paths"]["processed_dir"], "test.csv"))

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = NewsDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length)
    test_dataset = NewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, max_length)

    out_dir = os.path.join(cfg["paths"]["models_dir"], "bert")
    ensure_dir(out_dir)

    # Force convert values to correct types (fixes string vs float issue)
    lr = float(cfg["bert"]["learning_rate"])
    wd = float(cfg["bert"]["weight_decay"])
    epochs = int(cfg["bert"]["num_train_epochs"])
    batch_train = int(cfg["bert"]["per_device_train_batch_size"])
    batch_eval = int(cfg["bert"]["per_device_eval_batch_size"])
    warmup = float(cfg["bert"]["warmup_ratio"])
    fp16 = bool(cfg["bert"]["fp16"])


    training_args = TrainingArguments(
        output_dir=out_dir,
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_train,
        per_device_eval_batch_size=batch_eval,
        num_train_epochs=epochs,
        weight_decay=wd,
        warmup_ratio=warmup,
        logging_steps=50,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        report_to=[],
    )


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    preds = trainer.predict(test_dataset)
    metrics = preds.metrics
    LOGGER.info(f"Test metrics: {metrics}")
    # Confusion matrix
    from sklearn.metrics import confusion_matrix as _cm
    y_true = test_df["label"].values
    import numpy as _np
    y_pred = _np.argmax(preds.predictions, axis=-1)
    cm = _cm(y_true, y_pred)
    out_fig = os.path.join(cfg["paths"]["reports_dir"], "figures", "cm_bert.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plot_confusion(cm, out_fig)

    # Save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()
