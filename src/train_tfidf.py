
import argparse
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from utils import load_config, ensure_dir, set_seed, get_logger
from preprocess import clean_for_bow

LOGGER = get_logger("tfidf")

def plot_confusion(cm, out_path, classes=("FAKE","REAL")):
    fig = plt.figure()
    import numpy as np
    plt.imshow(cm, interpolation='nearest')
    plt.title("TF-IDF + LogReg Confusion Matrix")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])

    train_path = os.path.join(cfg["paths"]["processed_dir"], "train.csv")
    val_path = os.path.join(cfg["paths"]["processed_dir"], "val.csv")
    test_path = os.path.join(cfg["paths"]["processed_dir"], "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Clean for BoW
    train_texts = train_df["text"].map(clean_for_bow).tolist()
    val_texts = val_df["text"].map(clean_for_bow).tolist()
    test_texts = test_df["text"].map(clean_for_bow).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    vectorizer = TfidfVectorizer(
        max_features=cfg["tfidf"]["max_features"],
        ngram_range=(cfg["tfidf"]["ngram_min"], cfg["tfidf"]["ngram_max"]),
        min_df=cfg["tfidf"]["min_df"],
    )

    C = args.C if args.C is not None else cfg["tfidf"]["C"]
    clf = LogisticRegression(C=C, max_iter=cfg["tfidf"]["max_iter"])

    pipe = Pipeline([("tfidf", vectorizer), ("logreg", clf)])
    pipe.fit(train_texts, y_train)

    # Evaluate on val+test
    def eval_split(name, texts, labels):
        preds = pipe.predict(texts)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=["FAKE","REAL"], output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds)
        LOGGER.info(f"{name} accuracy: {acc:.4f}")
        return acc, report, cm

    results = {}
    for split_name, texts, labels in [
        ("val", val_texts, y_val),
        ("test", test_texts, y_test),
    ]:
        acc, report, cm = eval_split(split_name, texts, labels)
        results[split_name] = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}
        # Plot confusion matrix for test only
        if split_name == "test":
            out_fig = os.path.join(cfg["paths"]["reports_dir"], "figures", "cm_tfidf.png")
            os.makedirs(os.path.dirname(out_fig), exist_ok=True)
            plot_confusion(cm, out_fig)

    # Save model
    out_dir = os.path.join(cfg["paths"]["models_dir"], "tfidf_logreg")
    ensure_dir(out_dir)
    joblib.dump(pipe, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2)

    LOGGER.info(f"Saved model to {out_dir}")

if __name__ == "__main__":
    main()
