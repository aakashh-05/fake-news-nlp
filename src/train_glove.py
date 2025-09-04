
import argparse
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config, ensure_dir, set_seed, get_logger
from preprocess import clean_for_bow

LOGGER = get_logger("glove")

def plot_confusion(cm, out_path, classes=("FAKE","REAL")):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("GloVe Avg + LogReg Confusion Matrix")
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

def texts_to_avg_vectors(texts, keyed_vectors, dim):
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(tqdm(texts, desc="Embedding")):
        words = t.split()
        if not words:
            continue
        word_vecs = []
        for w in words:
            if w in keyed_vectors:
                word_vecs.append(keyed_vectors[w])
        if word_vecs:
            vectors[i] = np.mean(word_vecs, axis=0)
    return vectors

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

    # Clean and tokenize simply
    train_texts = train_df["text"].map(clean_for_bow).tolist()
    val_texts = val_df["text"].map(clean_for_bow).tolist()
    test_texts = test_df["text"].map(clean_for_bow).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # Download/load GloVe via gensim
    import gensim.downloader as api
    kv_name = cfg["glove"]["vector_name"]
    LOGGER.info(f"Loading vectors: {kv_name} (downloads on first run)")
    keyed_vectors = api.load(kv_name)  # e.g., 100-dim
    dim = keyed_vectors.vector_size

    X_train = texts_to_avg_vectors(train_texts, keyed_vectors, dim)
    X_val = texts_to_avg_vectors(val_texts, keyed_vectors, dim)
    X_test = texts_to_avg_vectors(test_texts, keyed_vectors, dim)

    C = args.C if args.C is not None else cfg["glove"]["C"]
    clf = LogisticRegression(C=C, max_iter=cfg["glove"]["max_iter"])
    clf.fit(X_train, y_train)

    def eval_split(name, X, y):
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds, target_names=["FAKE","REAL"], output_dict=True, zero_division=0)
        cm = confusion_matrix(y, preds)
        LOGGER.info(f"{name} accuracy: {acc:.4f}")
        return acc, report, cm

    results = {}
    for split_name, X, y in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        acc, report, cm = eval_split(split_name, X, y)
        results[split_name] = {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}
        if split_name == "test":
            out_fig = os.path.join(cfg["paths"]["reports_dir"], "figures", "cm_glove.png")
            os.makedirs(os.path.dirname(out_fig), exist_ok=True)
            plot_confusion(cm, out_fig)

    out_dir = os.path.join(cfg["paths"]["models_dir"], "glove_logreg")
    ensure_dir(out_dir)
    joblib.dump(clf, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "vector_info.json"), "w") as f:
        json.dump({"vector_name": kv_name, "dim": dim}, f)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"Saved model to {out_dir}")

if __name__ == "__main__":
    main()
