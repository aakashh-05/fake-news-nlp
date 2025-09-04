
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from utils import load_config, get_logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LOGGER = get_logger("eval")

def eval_tfidf(test_df, models_dir):
    path = os.path.join(models_dir, "tfidf_logreg", "model.joblib")
    if not os.path.exists(path):
        LOGGER.info("TF-IDF model not found; skipping.")
        return None
    pipe = joblib.load(path)
    preds = pipe.predict(test_df["text"].tolist())
    acc = accuracy_score(test_df["label"], preds)
    report = classification_report(test_df["label"], preds, target_names=["FAKE","REAL"], output_dict=True, zero_division=0)
    return {"accuracy": acc, "report": report}

def eval_glove(test_df, models_dir):
    path = os.path.join(models_dir, "glove_logreg", "model.joblib")
    if not os.path.exists(path):
        LOGGER.info("GloVe model not found; skipping.")
        return None
    clf = joblib.load(path)
    # Rebuild GloVe averages for test set
    from preprocess import clean_for_bow
    import gensim.downloader as api
    import numpy as np
    kv = api.load("glove-wiki-gigaword-100")
    dim = kv.vector_size
    def avg_vec(text):
        words = clean_for_bow(text).split()
        vecs = [kv[w] for w in words if w in kv]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)
    X = np.vstack([avg_vec(t) for t in test_df["text"].tolist()])
    preds = clf.predict(X)
    acc = accuracy_score(test_df["label"], preds)
    report = classification_report(test_df["label"], preds, target_names=["FAKE","REAL"], output_dict=True, zero_division=0)
    return {"accuracy": acc, "report": report}

def eval_bert(test_df, models_dir):
    path = os.path.join(models_dir, "bert")
    if not os.path.exists(path):
        LOGGER.info("BERT model dir not found; skipping.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    texts = test_df["text"].tolist()
    labels = test_df["label"].values
    all_preds = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        pred = int(torch.argmax(logits, dim=-1).cpu().item())
        all_preds.append(pred)
    acc = accuracy_score(labels, all_preds)
    report = classification_report(labels, all_preds, target_names=["FAKE","REAL"], output_dict=True, zero_division=0)
    return {"accuracy": acc, "report": report}

def main():
    cfg = load_config()
    test_df = pd.read_csv(os.path.join(cfg["paths"]["processed_dir"], "test.csv"))
    models_dir = cfg["paths"]["models_dir"]

    results = {}
    res = eval_tfidf(test_df, models_dir)
    if res: results["tfidf_logreg"] = res

    res = eval_glove(test_df, models_dir)
    if res: results["glove_logreg"] = res

    res = eval_bert(test_df, models_dir)
    if res: results["bert"] = res

    out_path = os.path.join(cfg["paths"]["reports_dir"], "evaluation_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info(f"Wrote {out_path}")
    for k, v in results.items():
        LOGGER.info(f"{k}: accuracy={v['accuracy']:.4f}")

if __name__ == "__main__":
    main()
