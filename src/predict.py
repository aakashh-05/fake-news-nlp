
import argparse
import os
import joblib
from utils import load_config, get_logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

LOGGER = get_logger("predict")

def predict_tfidf(text, models_dir):
    pipe = joblib.load(os.path.join(models_dir, "tfidf_logreg", "model.joblib"))
    pred = int(pipe.predict([text])[0])
    proba = pipe.predict_proba([text])[0][pred]
    return pred, float(proba)

def predict_bert(text, models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(models_dir, "bert")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    enc = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    proba = torch.softmax(logits, dim=-1)[0]
    pred = int(torch.argmax(proba).cpu().item())
    return pred, float(proba[pred].cpu().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["tfidf","bert"], default="tfidf")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config()
    label_map = {0: "FAKE", 1: "REAL"}

    if args.model == "tfidf":
        pred, score = predict_tfidf(args.text, cfg["paths"]["models_dir"])
    else:
        pred, score = predict_bert(args.text, cfg["paths"]["models_dir"])

    LOGGER.info(f"Prediction: {label_map[pred]} (score={score:.3f})")

if __name__ == "__main__":
    main()
