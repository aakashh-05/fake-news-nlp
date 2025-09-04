
# Fake News Detection with NLP (TF‑IDF, GloVe, and Transformers)

A complete, **batteries-included** project to classify news as **fake (0)** or **real (1)** using three approaches:
1) TF‑IDF + Logistic Regression (strong classical baseline)  
2) Averaged **GloVe** word embeddings + Logistic Regression  
3) Fine‑tuned **DistilBERT** (transformer)

> ✅ You can run everything on CPU. DistilBERT will be slower on CPU but still works with the defaults.

---

## 1) Quickstart (works with the included sample)

```bash
# 1. Create venv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# 2. Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 3. Prepare data (uses included data/raw/sample.csv by default)
python src/prepare_data.py --input data/raw/sample.csv

# 4. Train baselines
python src/train_tfidf.py
python src/train_glove.py

# 5. (Optional) Fine‑tune DistilBERT
python src/train_bert.py

# 6. Evaluate all saved models on the test set
python src/evaluate.py

# 7. Try a custom prediction
python src/predict.py --model tfidf --text "This is a shocking secret to become rich overnight!"
```

Linux/macOS users: replace the venv activation line with `source .venv/bin/activate`.

---

## 2) Dataset Options

This repo includes a tiny **sample** at `data/raw/sample.csv` so the pipeline runs end-to-end immediately.

For your real project, choose one public dataset (examples):  
- Kaggle – Fake and Real News Dataset (`Fake.csv`, `True.csv`).  
- LIAR dataset (short statements with truth ratings).  

### Use Kaggle Fake/True CSVs
1. Put `Fake.csv` and `True.csv` into `data/raw/`.  
2. Run:
   ```bash
   python src/prepare_data.py --fake-csv data/raw/Fake.csv --true-csv data/raw/True.csv
   ```

### Use a single CSV
Provide a CSV with columns:  
- `text` (string)  
- `label` (0 for fake, 1 for real)  
Then run:
```bash
python src/prepare_data.py --input path/to/your.csv
```

The script will create stratified **train/val/test** splits here:
```
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

---

## 3) What each approach does

- **TF‑IDF + Logistic Regression**: sparse bag-of-words + n‑grams. Fast, strong baseline.
- **GloVe + Logistic Regression**: download pre-trained 100‑dim vectors (gensim) and average word vectors per document.
- **DistilBERT fine‑tuning**: contextual embeddings trained end-to-end for this task.

All models save to `models/…` and are evaluated with Accuracy, Precision, Recall, F1, and a confusion matrix plot in `reports/figures/`.

---

## 4) Project Structure

```
fake-news-nlp/
├─ data/
│  ├─ raw/                # put your CSVs here (sample.csv included)
│  └─ processed/          # train/val/test CSVs created by prepare_data.py
├─ models/
│  ├─ tfidf_logreg/       # saved scikit-learn pipeline
│  ├─ glove_logreg/       # saved scikit-learn pipeline
│  └─ bert/               # HF model + tokenizer + trainer state
├─ reports/
│  └─ figures/            # confusion matrices, etc.
├─ src/
│  ├─ prepare_data.py     # unify dataset + create splits
│  ├─ preprocess.py       # simple text cleaning (classical models only)
│  ├─ train_tfidf.py      # TF‑IDF + Logistic Regression
│  ├─ train_glove.py      # GloVe averaging + Logistic Regression
│  ├─ train_bert.py       # DistilBERT fine‑tuning
│  ├─ evaluate.py         # evaluate all saved models
│  ├─ predict.py          # run a single prediction from CLI
│  └─ utils.py            # shared helpers (logging, seeding, IO)
├─ config.yaml            # hyperparams + paths
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## 5) Tips & Notes

- **Labels**: by convention here, **0 = FAKE**, **1 = REAL**. The scripts enforce this mapping.
- **Cleaning**: Transformers **do not** use aggressive cleaning; we only strip spaces. Classical models use basic cleaning (lowercase, remove URLs, etc.).
- **Reproducibility**: `seed` is set in `config.yaml`. Exact determinism for GPUs may vary.
- **Hardware**: DistilBERT will download once from Hugging Face on first run. CPU is fine; a GPU (if available) will speed it up automatically.
- **Where to write**: change defaults in `config.yaml` or pass CLI flags to scripts (run with `-h` to see all options).
- **Reports**: Confusion matrices get written to `reports/figures/`.

Good luck — you’ve got a complete, clear, step-by-step pipeline. Read the comments in each script if you’re curious.
