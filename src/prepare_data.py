
import argparse
import os
import pandas as pd
from utils import load_config, ensure_dir, train_val_test_split, get_logger

LOGGER = get_logger("prepare")

def from_fake_true(fake_csv: str, true_csv: str) -> pd.DataFrame:
    fake = pd.read_csv(fake_csv)
    true = pd.read_csv(true_csv)
    # Try to build a 'text' column
    def build_text(df):
        cols = [c for c in ["title","text","subject"] if c in df.columns]
        if not cols:
            # Fallback to concatenating all string columns
            cols = [c for c in df.columns if df[c].dtype==object]
        return df[cols].astype(str).agg(" ".join, axis=1)
    fake_text = build_text(fake)
    true_text = build_text(true)
    df = pd.concat(
        [
            pd.DataFrame({"text": fake_text, "label": 0}),
            pd.DataFrame({"text": true_text, "label": 1}),
        ],
        ignore_index=True,
    )
    return df

def from_single_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Map labels to {0,1} if necessary
    if df[label_col].dtype == object:
        mapping = {"fake": 0, "FAKE": 0, "False": 0, "FALSE": 0, "true": 1, "TRUE": 1, "True": 1}
        df[label_col] = df[label_col].map(lambda x: mapping.get(str(x), x))
    df = df.rename(columns={text_col: "text", label_col: "label"})
    # Enforce 0/1
    df["label"] = df["label"].astype(int)
    return df[["text","label"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/sample.csv", help="Single CSV path with columns text,label (0/1)")
    parser.add_argument("--text-col", type=str, default=None, help="If your CSV uses a different text column name")
    parser.add_argument("--label-col", type=str, default=None, help="If your CSV uses a different label column name")
    parser.add_argument("--fake-csv", type=str, default=None, help="Path to Fake.csv")
    parser.add_argument("--true-csv", type=str, default=None, help="Path to True.csv")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = cfg["paths"]["processed_dir"]
    ensure_dir(out_dir)

    if args.fake_csv and args.true_csv:
        LOGGER.info(f"Loading Kaggle-style Fake/True CSVs: {args.fake_csv} / {args.true_csv}")
        df = from_fake_true(args.fake_csv, args.true_csv)
    else:
        LOGGER.info(f"Loading single CSV: {args.input}")
        text_col = args.text_col or cfg["data"]["text_col"]
        label_col = args.label_col or cfg["data"]["label_col"]
        df = from_single_csv(args.input, text_col, label_col)

    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=cfg["seed"]).reset_index(drop=True)

    # Drop empties
    df = df.dropna(subset=["text","label"]).reset_index(drop=True)

    # Splits
    train_df, val_df, test_df = train_val_test_split(
        df,
        val_size=cfg["data"]["val_size"],
        test_size=cfg["data"]["test_size"],
        stratify=cfg["data"]["stratify"],
        seed=cfg["seed"],
    )

    # Save
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    LOGGER.info(f"Wrote: {out_dir}/train.csv, val.csv, test.csv (sizes: {len(train_df)}, {len(val_df)}, {len(test_df)})")

if __name__ == "__main__":
    main()
