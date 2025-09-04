
import os
import json
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
import yaml
from typing import Tuple

def get_logger(name: str = "app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(ch)
    return logger

LOGGER = get_logger()

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def train_val_test_split(df: pd.DataFrame, val_size=0.15, test_size=0.15, stratify=True, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    y = df["label"] if ("label" in df.columns) else None
    stratify_y = y if (stratify and y is not None) else None
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, stratify=stratify_y, random_state=seed)
    rel_test_size = test_size / (val_size + test_size)
    y_temp = temp_df["label"] if ("label" in temp_df.columns) else None
    stratify_temp = y_temp if (stratify and y_temp is not None) else None
    val_df, test_df = train_test_split(temp_df, test_size=rel_test_size, stratify=stratify_temp, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
