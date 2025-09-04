import pandas as pd

df = pd.read_csv("data/processed/train.csv")
print(df["label"].value_counts())
print(df.head())
