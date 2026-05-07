import pandas as pd
import re
import emoji

LABEL_COLS = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

def clean_text(text: str) -> str:
    text = text.lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

def preprocess_data():
    df = pd.read_csv("data/raw/train.csv")

    df["clean_text"] = df["comment_text"].astype(str).apply(clean_text)

    df = df[["clean_text"] + LABEL_COLS]

    df.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Preprocessing completed. cleaned_data.csv created")

if __name__ == "__main__":
    preprocess_data()

