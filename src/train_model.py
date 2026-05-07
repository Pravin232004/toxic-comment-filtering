import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


def train_model():

    print("Loading data...")
    df = pd.read_csv("data/processed/cleaned_data.csv")

    # Features & Labels
    X = df["clean_text"].fillna("").astype(str)
    y = df[LABEL_COLS]

    # Remove empty rows
    mask = X.str.strip() != ""
    X = X[mask]
    y = y[mask]

    print(f"Data shape: {X.shape}")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Strong TF-IDF
    print("Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=50000,       # more vocabulary
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words='english',     # remove common words
        min_df=2,                 # ignore rare words
        max_df=0.9                # ignore overly common words
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

   
    # Strong Model
    print("Training model...")

    base_model = LogisticRegression(
        class_weight='balanced',  # fixes imbalance
        max_iter=1000,
        C=1.5                     # slight regularization tuning
    )

    model = OneVsRestClassifier(base_model)

    model.fit(X_train_vec, y_train)

    print("Model training completed")

    # Save model
    print("Saving model...")

    joblib.dump(model, "models/toxicity_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model & vectorizer saved successfully!")


if __name__ == "__main__":
    train_model()