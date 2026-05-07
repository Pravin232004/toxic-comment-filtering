import pandas as pd
import joblib
import warnings
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning

from moderation_rules import compute_severity, moderation_action

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Thresholds (tunable)
THRESHOLDS = {
    "toxic": 0.4,
    "severe_toxic": 0.3,
    "obscene": 0.4,
    "threat": 0.25,
    "insult": 0.4,
    "identity_hate": 0.3
}


def evaluate_model():

    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Load dataset
    df = pd.read_csv("data/processed/cleaned_data.csv")

    X = df["clean_text"]
    y = df[LABEL_COLS]

    # Train-validation split
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model + vectorizer
    model = joblib.load("models/toxicity_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Clean validation data
    X_val = X_val.fillna("").astype(str)
    mask = X_val.str.strip() != ""

    X_val = X_val[mask]
    y_val = y_val[mask]

    # Shuffle for randomness
    X_val = X_val.sample(frac=1, random_state=42)
    y_val = y_val.loc[X_val.index]

    # Vectorize
    X_val_vec = vectorizer.transform(X_val)

    # Probability-based Prediction
    y_proba = model.predict_proba(X_val_vec)

    y_pred = []
    for row in y_proba:
        pred_row = []
        for i, label in enumerate(LABEL_COLS):
            prob = row[i] 
            pred = 1 if prob >= THRESHOLDS[label] else 0
            pred_row.append(pred)
        y_pred.append(pred_row)

    y_pred = np.array(y_pred)

    # Classification Report
    print("\nEvaluation Report:\n")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=LABEL_COLS,
            zero_division=0
        )
    )

    # Label Distribution
    print("\nLabel Distribution (Validation Set):\n")
    print(y_val.sum())

    # Sample Predictions
    print("\nSample Predictions (Actual vs Predicted):\n")

    for i in range(5):
        actual_labels = dict(zip(LABEL_COLS, y_val.iloc[i]))
        predicted_labels = dict(zip(LABEL_COLS, y_pred[i]))

        probs = {
            label: round(y_proba[i][j], 3)
            for j, label in enumerate(LABEL_COLS)
        }

        severity = compute_severity(predicted_labels)
        action = moderation_action(severity)

        print(f"Sample {i + 1}")
        print("Text:", X_val.iloc[i])
        print("Actual:", actual_labels)
        print("Probabilities:", probs)
        print("Predicted:", predicted_labels)
        print("Severity score:", severity)
        print("Moderation action:", action)
        print("-" * 60)

    # Toxic Sample Testing
    print("\nToxic Sample Predictions:\n")

    toxic_mask = y_val.sum(axis=1) > 0

    X_toxic = X_val[toxic_mask]
    y_toxic = y_val[toxic_mask]

    if len(X_toxic) > 0:
        X_toxic_vec = vectorizer.transform(X_toxic)
        y_proba_toxic = model.predict_proba(X_toxic_vec)

        for i in range(min(5, len(X_toxic))):
            probs = {
                label: round(y_proba_toxic[i][j][1], 3)
                for j, label in enumerate(LABEL_COLS)
            }

            pred = [
                1 if probs[label] >= THRESHOLDS[label] else 0
                for label in LABEL_COLS
            ]

            pred_dict = dict(zip(LABEL_COLS, pred))
            actual_dict = dict(zip(LABEL_COLS, y_toxic.iloc[i]))

            severity = compute_severity(pred_dict)
            action = moderation_action(severity)

            print(f"Toxic Sample {i + 1}")
            print("Text:", X_toxic.iloc[i])
            print("Actual:", actual_dict)
            print("Probabilities:", probs)
            print("Predicted:", pred_dict)
            print("Severity score:", severity)
            print("Moderation action:", action)
            print("-" * 60)
    else:
        print("No toxic samples found in validation set.")


if __name__ == "__main__":
    evaluate_model()