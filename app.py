import streamlit as st
import joblib

from src.moderation_rules import compute_severity, moderation_action

# LABELS 
LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# THRESHOLDS 
THRESHOLDS = {
    "toxic": 0.5,
    "severe_toxic": 0.5,
    "obscene": 0.5,
    "threat": 0.5,
    "insult": 0.5,
    "identity_hate": 0.5
}

# LOAD MODEL 
@st.cache_resource
def load_model():
    model = joblib.load("models/toxicity_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# UI CONFIG 
st.set_page_config(page_title="Toxic Comment Filter", layout="centered")

st.title("Toxic Comment Detection System")
st.write("Enter a comment below to analyze toxicity and moderation action.")

# INPUT 
user_input = st.text_area("Enter Comment", height=150)

# BUTTON
if st.button("Analyze Comment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        # VECTORIZE
        X = vectorizer.transform([user_input])

        # PROBABILITIES 
        y_proba = model.predict_proba(X)

        probs = {
            label: float(y_proba[0][i])
            for i, label in enumerate(LABEL_COLS)
        }

        # APPLY THRESHOLDS 
        predictions = {
            label: 1 if probs[label] >= THRESHOLDS[label] else 0
            for label in LABEL_COLS
        }

        # MODERATION LOGIC 
        severity = compute_severity(predictions)
        action = moderation_action(severity)

        # UI DISPLAY

        st.markdown("## Toxicity Analysis Dashboard")

        col1, col2 = st.columns([2, 1])

        # LEFT: PROBABILITIES 
        with col1:
            st.markdown("### Toxicity Breakdown")

            for label in LABEL_COLS:
                value = probs[label]
                st.write(f"**{label.replace('_',' ').title()}** ({value:.3f})")
                st.progress(float(value))

        # RIGHT: DECISION
        with col2:
            st.markdown("### Moderation Decision")

            st.metric("Severity Score", severity)

            if action == "Allow":
                st.success("Allow")
            elif action == "Warn":
                st.warning("Warn User")
            else:
                st.error("Auto Ban")

        # OPTIONAL DETAILS 
        st.markdown("---")
        with st.expander("View Detailed Scores"):
            st.json(probs)