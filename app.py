
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from features import (
    EXTRA_NAMES,
    FEATURE_NAMES,
    extra_features,
    hand_features,
    strip_code,
)


# Configure the Streamlit page.
st.set_page_config(
    page_title="Big-O Classifier",
    layout="centered",
)

MODEL_DIR = os.path.dirname(__file__)

# Map the model's class labels to their Big-O representations.
BIGO = {
    "constant": "O(1)",
    "logn": "O(log n)",
    "linear": "O(n)",
    "nlogn": "O(n log n)",
    "quadratic": "O(n²)",
    "cubic": "O(n³)",
    "np": "O(2ⁿ) — exponential",
}


@st.cache_resource
def load_models():
    """Load the trained classifier, TF-IDF vectorizer, and label encoder."""
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as file:
        model = pickle.load(file)

    with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb") as file:
        tfidf = pickle.load(file)

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as file:
        label_encoder = pickle.load(file)

    return model, tfidf, label_encoder


model, tfidf, label_encoder = load_models()

st.title("Big-O Complexity Classifier")

st.caption(
    "Paste a Java solution — the model predicts its time complexity. "
    "XGBoost on 21 hand-engineered code features + TF-IDF over tokens. "
    "Trained on the CodeComplex dataset (Java)."
)

# Sample Java code shown when the application opens.
SAMPLE = """public class Main {
    public static void main(String[] args) {
        int n = 3;
        int[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[][] b = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
        int[][] c = new int[n][n];

        // O(n^3): three nested loops
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        // Print the result matrix.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(c[i][j] + " ");
            }
            System.out.println();
        }
    }
}"""

code = st.text_area(
    "Java code",
    value=SAMPLE,
    height=280,
)

if st.button("Predict complexity", type="primary"):
    if not code.strip():
        st.warning("Paste some Java code first.")
    else:
        # Extract the 21 handcrafted features.
        features = hand_features(code) + extra_features(code)
        hand = np.array([features], dtype=float)

        # Combine handcrafted features with TF-IDF token features.
        tfidf_features = tfidf.transform([strip_code(code)])
        X = hstack([csr_matrix(hand), tfidf_features]).tocsr()

        # Predict the complexity class and its estimated probability.
        probabilities = model.predict_proba(X)[0]
        best_index = int(probabilities.argmax())
        predicted_class = label_encoder.classes_[best_index]

        st.markdown(f"## Prediction: {BIGO[predicted_class]}")
        st.markdown(
            f"Label `{predicted_class}` · "
            f"confidence **{probabilities[best_index] * 100:.1f}%**"
        )

        # Display the scores for every complexity class.
        st.subheader("Confidence per class")

        for index in probabilities.argsort()[::-1]:
            class_label = label_encoder.classes_[index]
            probability = probabilities[index]

            st.progress(
                float(probability),
                text=(
                    f"{BIGO[class_label]} ({class_label}) — "
                    f"{probability * 100:.1f}%"
                ),
            )

        # Show the handcrafted features extracted from the input.
        with st.expander("Extracted hand features (why the model decided this)"):
            st.dataframe(
                pd.DataFrame({
                    "feature": FEATURE_NAMES + EXTRA_NAMES,
                    "value": features,
                }),
                hide_index=True,
                use_container_width=True,
            )

st.divider()

st.caption(
    "Trained on the CodeComplex dataset (Java). "
    "Note: evaluated with a problem-level split so no problem's "
    "solutions appear in both train and test."
)
