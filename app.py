import os, pickle, joblib
import numpy as np
import pandas as pd
import streamlit as st
from features import hand_features, extra_features, FEATURE_NAMES, EXTRA_NAMES

st.set_page_config(page_title="Big-O Classifier", layout="centered")

MODEL_DIR = os.path.dirname(__file__)   

BIGO = {
    "constant": "O(1)", "logn": "O(log n)", "linear": "O(n)",
    "nlogn": "O(n log n)", "quadratic": "O(n²)",
    "cubic": "O(n³)", "np": "O(2ⁿ) — exponential",
}

@st.cache_resource
def load_models():
    m  = joblib.load(os.path.join(MODEL_DIR, "et_model.pkl"))
    le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    return m, le

model, le = load_models()

st.title("Big-O Complexity Classifier")
st.caption("Paste a Java solution — the model predicts its time complexity. "
           "Extra Trees on 21 hand-engineered code features. "
           "Trained on the CodeComplex dataset (Java).")

SAMPLE = """ public class Main {
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

        // print the result matrix
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(c[i][j] + " ");
            }
            System.out.println();
        }
    }
}"""

code = st.text_area("Java code", value=SAMPLE, height=280)

if st.button("Predict complexity", type="primary"):
    if not code.strip():
        st.warning("Paste some Java code first.")
    else:
        feats = hand_features(code) + extra_features(code)
        X = np.array([feats], dtype=float)
        proba = model.predict_proba(X)[0]
        i = int(proba.argmax())
        cls = le.classes_[i]

        st.markdown(f"## Prediction: {BIGO[cls]}")
        st.markdown(f"label `{cls}` · confidence **{proba[i]*100:.1f}%**")

        st.subheader("Confidence per class")
        for j in proba.argsort()[::-1]:
            c = le.classes_[j]
            st.progress(float(proba[j]), text=f"{BIGO[c]}  ({c}) — {proba[j]*100:.1f}%")

        with st.expander("Extracted hand features (why the model decided this)"):
            st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES + EXTRA_NAMES,
                                       "value": feats}),
                         hide_index=True, use_container_width=True)

st.divider()
st.caption("Trained on the CodeComplex dataset (Java).")
