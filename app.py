import os, pickle
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from features import hand_features, strip_code, FEATURE_NAMES

st.set_page_config(page_title="Big-O Classifier", page_icon="⚡", layout="centered")

MODEL_DIR = os.path.dirname(__file__)   # pkls sit next to this file

BIGO = {
    "constant": "O(1)", "logn": "O(log n)", "linear": "O(n)",
    "nlogn": "O(n log n)", "quadratic": "O(n²)",
    "cubic": "O(n³)", "np": "O(2ⁿ) — exponential",
}

@st.cache_resource
def load_models():
    m  = pickle.load(open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb"))
    tf = pickle.load(open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb"))
    le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
    return m, tf, le

model, tfidf, le = load_models()

st.title("⚡ Big-O Complexity Classifier")
st.caption("Paste a Java solution — the model predicts its time complexity. "
           "XGBoost on 14 hand-engineered code features + TF-IDF over tokens. "
           "Trained on the CodeComplex dataset (Java).")

SAMPLE = """public class Main {
    public static void main(String[] args) {
        int[] a = {5, 2, 8, 1, 9, 3};
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < a.length - 1; j++)
                if (a[j] > a[j + 1]) {
                    int t = a[j]; a[j] = a[j + 1]; a[j + 1] = t;
                }
    }
}"""

code = st.text_area("Java code", value=SAMPLE, height=280)

if st.button("Predict complexity", type="primary"):
    if not code.strip():
        st.warning("Paste some Java code first.")
    else:
        hand = np.array([hand_features(code)], dtype=float)
        X = hstack([csr_matrix(hand), tfidf.transform([strip_code(code)])]).tocsr()
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
            st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES,
                                       "value": hand_features(code)}),
                         hide_index=True, use_container_width=True)

st.divider()
st.caption("Trained on the CodeComplex dataset (Java). Note: evaluated with a "
           "problem-level split so no problem's solutions appear in both train and test.")