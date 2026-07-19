# ⚡ Big-O Complexity Classifier

Predict the **time complexity** (Big-O) of a Java solution straight from its source code.

🔗 **Live demo:** https://phi5qpbnnnkhhxryh8y9hv.streamlit.app/

Paste any Java code → get its predicted complexity, per-class confidence, and the features the model used.

**7 classes:** `O(1)` · `O(log n)` · `O(n)` · `O(n log n)` · `O(n²)` · `O(n³)` · `O(2ⁿ)`

---

## Key result

The honest accuracy is **~56%** 

A **random** train/test split scores **84%**, but it's misleading: multiple solutions to the same problem share a complexity, so a random split leaks and the model just memorizes problems. Splitting **by problem** — so no problem appears in both train and test — gives the real **~56%**.

On that honest split, a simple **XGBoost on 14 hand-crafted features** actually *beats* a fine-tuned **GraphCodeBERT** transformer:

| Model | Honest accuracy (problem split) |
|---|---|
| XGBoost + hand features | **56.4%** |
| GraphCodeBERT (fine-tuned) | 54.4% |

The hand features (loop nesting, sort, recursion) encode the signal directly; the transformer has to learn it from a small dataset and truncates long code at 512 tokens.

---

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (mac/linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Opens at `http://localhost:8501`.

Keep these files together: `app.py`, `features.py`, `requirements.txt`, and the models `xgb_model.pkl`, `tfidf.pkl`, `label_encoder.pkl`.

---

## How it works

- **Features** — 14 hand-engineered signals (loop-nesting depth, loop/sort counts, recursion, binary-search & log-loop hints, LOC) combined with TF-IDF over code tokens.
- **Model** — XGBoost, trained on the [CodeComplex](https://github.com/sybaik1/CodeComplex-Data) dataset (Codeforces Java solutions).
- `features.py` is imported by both training and the app, so features are computed identically at train and serve time.

## Notebooks

- `XGBoost.ipynb` — features, both splits, final model
- `Transformer.ipynb` — GraphCodeBERT fine-tune on the same split