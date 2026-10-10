# Big-O Complexity Classifier

A machine learning application that predicts the **time complexity of Java programs** from their source code. It combines handcrafted static code features with TF-IDF representations and an XGBoost classifier to predict one of seven complexity classes.

**Repository:** [github.com/raunaktiwari1729/BIGO](https://github.com/raunaktiwari1729/BIGO)

**Live Demo:** [Try the application](https://phi5qpbnnnkhhxryh8y9hv.streamlit.app/)

## Overview

Estimating the time complexity of a program is an important part of algorithm analysis. This project explores how machine learning can identify code patterns associated with different asymptotic complexities.

The classifier supports seven classes:

- `O(1)` — Constant
- `O(log n)` — Logarithmic
- `O(n)` — Linear
- `O(n log n)` — Linearithmic
- `O(n²)` — Quadratic
- `O(n³)` — Cubic
- `O(2ⁿ)` — Exponential

## How It Works

The deployed pipeline combines two types of features extracted from Java source code.

### 1. Handcrafted Static Code Features

The application extracts 21 numerical features, including:

- Loop counts and estimated nesting depth
- Sorting operations and hash-based data structures
- Recursion indicators
- Binary-search-related patterns and bit shifts
- Method counts, conditionals, statements, and variable declarations

### 2. TF-IDF Feature Extraction

A fitted TF-IDF vectorizer converts source-code tokens into a numerical representation, capturing patterns that may not be fully represented by handcrafted features.

### 3. XGBoost Classification

The handcrafted features and TF-IDF representation are combined and passed to a trained XGBoost classifier, which predicts one of the seven complexity classes.

## Model Evaluation

The project uses the CodeComplex Java dataset, containing 4,900 solutions from 258 problems.

| Evaluation Setup | Accuracy |
|---|---:|
| Random train-test split | 82.9% |
| Problem-grouped evaluation | 58.0% |

The random-split result can be optimistic because solutions to the same problem may appear in both training and testing sets. Problem-grouped evaluation separates solutions by problem, providing a more informative assessment of generalization to unseen problems.

### Classical Model Comparison

Ten classical machine learning models were evaluated using the same 21 handcrafted features without TF-IDF, with problem-level five-fold evaluation.

| Model | Accuracy |
|---|---:|
| Gradient Boosting | 56.1% |
| Extra Trees | 56.0% |
| XGBoost | 55.9% |
| Random Forest | 55.9% |
| SVM (RBF) | 54.1% |
| Decision Tree | 53.6% |
| Logistic Regression | 52.9% |
| MLP | 52.9% |
| k-Nearest Neighbors | 52.3% |
| Naive Bayes | 32.7% |

These results suggest that handcrafted features provide a useful baseline, while the algorithm choice alone makes relatively little difference among the strongest classical models.

### Additional Experiments

**AST-based Features**

The project also explores Abstract Syntax Tree (AST) features using Java parsing. These features capture structural information such as loop nesting, recursion, and patterns within loop bodies. The recorded experiments show modest improvements over handcrafted features alone.

**GraphCodeBERT**

`Transformer.ipynb` explores fine-tuning `microsoft/graphcodebert-base` for seven-class complexity classification. Its final accuracy remains to be recorded after completing the experiment.

## Tech Stack

- **Language:** Python
- **Machine Learning:** XGBoost, scikit-learn
- **Feature Engineering:** Regular expressions, handcrafted static code features
- **Text Representation:** TF-IDF
- **AST Analysis:** javalang
- **Application:** Streamlit
- **Data Processing:** NumPy, pandas, SciPy
- **Transformer Experiment:** Hugging Face Transformers, GraphCodeBERT

## Project Structure

```text
BIGO/
├── app.py
├── features.py
├── ast_features.py
├── xgb_model.pkl
├── tfidf.pkl
├── label_encoder.pkl
├── model_comparison.csv
├── single_feature_accuracy.csv
├── ast_comparison.csv
├── model_comparison.png
├── confusion_matrix.png
├── feature_importance.png
└── README.md
```

### Links for training notebooks(ipynb)

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/raunaktiwari1729/BIGO.git
cd BIGO
```

### 2. Install Dependencies

```bash
pip install streamlit numpy pandas scipy scikit-learn xgboost
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, paste Java code into the input area, and select **Predict complexity**.

## Limitations

- Regular-expression-based features approximate code structure and may misinterpret some Java constructs.
- Programs with similar surface-level patterns can have different actual time complexities.
- Problem-grouped accuracy is considerably lower than random-split accuracy, indicating limited generalization to unseen problems.
- The application predicts learned complexity labels; it does not perform a formal proof of algorithmic time complexity.
- Model performance depends on the dataset, feature representation, and distribution of problems encountered during training.

## Future Improvements

- Improve Java parsing and structural feature extraction.
- Investigate class imbalance and errors between similar complexity classes.
- Evaluate additional source-code representations and transformer-based models.
- Improve generalization using larger and more diverse datasets.
- Make training and evaluation fully reproducible.

## Acknowledgements

This project uses the CodeComplex dataset for Java algorithm-complexity classification. The additional handcrafted features were inspired in part by the CoRCoD paper (Sikka et al., 2019).
