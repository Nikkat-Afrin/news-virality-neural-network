# News Article Virality Prediction — Neural Network 🗞️🧠

**Predict how widely an online article will be shared from its content, metadata, and publication features — using a feed-forward neural network on ~39,600 articles.**

![Python](https://img.shields.io/badge/Python-3.12-blue) ![DL](https://img.shields.io/badge/TensorFlow%2FKeras-Neural%20Network-FF6F00?logo=tensorflow&logoColor=white)

---

## 💼 Problem
Publishers want to know *before* publishing which articles are likely to go viral, to guide promotion and editorial decisions. This project frames virality as a **classification** problem and trains a neural network to predict an article's share-popularity tier from features available at publication time.

## 📊 Dataset
**UCI Online News Popularity** — `data/news_articles.csv` (bundled): **~39,600 articles × 61 features** including token counts, keyword metrics, channel/topic indicators, publication day, sentiment/subjectivity (NLP-derived), and reference counts. The target is derived from `shares` (binned into popularity classes).

## 🔬 Methodology
1. **Preprocessing** — cleaning, scaling (`StandardScaler`), and class construction from `shares`.
2. **Dimensionality reduction / feature selection** — to tame 61 correlated features (forward selection via `mlxtend`, plus a companion PCA/feature-selection assignment).
3. **Modeling** — a **feed-forward neural network (TensorFlow/Keras)** with a tunable architecture (hidden layers / neurons / activation / learning rate), softmax output, compared against a logistic-regression baseline.
4. **Evaluation** — accuracy, classification report, and confusion matrix.

## 📈 Status & results
- The notebook ([`notebooks/news_virality_nn.ipynb`](notebooks/news_virality_nn.ipynb)) builds and trains the network end-to-end on TensorFlow 2.17 / Keras 3.
- ⚙️ **Compute note:** the architecture-search portion trains many candidate networks and is **CPU-intensive** (allow time, or reduce the search grid / epochs for a quick pass). It runs to completion in an interactive Jupyter session.

## ▶️ How to run
```bash
pip install -r requirements.txt
jupyter lab notebooks/news_virality_nn.ipynb
```

## 🛠️ Tech stack
`Python` · `TensorFlow` · `Keras` · `scikit-learn` · `mlxtend` · `pandas` · `Matplotlib` · `Seaborn`

## 🚀 Future improvements
- Early stopping + `KerasTuner` for efficient architecture search; GPU acceleration.
- Compare against gradient-boosted trees (often strong on this tabular dataset) and add SHAP for feature attribution.

---
*Academic project (DAV 6150, M13 — Neural Networks; with M4 feature-selection groundwork). Data: UCI Online News Popularity.*
