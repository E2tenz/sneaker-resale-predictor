# 👟 Sneaker Resale Price Prediction
**A machine learning project to predict sneaker resale premiums using StockX sales data**

> Built as part of a data science portfolio — with real domain knowledge from running an active resale operation StockX.

---

## 🎯 Objective
Predict the **resale premium** of a sneaker (how much above retail it will sell for) based on brand, silhouette, release timing, region, and size demand.

---

## 📁 Project Structure
```
sneaker-resale-predictor/
│
├── data/
│   └── stockx_data.csv          # Raw dataset (download from Kaggle — link below)
│
├── notebooks/
│   └── sneaker_analysis.ipynb   # Main Jupyter notebook (mirrors analysis.py)
│
├── models/
│   └── best_model.pkl           # Saved best-performing model
│
├── visuals/
│   └── *.png                    # Exported charts for portfolio/README
│
├── app/
│   └── streamlit_app.py         # (Optional) Interactive prediction web app
│
├── analysis.py                  # Full pipeline script
└── README.md
```

---

## 📦 Dataset
**StockX Data Challenge 2019** — publicly available on Kaggle.
- Link: https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest
- ~99,000 sneaker transactions
- Columns: `Order Date`, `Brand`, `Sneaker Name`, `Sale Price ($)`, `Retail Price ($)`, `Release Date`, `Shoe Size`, `Buyer Region`

---

## 🔬 Methodology
1. **EDA** — distributions, brand breakdowns, price trends over time
2. **Feature Engineering** — resale premium, days since release, size demand index, season
3. **Modelling** — Linear Regression → Random Forest → XGBoost (with GridSearchCV tuning)
4. **Evaluation** — RMSE, R², feature importance
5. **Insights** — What actually drives resale value?

---

## 🛠 Stack
- Python 3.10+
- pandas, numpy
- scikit-learn, xgboost
- matplotlib, seaborn
- (Optional) Streamlit for deployment

---

## 📊 Key Questions Answered
- Which brands command the highest resale premiums?
- Does resale value decay over time after release?
- Which shoe sizes are most in-demand (and most profitable)?
- Does buyer region affect resale price?

---

## 🚀 How to Run
```bash
pip install -r requirements.txt
python analysis.py
# or open notebooks/sneaker_analysis.ipynb in Jupyter
```
