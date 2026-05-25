

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 5)


def load_data(path="stockx_data.csv"):
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} rows | Columns: {list(df.columns)}")
    return df



def clean_data(df):
    df = df.copy()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["order_date"]   = pd.to_datetime(df["order_date"],   errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

    
    for col in ["sale_price", "retail_price"]:
        df[col] = (df[col].astype(str)
                          .str.replace(r"[$,]", "", regex=True)
                          .astype(float))

    
    df.dropna(subset=["sale_price", "retail_price", "release_date", "order_date"], inplace=True)

    print(f"✅ After cleaning: {len(df):,} rows")
    return df




def engineer_features(df):
    df = df.copy()

    
    df["resale_premium_pct"] = ((df["sale_price"] - df["retail_price"]) / df["retail_price"]) * 100

   
    df["days_since_release"] = (df["order_date"] - df["release_date"]).dt.days

   
    df["sale_month"]  = df["order_date"].dt.month
    df["sale_season"] = df["sale_month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring",  4: "Spring", 5: "Spring",
        6: "Summer",  7: "Summer", 8: "Summer",
        9: "Fall",   10: "Fall",  11: "Fall"
    })

    df["shoe_size"] = pd.to_numeric(df["shoe_size"], errors="coerce")
    df["size_demand_index"] = df["shoe_size"].apply(
        lambda s: 1.0 if 9 <= s <= 11 else (0.7 if 7 <= s < 9 or 11 < s <= 13 else 0.4)
    )

    
    df["brand"] = df["brand"].str.strip()

    
    df["is_limited"] = (df["retail_price"] > 150).astype(int)

    
    df = df[df["resale_premium_pct"].between(-50, 1000)]

    print(f"✅ Features engineered | Target range: {df['resale_premium_pct'].min():.1f}% to {df['resale_premium_pct'].max():.1f}%")
    return df



def run_eda(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sneaker Resale Market — EDA", fontsize=16, fontweight="bold")

    
    sns.histplot(df["resale_premium_pct"], bins=60, kde=True, ax=axes[0,0], color="#FF6B35")
    axes[0,0].set_title("Resale Premium Distribution (%)")
    axes[0,0].set_xlabel("Premium (%)")

   
    brand_premium = df.groupby("brand")["resale_premium_pct"].median().sort_values(ascending=False)
    sns.barplot(x=brand_premium.values, y=brand_premium.index, ax=axes[0,1], palette="viridis")
    axes[0,1].set_title("Median Resale Premium by Brand")
    axes[0,1].set_xlabel("Median Premium (%)")

    
    df["days_bin"] = pd.cut(df["days_since_release"], bins=[0,30,90,180,365,730,9999],
                            labels=["<1mo","1-3mo","3-6mo","6-12mo","1-2yr","2yr+"])
    decay = df.groupby("days_bin")["resale_premium_pct"].median().reset_index()
    sns.lineplot(data=decay, x="days_bin", y="resale_premium_pct", marker="o", ax=axes[0,2], color="#4ECDC4")
    axes[0,2].set_title("Resale Premium Decay Over Time")
    axes[0,2].set_xlabel("Time Since Release")
    axes[0,2].set_ylabel("Median Premium (%)")

   
    top_shoes = (df.groupby("sneaker_name")["resale_premium_pct"]
                   .median()
                   .sort_values(ascending=False)
                   .head(10))
    sns.barplot(x=top_shoes.values, y=top_shoes.index, ax=axes[1,0], palette="rocket")
    axes[1,0].set_title("Top 10 Silhouettes by Median Premium")

    
    region_avg = df.groupby("buyer_region")["resale_premium_pct"].median().sort_values(ascending=False).head(10)
    sns.barplot(x=region_avg.values, y=region_avg.index, ax=axes[1,1], palette="mako")
    axes[1,1].set_title("Top 10 Buyer Regions by Premium")

    sample = df.sample(min(2000, len(df)), random_state=42)
    axes[1,2].scatter(sample["retail_price"], sample["sale_price"], alpha=0.3, c="#FF6B35", s=10)
    axes[1,2].plot([0, 1000], [0, 1000], "k--", linewidth=1, label="Retail = Resale")
    axes[1,2].set_title("Sale Price vs Retail Price")
    axes[1,2].set_xlabel("Retail Price ($)")
    axes[1,2].set_ylabel("Sale Price ($)")
    axes[1,2].legend()

    plt.tight_layout()
    plt.savefig("visuals/eda_overview.png", dpi=150)
    plt.show()
    print("✅ EDA saved to visuals/eda_overview.png")




def prepare_ml_features(df):
    df_ml = df.copy()

    for col in ["sneaker_name", "brand", "buyer_region"]:
        means = df_ml.groupby(col)["resale_premium_pct"].mean()
        df_ml[f"{col}_encoded"] = df_ml[col].map(means)

   
    season_order = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
    df_ml["sale_season_enc"] = df_ml["sale_season"].map(season_order)

    features = [
        "retail_price",
        "days_since_release",
        "shoe_size",
        "size_demand_index",
        "is_limited",
        "sale_month",
        "sale_season_enc",
        "brand_encoded",
        "buyer_region_encoded",
        "sneaker_name_encoded",  
    ]

    df_ml = df_ml[features + ["resale_premium_pct"]].dropna()
    X = df_ml[features]
    y = df_ml["resale_premium_pct"]

    print(f"✅ ML dataset: {X.shape[0]:,} samples | {X.shape[1]} features")
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=1),
        "XGBoost":           XGBRegressor(n_estimators=100, learning_rate=0.1,
                                          max_depth=4, random_state=42, verbosity=0),
    }

    print("\n📊 Model Comparison")
    print(f"{'Model':<22} {'RMSE':>10} {'R²':>10} {'CV R² (mean)':>15}")
    print("─" * 60)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        cv    = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
        results[name] = {"model": model, "rmse": rmse, "r2": r2, "cv_r2": cv}
        print(f"{name:<22} {rmse:>10.2f} {r2:>10.4f} {cv:>15.4f}")

  
    best_name = max(results, key=lambda k: results[k]["cv_r2"])
    best_model = results[best_name]["model"]
    print(f"\n🏆 Best model: {best_name} (CV R² = {results[best_name]['cv_r2']:.4f})")

    return best_model, X_test, y_test, results





def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="barh", ax=ax, color="#FF6B35")
        ax.set_title("Feature Importance — What Drives Resale Premium?", fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("visuals/feature_importance.png", dpi=150)
        plt.show()
        print("✅ Feature importance saved to visuals/feature_importance.png")




def save_model(model, path="models/best_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {path}")




def predict_sneaker(model):
    example = pd.DataFrame([{
        "retail_price":           160,
        "days_since_release":     45,
        "shoe_size":              10,
        "size_demand_index":      1.0,
        "is_limited":             1,
        "sale_month":             12,
        "sale_season_enc":        0,
        "brand_encoded":          270.0,
        "buyer_region_encoded":   80.0,
        "sneaker_name_encoded":   400.0,
    }])

    predicted_premium = model.predict(example)[0]
    print(f"\n Predicted resale premium: {predicted_premium:.1f}%")
    print("   (i.e. this sneaker is predicted to sell {:.0f}% above retail)".format(predicted_premium))




if __name__ == "__main__":
    print("=" * 60)
    print("  SNEAKER RESALE PRICE PREDICTOR")
    print("  Urban Cage x Data Science Portfolio")
    print("=" * 60)

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    run_eda(df)

    X, y = prepare_ml_features(df)
    best_model, X_test, y_test, results = train_models(X, y)

    plot_feature_importance(best_model, X.columns.tolist())
    save_model(best_model)
    predict_sneaker(best_model)

    print("\n✅ Pipeline complete.")
