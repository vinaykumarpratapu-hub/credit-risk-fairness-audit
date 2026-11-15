"""
Credit Risk Model + Fairness Audit — training script.

Dataset: Statlog German Credit Data (UCI), sourced via GitHub mirror
(DataProfunda/German-Credit-Analysis), 1000 applicants, 20 features + binary target
(Creditability: 1 = good credit, 0 = bad credit, in the raw encoding Good=1/Bad=0 per
this mirror's convention — note original Statlog is Good=1/Bad=2; this mirror remaps to 0/1).

Models: Logistic Regression (baseline) vs XGBoost (main model), both evaluated with
5-fold cross-validated ROC-AUC. No deep learning model — deliberately out of scope for
a few hundred rows of structured tabular data.

Fairness audit: Fairlearn MetricFrame across three protected attributes derived from the
raw columns — sex (from "Sex & Marital Status"), age band (<25 vs 25+), and foreign-worker
status.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

DATA_PATH = "data/ger_creditability.csv"
MODELS_DIR = "models"

CATEGORICAL = [
    "Account Balance", "Payment Status of Previous Credit", "Purpose",
    "Value Savings/Stocks", "Length of current employment", "Sex & Marital Status",
    "Guarantors", "Most valuable available asset", "Concurrent Credits",
    "Type of apartment", "Occupation", "Telephone", "Foreign Worker",
]
NUMERIC = [
    "Duration of Credit (month)", "Credit Amount", "Instalment per cent",
    "Duration in Current address", "Age (years)", "No of Credits at this Bank",
    "No of dependents",
]
TARGET = "Creditability"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def build_pipeline(model):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def main():
    df = load_data()
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logreg = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    xgb = build_pipeline(
        XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logreg_cv = cross_val_score(logreg, X_train, y_train, cv=cv, scoring="roc_auc")
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="roc_auc")

    logreg.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    results = {}
    for name, pipe, cv_scores in [("logreg", logreg, logreg_cv), ("xgboost", xgb, xgb_cv)]:
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        results[name] = {
            "f1": round(float(f1_score(y_test, pred)), 3),
            "roc_auc_single_split": round(float(roc_auc_score(y_test, proba)), 3),
            "roc_auc_cv_mean": round(float(cv_scores.mean()), 3),
            "roc_auc_cv_std": round(float(cv_scores.std()), 3),
        }

    print(json.dumps(results, indent=2))

    # --- Fairness audit (on the XGBoost model, the selected model) ---
    sex_raw = df.loc[X_test.index, "Sex & Marital Status"]
    # Statlog encoding: 1=male divorced/separated, 2=female divorced/separated/married,
    # 3=male single, 4=male married/widowed, 5=female single -> map to sex
    sex = sex_raw.map({1: "male", 2: "female", 3: "male", 4: "male", 5: "female"})
    age_band = df.loc[X_test.index, "Age (years)"].apply(lambda a: "under_25" if a < 25 else "25_plus")
    foreign = df.loc[X_test.index, "Foreign Worker"].map({1: "yes", 2: "no"})

    xgb_pred = xgb.predict(X_test)

    fairness = {}
    for attr_name, attr in [("sex", sex), ("age_band", age_band), ("foreign_worker", foreign)]:
        mf = MetricFrame(
            metrics={"selection_rate": lambda yt, yp: np.mean(yp)},
            y_true=y_test, y_pred=xgb_pred, sensitive_features=attr,
        )
        dpd = demographic_parity_difference(y_test, xgb_pred, sensitive_features=attr)
        eod = equalized_odds_difference(y_test, xgb_pred, sensitive_features=attr)
        group_counts = attr.value_counts().to_dict()
        fairness[attr_name] = {
            "by_group_selection_rate": mf.by_group.to_dict()["selection_rate"],
            "group_counts": group_counts,
            "demographic_parity_difference": round(float(dpd), 3),
            "equalized_odds_difference": round(float(eod), 3),
            "small_sample_warning": min(group_counts.values()) < 10,
        }

    print(json.dumps(fairness, indent=2))

    joblib.dump(xgb, f"{MODELS_DIR}/xgb_credit_model.joblib")
    joblib.dump(logreg, f"{MODELS_DIR}/logreg_credit_model.joblib")
    with open(f"{MODELS_DIR}/results.json", "w") as f:
        json.dump({"model_comparison": results, "fairness_audit": fairness}, f, indent=2)
    with open(f"{MODELS_DIR}/feature_schema.json", "w") as f:
        json.dump({"numeric": NUMERIC, "categorical": CATEGORICAL, "target": TARGET}, f, indent=2)


if __name__ == "__main__":
    main()
