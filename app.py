import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Credit Risk + Fairness Audit", page_icon="🏦", layout="wide")

MODELS_DIR = "models"


@st.cache_resource
def load_artifacts():
    xgb = joblib.load(f"{MODELS_DIR}/xgb_credit_model.joblib")
    logreg = joblib.load(f"{MODELS_DIR}/logreg_credit_model.joblib")
    with open(f"{MODELS_DIR}/results.json") as f:
        results = json.load(f)
    with open(f"{MODELS_DIR}/feature_schema.json") as f:
        schema = json.load(f)
    return xgb, logreg, results, schema


xgb_pipe, logreg_pipe, results, schema = load_artifacts()
NUMERIC = schema["numeric"]
CATEGORICAL = schema["categorical"]

CATEGORY_OPTIONS = {
    "Account Balance": {1: "No account", 2: "< 0 DM", 3: "0-200 DM", 4: ">= 200 DM"},
    "Payment Status of Previous Credit": {
        0: "Delay in past", 1: "Critical/other credits elsewhere", 2: "No credits taken",
        3: "Paid duly so far", 4: "All credits paid duly",
    },
    "Purpose": {
        0: "Other", 1: "New car", 2: "Used car", 3: "Furniture/equipment",
        4: "Radio/TV", 5: "Domestic appliances", 6: "Repairs", 8: "Vacation",
        9: "Retraining", 10: "Business",
    },
    "Value Savings/Stocks": {1: "< 100 DM", 2: "100-500 DM", 3: "500-1000 DM", 4: ">= 1000 DM", 5: "Unknown/none"},
    "Length of current employment": {1: "Unemployed", 2: "< 1 year", 3: "1-4 years", 4: "4-7 years", 5: ">= 7 years"},
    "Sex & Marital Status": {1: "Male: divorced/separated", 2: "Female: divorced/separated/married", 3: "Male: single", 4: "Male: married/widowed", 5: "Female: single"},
    "Guarantors": {1: "None", 2: "Co-applicant", 3: "Guarantor"},
    "Most valuable available asset": {1: "None", 2: "Car/other", 3: "Life insurance", 4: "Real estate"},
    "Concurrent Credits": {1: "Bank", 2: "Stores", 3: "None"},
    "Type of apartment": {1: "Free", 2: "Rented", 3: "Owned"},
    "Occupation": {1: "Unemployed/unskilled", 2: "Unskilled resident", 3: "Skilled employee", 4: "Management/self-employed"},
    "Telephone": {1: "None", 2: "Registered"},
    "Foreign Worker": {1: "Yes", 2: "No"},
}

tab1, tab2, tab3 = st.tabs(["📋 Score an Applicant", "📊 Model Comparison", "⚖️ Fairness Dashboard"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.header("Score a loan applicant")
    st.caption("Statlog German Credit Data schema · XGBoost model · SHAP + Claude explanation")

    with st.form("applicant_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(NUMERIC):
            with cols[i % 3]:
                default = {"Duration of Credit (month)": 18, "Credit Amount": 2000,
                           "Instalment per cent": 3, "Duration in Current address": 2,
                           "Age (years)": 30, "No of Credits at this Bank": 1, "No of dependents": 1}.get(feat, 0)
                inputs[feat] = st.number_input(feat, value=default, step=1)
        for i, feat in enumerate(CATEGORICAL):
            with cols[i % 3]:
                opts = CATEGORY_OPTIONS.get(feat, {})
                inputs[feat] = st.selectbox(feat, options=list(opts.keys()), format_func=lambda k, o=opts: o.get(k, k))
        submitted = st.form_submit_button("Score applicant")

    if submitted:
        row = pd.DataFrame([inputs])[NUMERIC + CATEGORICAL]
        proba = xgb_pipe.predict_proba(row)[0, 1]
        pred = "Good credit risk ✅" if proba >= 0.5 else "Bad credit risk ⚠️"
        st.metric("Prediction", pred, f"{proba:.1%} probability of good credit")

        explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])
        transformed = xgb_pipe.named_steps["pre"].transform(row)
        feature_names = xgb_pipe.named_steps["pre"].get_feature_names_out()
        shap_values = explainer.shap_values(transformed)
        contrib = pd.Series(np.array(shap_values).flatten(), index=feature_names).sort_values(key=abs, ascending=False)

        st.subheader("Top factors behind this prediction (SHAP)")
        st.dataframe(contrib.head(8).rename("SHAP contribution"))

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                top_factors = "\n".join(f"- {k}: {v:+.3f}" for k, v in contrib.head(5).items())
                msg = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=300,
                    messages=[{"role": "user", "content": (
                        f"A credit risk model scored a loan applicant as {pred} "
                        f"({proba:.1%} probability of good credit). The top SHAP feature "
                        f"contributions were:\n{top_factors}\n\n"
                        "In 3-4 plain-English sentences, explain this decision to a loan "
                        "officer who is not a data scientist. Be concrete about which "
                        "factors pushed the decision which way."
                    )}],
                )
                st.subheader("Plain-English explanation (Claude)")
                st.write(msg.content[0].text)
            except Exception as e:
                st.info(f"Claude explanation unavailable: {e}")
        else:
            st.info("Set ANTHROPIC_API_KEY as a secret to enable the plain-English explanation.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.header("Model comparison: Logistic Regression vs. XGBoost")
    st.caption(
        "No deep learning model is included here — deliberately. Credit risk on a few "
        "hundred rows of structured tabular data isn't a problem deep learning is suited "
        "to; a PyTorch MLP was tested during development and came out statistically tied "
        "with these two (overlapping cross-validation error bars), so it was removed "
        "rather than kept as padding."
    )
    comp_df = pd.DataFrame(results["model_comparison"]).T
    comp_df.columns = ["F1", "ROC-AUC (single split)", "ROC-AUC (5-fold CV mean)", "ROC-AUC (5-fold CV std)"]
    st.dataframe(comp_df, use_container_width=True)
    st.markdown(
        "**XGBoost** is the selected model — it edges out the Logistic Regression "
        "baseline on F1 and single-split ROC-AUC, and the 5-fold cross-validated ROC-AUC "
        "confirms the comparison isn't just an artifact of one lucky train/test split."
    )

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.header("Fairness audit (XGBoost model)")
    st.caption("Fairlearn MetricFrame · demographic parity & equalized odds differences across protected attributes")

    for attr_name, label in [("sex", "Sex"), ("age_band", "Age band (under 25 vs. 25+)"), ("foreign_worker", "Foreign worker status")]:
        fa = results["fairness_audit"][attr_name]
        st.subheader(label)
        rate_df = pd.DataFrame.from_dict(fa["by_group_selection_rate"], orient="index", columns=["Approval rate"])
        rate_df["Group size (test set)"] = pd.Series(fa["group_counts"])
        st.dataframe(rate_df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Demographic parity difference", f"{fa['demographic_parity_difference']:.3f}")
        c2.metric("Equalized odds difference", f"{fa['equalized_odds_difference']:.3f}")
        if fa["small_sample_warning"]:
            st.warning(
                f"⚠️ The smallest group here has fewer than 10 test examples "
                f"({min(fa['group_counts'].values())}). This gap is **too small a sample to "
                "trust** — reporting it uncritically would be misleading, even though the "
                "raw numbers look dramatic. Knowing when a fairness metric isn't reliable "
                "is itself the point here."
            )
        else:
            st.info(
                "This gap is based on a reasonably sized sample and is a real finding "
                "worth investigating further — not proof of illegal discrimination on "
                "its own, but a signal that the model's approval rates differ across "
                "this group and deserve a closer look before deployment."
            )
        st.divider()
