# Credit Risk Model + Fairness Audit (Portfolio Project)

Fifth project in an AI/ML portfolio series built toward AI Product Manager roles in
banking/fintech, alongside iPhone Review Sentiment Analysis (v1/v2), Fraud Detection +
Explainability, and LoanForge. Not connected to any employer's product or real customer
data — built on the public Statlog German Credit Data (UCI).

**Live demo:** _add your deployed Streamlit Cloud URL here after deploying_
**GitHub:** this repo

## What this project demonstrates

Most credit-risk portfolio projects stop at "a model scored 0.8 AUC." This one adds the
piece that actually matters once a model like this is proposed for real use: **does it
treat people fairly, and do I actually know that — or did I just not check?**

## What was built

- **Model comparison:** Logistic Regression (baseline) vs. XGBoost (selected model), both
  evaluated with 5-fold cross-validated ROC-AUC rather than a single train/test split, so
  the "XGBoost edges out the baseline" claim is defensible rather than eyeballed.
- **Deliberately no deep learning model.** A PyTorch MLP was built and compared during
  development; it came out statistically tied with Logistic Regression and XGBoost
  (overlapping cross-validation error bars on ~750 training rows). Rather than keep it in
  as a "we tried deep learning too" talking point, it was removed entirely — forcing a
  technique into a project where it doesn't belong isn't a stronger portfolio piece, it's
  a weaker one. (The deep-learning showcase in this portfolio lives in the OCR/computer-
  vision work in LoanForge, where it's actually the right tool.)
- **Fairness audit:** Fairlearn `MetricFrame`, demographic parity difference, and
  equalized odds difference across three protected attributes — sex, age band (under 25
  vs. 25+), and foreign-worker status.
- **Streamlit app**, three tabs:
  1. *Score an Applicant* — fill in the Statlog schema, get a prediction, SHAP feature
     attributions, and (with `ANTHROPIC_API_KEY` set) a plain-English explanation from
     Claude.
  2. *Model Comparison* — Logistic Regression vs. XGBoost only, with an explicit note on
     why no deep learning model is shown.
  3. *Fairness Dashboard* — per-group approval rates and both fairness-gap metrics for
     each protected attribute, with small-sample warnings surfaced rather than hidden.

## Results

| Model | F1 | ROC-AUC (single split) | ROC-AUC (5-fold CV) |
|---|---|---|---|
| Logistic Regression | 0.801 | 0.809 | 0.756 ± 0.032 |
| XGBoost (selected) | 0.806 | 0.821 | 0.753 ± 0.016 |

Fairness: sex gap (70.1% male vs. 50.0% female approval, demographic parity diff. 0.201),
age gap (68.5% 25+ vs. 35.3% under-25, demographic parity diff. 0.332) — both real
findings worth investigating. Foreign-worker gap (83.3% vs. 63.5%) is flagged as
statistically unreliable (n=6 for the smaller group).

Full numbers (including equalized odds differences) are in `models/results.json`.
Re-running `train.py` reshuffles the train/test split, so figures move slightly run to
run — this is expected and is itself consistent with the "don't trust a single split"
point above.

**Fairness finding worth flagging honestly:** the foreign-worker protected group has very
few test examples. The app surfaces this as a small-sample warning rather than reporting
the raw gap uncritically — knowing when a fairness metric isn't statistically reliable is
itself an interview talking point, not a caveat to bury in a footnote.

## Tech stack

Python, scikit-learn (preprocessing pipeline + Logistic Regression), XGBoost, SHAP
(`TreeExplainer`), Fairlearn (`MetricFrame`, demographic parity / equalized odds),
Streamlit, Claude API for plain-English explanations.

## Why the dataset is public, not synthetic

Unlike this portfolio's other projects (which had to generate synthetic data from first
principles), this one uses the real, well-known Statlog German Credit Data — a standard
benchmark dataset for credit scoring research, fine to use for a non-commercial portfolio
project. Sourced via a GitHub mirror since direct access to the UCI archive wasn't
reachable from the build environment; verified against the standard Statlog column
encoding before use.

## Running locally

```bash
pip install -r requirements-dev.txt   # includes fairlearn, for retraining
python train.py                        # retrains both models + fairness audit
streamlit run app.py                   # requirements.txt is enough just to run the app
```

Set `ANTHROPIC_API_KEY` as an environment variable (or a Streamlit secret when deployed)
to enable the plain-English explanation in the "Score an Applicant" tab.

## Interview talking points

- **Why Logistic Regression vs. XGBoost only, with cross-validation** — a defensible
  model comparison needs more than one split; CV error bars are what make "XGBoost wins"
  a claim rather than an impression.
- **Why deep learning was tested and then deliberately removed** — knowing when *not* to
  reach for a technique is as much a product/technical judgment call as knowing when to
  use one. Including a losing DL comparison just to show "we considered it" would have
  been padding, not insight.
- **Why a fairness audit at all** — a credit model this accurate is exactly the kind of
  model that needs a fairness check before anyone proposes using it for real decisions;
  skipping that step is itself a choice, and a risky one in a regulated industry.
- **Why the foreign-worker gap is flagged as unreliable rather than reported as fact** —
  a six-person test group isn't enough to support a claim about whether a 20-point
  approval-rate gap is real discrimination or noise. Saying so plainly, instead of
  reporting the dramatic number uncritically, is the actual skill being demonstrated.
