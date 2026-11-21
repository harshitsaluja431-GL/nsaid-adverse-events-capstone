"""Prototype batch scoring of NSAID adverse event reports (QM 640 capstone).

Usage:
    python score_reports.py train  <clean.csv> <model.joblib> --until 2023 --calibrate-year 2024
    python score_reports.py score  <model.joblib> <new_reports.csv> <scored.csv>

`train` fits the random forest pipeline on reports up to --until, stores the
serious rate of the training years and of the calibration year, and saves both
with the pipeline. `score` applies the same preprocessing, recalibrates the
probabilities to the most recent serious rate (prior shift; Saerens et al., 2002),
assigns a priority band, and lists the three features that raised or lowered each
report's score the most (SHAP).
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age_years", "weight_kg"]
CATEGORICAL = ["species", "drug", "sex", "repro_status", "breed_group", "route", "reporter"]
FLAGS = ["rx_gi", "rx_renal", "rx_hepatic", "rx_neuro"]
BANDS = [(0.70, "High"), (0.45, "Medium"), (0.0, "Low")]


def add_breed_group(df, top_breeds):
    df = df.copy()
    df["breed_group"] = df["breed"].where(df["breed"].isin(top_breeds), "Other")
    return df


def build_pipeline():
    prep = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("flag", "passthrough", FLAGS),
    ])
    model = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    return Pipeline([("prep", prep), ("model", model)])


def prior_shift(p, old, new):
    odds = p / (1 - p) * (new / (1 - new)) / (old / (1 - old))
    return odds / (1 + odds)


def train(args):
    df = pd.read_csv(args.data).sort_values("report_id")
    year = pd.to_datetime(df["receive_date"]).dt.year
    tr, cal = df[year <= args.until], df[year == args.calibrate_year]
    top_breeds = tr["breed"].value_counts().head(15).index.tolist()
    tr = add_breed_group(tr, top_breeds)
    pipe = build_pipeline().fit(tr[NUMERIC + CATEGORICAL + FLAGS], tr["serious"])
    bundle = {"pipeline": pipe, "top_breeds": top_breeds,
              "train_rate": float(tr["serious"].mean()),
              "recent_rate": float(cal["serious"].mean()) if len(cal) else float(tr["serious"].mean())}
    joblib.dump(bundle, args.model)
    print(f"Trained on {len(tr)} reports; serious rate {bundle['train_rate']:.3f} -> "
          f"recalibrated to {bundle['recent_rate']:.3f} ({args.calibrate_year})")


def score(args):
    bundle = joblib.load(args.model)
    pipe = bundle["pipeline"]
    df = add_breed_group(pd.read_csv(args.reports), bundle["top_breeds"])
    X = df[NUMERIC + CATEGORICAL + FLAGS]
    p_raw = pipe.predict_proba(X)[:, 1]
    p = prior_shift(p_raw, bundle["train_rate"], bundle["recent_rate"])

    # SHAP: sum one-hot columns back to their source variable, keep the top three
    Xt = pipe.named_steps["prep"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    sv = sv[:, :, 1] if sv.ndim == 3 else sv
    names = pipe.named_steps["prep"].get_feature_names_out()

    def label(n):
        group, col = n.split("__", 1)
        if group == "num" and col.startswith("missingindicator_"):
            return col.replace("missingindicator_", "") + " missing"
        return col

    labels = np.array([label(n) for n in names])
    reasons = []
    for i in range(len(X)):
        row = sv[i]
        idx = np.argsort(-np.abs(row))[:3]
        reasons.append("; ".join(f"{labels[j]} ({'+' if row[j] > 0 else '-'}{abs(row[j]):.2f})" for j in idx))

    out = df[["report_id", "receive_date", "species", "drug"]].copy()
    out["p_serious"] = p.round(3)
    out["priority"] = [next(b for t, b in BANDS if v >= t) for v in p]
    out["top_reasons"] = reasons
    if "serious" in df:                      # present only when validating on labelled data
        out["serious_actual"] = df["serious"].values
    out.sort_values("p_serious", ascending=False).to_csv(args.out, index=False)
    print(out["priority"].value_counts().reindex(["High", "Medium", "Low"]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("data"); t.add_argument("model")
    t.add_argument("--until", type=int, default=2023)
    t.add_argument("--calibrate-year", type=int, default=2024)
    s = sub.add_parser("score")
    s.add_argument("model"); s.add_argument("reports"); s.add_argument("out")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else score(a)
