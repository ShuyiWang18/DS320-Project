"""
DS320 Final Project – PA Credit Approval with Multi-Source Fusion
Data source:
- PA HMDA.xlsx: Loan Level Sample + Tag (actiontaken)
- ACS.csv: Pennsylvania state-level summary statistics (converted into macro features, broadcast to each loan)
- PMMShistory.csv: Freddie Mac PMMS mortgage rate time series

Fusion ideas (feature-level fusion):
- HMDA: loanamount, income, interestrate, age, race, sex, loanpurpose, loantype
- + ACS: Add state-level ACS indicators as macro features (starting with acs) to each HMDA record
- + PMMS: closest match to PMMS date by actiondate, concatenating 30/15 annual rates

Model:
-Logistic Regression
- Random Forest

Rating:
- 60% / 20% / 20% adaptive split according to time
- ROC AUC, PR AUC
- Three solutions: HMDA / HMDA+ACS / HMDA+ACS+PMMS
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = "/Users/wangshuyi/Desktop"
DATA_DIR = f"{BASE_DIR}/data_raw"
REPORT_DIR = f"{BASE_DIR}/reports"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

HMDA_PATH = f"{DATA_DIR}/PA HMDA.xlsx"
ACS_PATH = f"{DATA_DIR}/ACS.csv"
PMMS_PATH = f"{DATA_DIR}/PMMS_history.csv"



def load_hmda(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    def label(x):
        s = str(x).strip().lower()
        if s in {"1", "2"} or "originated" in s or "approved" in s:
            return 1
        if s == "3" or "denied" in s:
            return 0
        return np.nan

    df["y"] = df["action_taken"].apply(label)

    df["action_date"] = pd.to_datetime(
        df["activity_year"].astype(str) + "-06-30", errors="coerce"
    )

    s = (
        df["census_tract"]
        .astype("string")
        .str.replace(r"[^0-9]", "", regex=True)
        .str.zfill(11)
    )
    s = s.where(s.str.fullmatch(r"\d{11}", na=False), np.nan)
    df["GEOID"] = s

    sc = df["state_code"].astype(str).str.upper()
    df = df[(sc == "PA") | (sc.str.zfill(2) == "42")].copy()

    mapping = {
        "loan_amount": "loan_amount",
        "income": "income",
        "interest_rate": "interest_rate",
        "applicant_age": "age",
        "derived_race": "race",
        "applicant_race-1": "race",
        "derived_sex": "sex",
        "applicant_sex": "sex",
        "loan_purpose": "loan_purpose",
        "loan_type": "loan_type",
    }
    for orig, new in mapping.items():
        if orig in df.columns:
            df[new] = df[orig]

    df = df.dropna(subset=["y", "action_date"]).copy()

    print(f"[HMDA] rows after PA filter: {len(df):,}")
    return df



def load_acs_macros(path: str) -> pd.DataFrame:
    acs_raw = pd.read_csv(path)

    if "Label (Grouping)" not in acs_raw.columns:
        raise ValueError("ACS.csv must contain 'Label (Grouping)' column.")

    est_col = "Pennsylvania!!Estimate"
    if est_col not in acs_raw.columns:
        raise ValueError("ACS.csv must contain 'Pennsylvania!!Estimate' column.")

    s = acs_raw[est_col].astype(str).str.replace(",", "", regex=False)
    nums = s.str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
    vals = pd.to_numeric(nums, errors="coerce")

    labels = acs_raw["Label (Grouping)"].astype(str)

    def canon(label: str) -> str:
        import re
        lab = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_").lower()
        if len(lab) > 40:
            lab = lab[:40]
        return lab or "feature"

    feat_dict = {}
    for lab, val in zip(labels, vals):
        colname = "acs_" + canon(lab)
        if colname not in feat_dict or pd.isna(feat_dict[colname]):
            feat_dict[colname] = val

    acs = pd.DataFrame([feat_dict])
    print(f"[ACS] macro features: {acs.shape[1]} columns from {len(acs_raw)} rows")
    return acs



def load_pmms(path: str) -> pd.DataFrame:
    pm = pd.read_csv(path)
    pm["pmms_date"] = pd.to_datetime(pm["date"], errors="coerce")
    for col in ["pmms30", "pmms15"]:
        if col in pm.columns:
            pm[col] = pd.to_numeric(pm[col], errors="coerce")
    pm = pm[["pmms_date"] + [c for c in ["pmms30", "pmms15"] if c in pm.columns]]
    pm = pm.dropna(subset=["pmms_date"]).sort_values("pmms_date")
    print(
        f"[PMMS] rows: {len(pm):,} ; date span: "
        f"{pm['pmms_date'].min().date()} → {pm['pmms_date'].max().date()}"
    )
    return pm



def join_views(hmda: pd.DataFrame, acs: pd.DataFrame, pmms: pd.DataFrame) -> pd.DataFrame:
    df = hmda.copy().sort_values("action_date")

    if acs is not None and not acs.empty:
        macro = acs.iloc[0]
        for col, val in macro.items():
            df[col] = val

    if pmms is not None and not pmms.empty:
        pmms_sorted = pmms.sort_values("pmms_date")
        df = pd.merge_asof(
            df,
            pmms_sorted,
            left_on="action_date",
            right_on="pmms_date",
            direction="nearest",
        )
        df.drop(columns=["pmms_date"], inplace=True)

    print(f"[Fusion] final rows: {len(df):,}")
    return df



def time_split(df: pd.DataFrame, date_col: str = "action_date",
               train_frac: float = 0.6, valid_frac: float = 0.2):
    df = df.sort_values(date_col)
    n = len(df)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))
    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    print(f"[Split] train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test



def build_schemes(df: pd.DataFrame):
    acs_cols = [c for c in df.columns if c.startswith("acs_")]
    pm_cols = [c for c in ["pmms30", "pmms15"] if c in df.columns]

    hmda_num = [c for c in ["loan_amount", "income", "interest_rate"] if c in df.columns]
    hmda_cat = [c for c in ["age", "race", "sex", "loan_purpose", "loan_type"] if c in df.columns]

    schemes = {
        "HMDA": {
            "num": hmda_num,
            "cat": hmda_cat,
        },
        "HMDA+ACS": {
            "num": hmda_num + acs_cols,
            "cat": hmda_cat,
        },
        "HMDA+ACS+PMMS": {
            "num": hmda_num + acs_cols + pm_cols,
            "cat": hmda_cat,
        },
    }
    return schemes

def run_scheme(name: str, cols: dict,
               train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    num_cols = cols["num"]
    cat_cols = cols["cat"]
    feats = num_cols + cat_cols

    X_train = train[feats].copy()
    X_valid = valid[feats].copy()
    X_test = test[feats].copy()

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_valid[c] = X_valid[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    y_train = train["y"].values
    y_valid = valid["y"].values
    y_test = test["y"].values

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    results = []
    for model_name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, proba_test)
        auc_pr = average_precision_score(y_test, proba_test)

        results.append(
            {
                "scheme": name,
                "model": model_name,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
            }
        )

    return results

def main():
    print("Loading PA HMDA ...")
    hmda = load_hmda(HMDA_PATH)

    print("Loading ACS (state-level macro) ...")
    acs = load_acs_macros(ACS_PATH)

    print("Loading PMMS ...")
    pmms = load_pmms(PMMS_PATH)

    print("Joining views (feature-level fusion) ...")
    full = join_views(hmda, acs, pmms)

    print("Time-based split ...")
    train, valid, test = time_split(full, date_col="action_date")

    schemes = build_schemes(full)

    all_results = []
    for name, cols in schemes.items():
        print(f"\n=== Scheme: {name} ===")
        res = run_scheme(name, cols, train, valid, test)
        for r in res:
            print(
                f"{r['model']}: AUC-ROC={r['auc_roc']:.3f}, "
                f"AUC-PR={r['auc_pr']:.3f}"
            )
        all_results.extend(res)

    results_df = pd.DataFrame(all_results)
    out_path = Path(REPORT_DIR) / "main_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
