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
-XGBoost
Rating:
- 60% / 20% / 20% adaptive split according to time
- ROC AUC, PR AUC
- Three solutions: HMDA / HMDA+ACS / HMDA+ACS+PMMS
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier


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
    pm = pm[["pmms_date"] + [c for c in ["pmms30", "pmms15"] if c in pm.columns]]
    pm = pm.dropna(subset=["pmms_date"])


    pm = pm[(pm["pmms_date"] >= "2024-01-01") & (pm["pmms_date"] <= "2024-12-31")]

    pm = pm.sort_values("pmms_date")

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
        direction="backward",                 
        tolerance=pd.Timedelta("60D"),        
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
def run_scheme(name: str, cols: dict, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    """
    cols: {"num":[...], "cat":[...]}  (from build_schemes)
    - TimeSeriesSplit CV on TRAIN (prints mean±std)
    - Fit on TRAIN, evaluate on TEST (AUC-ROC / AUC-PR)
    - Save per-income-band metrics (6 bands, qcut) to income_band_model_metrics_{scheme}.csv
    """

    # ---- Config ----
    DO_CV = True
    N_SPLITS = 5
    USE_QCUT = True   # True -> qcut=6 (balanced). False -> fixed thresholds
    QCUT_Q = 6
    THR = 0.5

    num_cols = cols["num"]
    cat_cols = cols["cat"]
    feats = num_cols + cat_cols

    # ---- Data ----
    X_train = train[feats].copy()
    X_test = test[feats].copy()

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    y_train = train["y"].values
    y_test = test["y"].values

    # ---- Preprocess ----
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    # ---- Models (keep yours) ----
    models = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    band_tables = []

    tscv = TimeSeriesSplit(n_splits=N_SPLITS) if DO_CV else None

    for model_name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])

        # ===== Real CV (TRAIN only) =====
        if DO_CV:
            cv_out = cross_validate(
                pipe,
                X_train, y_train,
                cv=tscv,
                scoring={"auc_roc": "roc_auc", "auc_pr": "average_precision"},
                n_jobs=-1,
            )
            print(
                f"[CV] {name}-{model_name}: "
                f"ROC-AUC={cv_out['test_auc_roc'].mean():.3f}±{cv_out['test_auc_roc'].std():.3f}, "
                f"PR-AUC={cv_out['test_auc_pr'].mean():.3f}±{cv_out['test_auc_pr'].std():.3f}"
            )

        # ===== Fit on TRAIN, test on TEST =====
        pipe.fit(X_train, y_train)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, proba_test)
        auc_pr = average_precision_score(y_test, proba_test)

        results.append({"scheme": name, "model": model_name, "auc_roc": auc_roc, "auc_pr": auc_pr})

        # ===== Income-band (6 bands) metrics on TEST =====
        try:
            if USE_QCUT:
                band_df = income_band_metrics_qcut(
                    model=pipe,
                    X=X_test,
                    y=y_test,
                    income_series=test["income"],
                    q=QCUT_Q,
                    thr=THR,
                )
            else:
                thresholds = _auto_income_thresholds(test["income"])
                band_df = income_band_metrics(
                    model=pipe,
                    X=X_test,
                    y=y_test,
                    income_series=test["income"],
                    thresholds=thresholds,
                    thr=THR,
                )

            band_df["scheme"] = name
            band_df["model"] = model_name
            band_tables.append(band_df)

        except Exception as e:
            print(f"[WARN] income-band metrics failed for {name}-{model_name}: {e}")

    # ---- Save per-band table for this scheme ----
    if band_tables:
        band_all = pd.concat(band_tables, ignore_index=True)
        out_band = Path(REPORT_DIR) / f"income_band_model_metrics_{name}.csv"
        band_all.to_csv(out_band, index=False)
        print(f"Saved income-band model metrics to: {out_band}")

    return results



def _auto_income_thresholds(income_series: pd.Series):
    inc = pd.to_numeric(income_series, errors="coerce")
    med = inc.median()
    if pd.isna(med):
        return [2, 5, 10, 15, 20, 25]
    return [2000, 5000, 10000, 15000, 20000, 25000] if med > 1000 else [2, 5, 10, 15, 20, 25]

def income_band_metrics_qcut(model, X, y, income_series, q=6, thr=0.5):
    inc = pd.to_numeric(income_series, errors="coerce")
    band = pd.qcut(inc, q=q, duplicates="drop")

    y_hat = model.predict_proba(X)[:, 1]
    pred_pos = (y_hat >= thr).astype(int)

    df = pd.DataFrame(
        {"income_band": band, "y": y, "y_hat": y_hat, "pred_pos": pred_pos}
    ).dropna(subset=["income_band"])

    total = len(df)
    rows = []
    for b, g in df.groupby("income_band", observed=True):
        auc_roc = np.nan
        auc_pr = np.nan
        if g["y"].nunique() == 2:
            auc_roc = roc_auc_score(g["y"], g["y_hat"])
            auc_pr = average_precision_score(g["y"], g["y_hat"])

        rows.append(
            {
                "income_band": str(b),
                "n": len(g),
                "pct_of_test": 100.0 * len(g) / max(total, 1),
                "approval_rate": g["y"].mean(),
                "mean_pred_prob": g["y_hat"].mean(),
                "pred_pos_rate@thr": g["pred_pos"].mean(),
                "auc_roc_band": auc_roc,
                "auc_pr_band": auc_pr,
            }
        )
    return pd.DataFrame(rows)

def income_band_metrics(model, X, y, income_series, thresholds, thr=0.5):
    """
    output
    - n
    - pct_of_test (%)
    - approval_rate
    - mean_pred_prob 
    - pred_pos_rate@thr 
    - auc_roc_band / auc_pr_band 
    """
    inc = pd.to_numeric(income_series, errors="coerce")

    bins = [0] + list(thresholds) + [np.inf]
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if hi == np.inf:
            labels.append(f"{int(lo)}+")
        else:
            labels.append(f"{int(lo)}–{int(hi)}")

    band = pd.cut(inc, bins=bins, labels=labels, right=False)

    y_hat = model.predict_proba(X)[:, 1]
    pred_pos = (y_hat >= thr).astype(int)

    df = pd.DataFrame(
        {"income_band": band, "y": y, "y_hat": y_hat, "pred_pos": pred_pos}
    ).dropna(subset=["income_band"])

    total = len(df)
    rows = []
    for b, g in df.groupby("income_band", observed=True):
        auc_roc = np.nan
        auc_pr = np.nan
        if g["y"].nunique() == 2:
            auc_roc = roc_auc_score(g["y"], g["y_hat"])
            auc_pr = average_precision_score(g["y"], g["y_hat"])

        rows.append(
            {
                "income_band": str(b),
                "n": len(g),
                "pct_of_test": 100.0 * len(g) / max(total, 1),
                "approval_rate": g["y"].mean(),
                "mean_pred_prob": g["y_hat"].mean(),
                "pred_pos_rate@thr": g["pred_pos"].mean(),
                "auc_roc_band": auc_roc,
                "auc_pr_band": auc_pr,
            }
        )

    return pd.DataFrame(rows)

def income_band_analysis(model, X_test, y_test, income_series, thresholds, title_prefix=""):
 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    bins = [0] + thresholds + [np.inf]
    labels = []
    for i in range(len(thresholds) + 1):
        lo = bins[i]
        hi = bins[i + 1]
        if hi == np.inf:
            labels.append(f"{int(lo)}+")
        else:
            labels.append(f"{int(lo)}–{int(hi)}")

    bands = pd.cut(income_series, bins=bins, labels=labels, right=False)


    df_band = pd.DataFrame({
        "income_band": bands,
        "y": y_test,
        "y_hat": model.predict_proba(X_test)[:, 1],
    })

    summary = (
        df_band
        .groupby("income_band", observed=True)
        .agg(
            n=("y", "size"),
            approval_rate=("y", "mean"),
            mean_pred_prob=("y_hat", "mean"),
        )
        .reset_index()
        .sort_values("income_band")
    )

    print("\n=== Income band analysis ===")
    print(summary)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    width = 0.35

    plt.bar(x - width/2, summary["approval_rate"], width, label="Actual approval rate")
    plt.bar(x + width/2, summary["mean_pred_prob"], width, label="Predicted approval prob")

    plt.xticks(x, summary["income_band"])
    plt.xlabel("Income band")
    plt.ylabel("Rate / probability")
    plt.title(f"{title_prefix} Income effect on approval")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return summary
def plot_income_bins(full: pd.DataFrame):
    """
    Cross-validation / sensitivity: how does approval rate change
    across different income levels?
    """
    df = full.copy()

    df["income_num"] = pd.to_numeric(df["income"], errors="coerce")

    bins = [0, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, np.inf]
    labels = ["≤2k", "2k–5k", "5k–10k", "10k–15k", "15k–20k", "20k–25k", ">25k"]
    df["income_bin"] = pd.cut(
        df["income_num"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    summary = (
        df.groupby("income_bin")["y"]
        .agg(approval_rate="mean", n="size")
        .reset_index()
    )

    print("\n=== Approval rate by income bucket ===")
    print(summary)

    plt.figure(figsize=(7, 4))
    plt.bar(summary["income_bin"].astype(str), summary["approval_rate"])
    plt.xlabel("Applicant income bucket")
    plt.ylabel("Approval rate (mean of y)")
    plt.title("Approval rate by income bucket")
    plt.tight_layout()

    out_path = os.path.join(REPORT_DIR, "approval_rate_by_income_bin.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved approval-by-income figure to {out_path}")


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



    print("\nIncome-band cross-validation: approval rate by income level ...")

    income_df = full[["income", "y"]].dropna().copy()

    income_df["income_num"] = pd.to_numeric(income_df["income"], errors="coerce")
    income_df = income_df.dropna(subset=["income_num"])

    bins = [0, 2, 5, 10, 15, 25, np.inf]
    labels = ["<=2", "2–5", "5–10", "10–15", "15–25", ">25"]

    income_df["income_band"] = pd.cut(
        income_df["income_num"],  
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
)

    income_summary = (
        income_df
        .groupby("income_band", observed=True)["y"]
        .agg(approval_rate="mean", n="size")
        .reset_index()
)

    print("\nApproval rate by income band:")
    print(income_summary)

    band_out = Path(REPORT_DIR) / "income_band_summary.csv"
    income_summary.to_csv(band_out, index=False)
    print(f"Saved income-band summary to: {band_out}")


if __name__ == "__main__":
    main()
