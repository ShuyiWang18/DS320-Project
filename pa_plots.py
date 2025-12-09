import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from pa_project import (
    HMDA_PATH,
    ACS_PATH,
    PMMS_PATH,
    REPORT_DIR,
    load_hmda,
    load_acs_macros,
    load_pmms,
    join_views,
    time_split,
    build_schemes,
)

def train_lr_rf_XGB_for_scheme(scheme_name, full, train, valid, test):
   
    schemes = build_schemes(full)
    cols = schemes[scheme_name]
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

    probs = {}
    for name, mdl in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", mdl)])
        pipe.fit(X_train, y_train)

        proba_test = pipe.predict_proba(X_test)[:, 1]
        probs[name] = {
            "proba": proba_test,
            "model": pipe,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }

        auc_roc = roc_auc_score(y_test, proba_test)
        auc_pr = average_precision_score(y_test, proba_test)
        print(f"[{scheme_name}] {name}: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}")

    return probs, y_test


def plot_roc_pr_hmda(probs, y_test):
    lr_proba = probs["LR"]["proba"]
    rf_proba = probs["RF"]["proba"]
    xgb_proba = probs["XGB"]["proba"]

    # ROC
    fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba,  pos_label=1)
    fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_proba,  pos_label=1)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba, pos_label=1)

    plt.figure()
    plt.plot(fpr_lr,  tpr_lr,  label="LR")
    plt.plot(fpr_rf,  tpr_rf,  label="RF")
    plt.plot(fpr_xgb, tpr_xgb, label="XGB")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (HMDA)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "roc_hmda_lr_rf_xgb.png")


    # PR
    prec_lr,  rec_lr,  _ = precision_recall_curve(y_test, lr_proba,  pos_label=1)
    prec_rf,  rec_rf,  _ = precision_recall_curve(y_test, rf_proba,  pos_label=1)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_proba, pos_label=1)

    plt.figure()
    plt.plot(rec_lr,  prec_lr,  label="LR")
    plt.plot(rec_rf,  prec_rf,  label="RF")
    plt.plot(rec_xgb, prec_xgb, label="XGB")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (HMDA)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "pr_hmda_lr_rf_xgb.png")


def plot_rf_importance(probs, scheme_name="HMDA+ACS+PMMS", top_k=5):
    """画 Random Forest 的特征重要性（取前 top_k 个）"""
    rf_info = probs["RF"]
    model = rf_info["model"]
    num_cols = rf_info["num_cols"]
    cat_cols = rf_info["cat_cols"]

    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    feature_names = []

    if "num" in pre.named_transformers_:
        num_names = num_cols
        feature_names.extend(num_names)

    if "cat" in pre.named_transformers_:
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        cat_oh_names = list(oh.get_feature_names_out(cat_cols))
        feature_names.extend(cat_oh_names)

    importances = clf.feature_importances_
    m = min(len(importances), len(feature_names))
    feature_names = feature_names[:m]
    importances = importances[:m]

    idx = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_importances[::-1])
    plt.yticks(y_pos, top_features[::-1])
    plt.xlabel("Importance")
    plt.title(f"Random Forest Feature Importance ({scheme_name}, top {top_k})")
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, f"rf_importance_{scheme_name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved RF importance figure to {out_path}")


def plot_group_approval(full):

    if "race" in full.columns:
        tmp = full.copy()
        tmp["race_str"] = tmp["race"].astype(str)
        rate_by_race = tmp.groupby("race_str")["y"].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        rate_by_race.plot(kind="bar", rot=0, ax=ax)  
        ax.set_ylabel("Approval rate (mean of y)")
        ax.set_title("Approval rate by race code")

        caption = (
            "Note: race_str uses HMDA race codes. "
            "Many 2-digit codes (21, 22, 24, etc.) represent multi-race or "
            "more detailed subcategories in the official HMDA codebook."
        )
        fig.subplots_adjust(bottom=0.22)  
        fig.text(0.5, 0.02, caption, ha="center", fontsize=8)

        out_path = os.path.join(REPORT_DIR, "approval_rate_by_race.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved approval-by-race figure to {out_path}")

    if "sex" in full.columns:
        tmp = full.copy()
        tmp["sex_str"] = tmp["sex"].astype(str)
        rate_by_sex = tmp.groupby("sex_str")["y"].mean().sort_values(ascending=False)
        sex_code_map = {
            "1": "1: Male",
            "2": "2: Female",
            "3": "3: Joint",
            "4": "4: Not provided",
            "6": "6: Not applicable",
        }
        labels = [sex_code_map.get(code, code) for code in rate_by_sex.index]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(rate_by_sex)), rate_by_sex.values)
        ax.set_xticks(range(len(rate_by_sex)))
        ax.set_xticklabels(labels, rotation=0)  # x 轴文字正着
        ax.set_ylabel("Approval rate (mean of y)")
        ax.set_title("Approval rate by sex (HMDA codes)")

        caption = (
            "HMDA sex codes: 1=Male, 2=Female, 3=Joint, "
            "4=Information not provided, 6=Not applicable."
        )
        fig.subplots_adjust(bottom=0.3)
        fig.text(0.5, 0.05, caption, ha="center", fontsize=8)

        out_path = os.path.join(REPORT_DIR, "approval_rate_by_sex.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved approval-by-sex figure to {out_path}")


def main():
    print("Loading data again for plotting ...")
    hmda = load_hmda(HMDA_PATH)
    acs = load_acs_macros(ACS_PATH)
    pmms = load_pmms(PMMS_PATH)
    full = join_views(hmda, acs, pmms)

    train, valid, test = time_split(full, date_col="action_date")

    print("\n=== Training HMDA scheme for ROC/PR plots ===")
    hmda_probs, y_test_hmda = train_lr_rf_XGB_for_scheme("HMDA", full, train, valid, test)
    plot_roc_pr_hmda(hmda_probs, y_test_hmda)

    print("\n=== Training HMDA+ACS+PMMS scheme for RF importance ===")
    fusion_probs, y_test_fusion = train_lr_rf_XGB_for_scheme("HMDA+ACS+PMMS", full, train, valid, test)
    plot_rf_importance(fusion_probs, scheme_name="HMDA+ACS+PMMS", top_k=5)

    print("\n=== Plotting group-wise approval rates ===")
    plot_group_approval(full)

    print("\nAll figures saved under:", REPORT_DIR)


if __name__ == "__main__":
    main()
