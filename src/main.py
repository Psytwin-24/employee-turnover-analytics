"""Employee turnover: IBM HR attrition, driver analysis, risk model."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data/hr_attrition.csv"))
df["target"] = (df["Attrition"] == "Yes").astype(int)
# NOTE: "target" must stay in this list. It is the same label as Attrition; if it
# ever lands in the feature matrix the model scores AUC 1.0 on nothing.
drop = ["Attrition", "target", "EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
X = df.drop(columns=drop)
X = pd.concat([X.select_dtypes(include=[np.number]),
               pd.get_dummies(X.select_dtypes(exclude=[np.number]), drop_first=True)], axis=1).astype(float)
y = df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
sc = StandardScaler().fit(X_tr)

# 1,470 rows x ~100 dummy features is linearly separable at default capacity, which
# is a model artefact, not signal. Both estimators are capacity-limited and the
# headline number is out-of-fold, so memorisation cannot inflate it.
lr = LogisticRegression(max_iter=4000, class_weight="balanced", C=0.05)
rf = RandomForestClassifier(n_estimators=500, max_depth=4, min_samples_leaf=25,
                            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
lr.fit(sc.transform(X_tr), y_tr); rf.fit(X_tr, y_tr)
cv = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
lrcv = cross_val_score(LogisticRegression(max_iter=4000, class_weight="balanced", C=0.05),
                       sc.transform(X), y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
res = {"attrition_rate": round(float(y.mean()), 4),
       "rf_cv_auc_mean": round(float(cv.mean()), 4),
       "rf_cv_auc_std": round(float(cv.std()), 4),
       "lr_cv_auc_mean": round(float(lrcv.mean()), 4),
       "rf_holdout_auc": round(float(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])), 4),
       "lr_holdout_auc": round(float(roc_auc_score(y_te, lr.predict_proba(sc.transform(X_te))[:, 1])), 4)}
p = rf.predict_proba(X_te)[:, 1]
res["rf_pr_auc"] = round(float(average_precision_score(y_te, p)), 4)
thr = 0.42
pr, rc, f1, _ = precision_recall_fscore_support(y_te, (p >= thr).astype(int), average="binary", zero_division=0)
res["flag_rate_at_0_42"] = round(float((p >= thr).mean()), 4)
res["precision_at_thr"] = round(float(pr), 3)
res["recall_at_thr"] = round(float(rc), 3)

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].barh(range(len(imp)), imp.values, color="tab:red")
axes[0].set_yticks(range(len(imp))); axes[0].set_yticklabels([i[:26] for i in imp.index], fontsize=7)
axes[0].invert_yaxis(); axes[0].set_title("Top attrition drivers (RF importance)")
ov = df.groupby("OverTime")["target"].agg(["mean", "size"])
axes[1].bar(ov.index, ov["mean"], color=["tab:gray", "tab:red"])
for i, v in enumerate(ov["mean"]): axes[1].text(i, v + 0.005, f"{v:.0%}", ha="center")
axes[1].set_ylabel("attrition rate"); axes[1].set_title("Attrition by overtime status")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/turnover.png"), dpi=110); plt.close()

metrics = {"results": res,
           "table": {"title": "Highest-importance drivers",
                     "headers": ["feature", "importance"],
                     "rows": [[i, round(float(v), 4)] for i, v in imp.items()]},
           "figures": [["turnover.png", "Driver ranking and the overtime effect"]]}
json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=1)
print(json.dumps(res))
