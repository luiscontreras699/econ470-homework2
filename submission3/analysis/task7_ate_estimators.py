from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "submission3" / "results"
OUT = RES
OUT.mkdir(exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(RES / "task6_county_level_with_ffs.csv")

# Outcome, treatment, covariates
Y = df["county_avg_bid_proxy"].astype(float).values
T = df["treatment"].astype(int).values.reshape(-1, 1)  # (n,1) for broadcasting

Q = df[["ffs_q1", "ffs_q2", "ffs_q3", "ffs_q4"]].astype(int).values  # (n,4)

treated = df[df["treatment"] == 1].copy()
control = df[df["treatment"] == 0].copy()

Qt = treated[["ffs_q1", "ffs_q2", "ffs_q3", "ffs_q4"]].astype(int).values
Qc = control[["ffs_q1", "ffs_q2", "ffs_q3", "ffs_q4"]].astype(int).values

# ----------------------------
# 1) Nearest Neighbor (1-to-1) – Inverse Variance distance on quartile dummies
# ----------------------------
var = Q.var(axis=0)
w_invvar = 1.0 / (var + 1e-8)
Qc_scaled = Qc * w_invvar
Qt_scaled = Qt * w_invvar

nn_iv = NearestNeighbors(n_neighbors=1)
nn_iv.fit(Qc_scaled)
dist_iv, idx_iv = nn_iv.kneighbors(Qt_scaled)

matched_control_iv = control.iloc[idx_iv.flatten()]
ate_nn_invvar = treated["county_avg_bid_proxy"].mean() - matched_control_iv["county_avg_bid_proxy"].mean()

# ----------------------------
# 2) Nearest Neighbor (1-to-1) – Mahalanobis distance on quartile dummies
# ----------------------------
cov = np.cov(Q.T)
inv_cov = np.linalg.pinv(cov)

nn_maha = NearestNeighbors(
    n_neighbors=1,
    metric="mahalanobis",
    metric_params={"VI": inv_cov}
)
nn_maha.fit(Qc)
dist_maha, idx_maha = nn_maha.kneighbors(Qt)

matched_control_maha = control.iloc[idx_maha.flatten()]
ate_nn_maha = treated["county_avg_bid_proxy"].mean() - matched_control_maha["county_avg_bid_proxy"].mean()

# ----------------------------
# 3) Inverse Propensity Weighting (IPW) with propensity based on quartiles
# ----------------------------
# Logistic regression p(T=1|Q)
logit = LogisticRegression(max_iter=5000, solver="lbfgs")
logit.fit(Q, df["treatment"].astype(int).values)
ps = logit.predict_proba(Q)[:, 1]

# clip to avoid exploding weights
eps = 1e-4
ps = np.clip(ps, eps, 1 - eps)

# Stabilized weights
p_treat = df["treatment"].mean()
sw = np.where(df["treatment"] == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

# IPW ATE = weighted mean(Y|T=1) - weighted mean(Y|T=0)
y1 = np.sum(sw[df["treatment"] == 1] * df.loc[df["treatment"] == 1, "county_avg_bid_proxy"]) / np.sum(sw[df["treatment"] == 1])
y0 = np.sum(sw[df["treatment"] == 0] * df.loc[df["treatment"] == 0, "county_avg_bid_proxy"]) / np.sum(sw[df["treatment"] == 0])
ate_ipw = y1 - y0

# ----------------------------
# 4) Linear regression with quartile dummies + interactions
# ----------------------------
# Model: Y = a + b*T + sum_q gq*Qq + sum_q dq*(T*Qq) + e
X = np.column_stack([T, Q, T * Q])  # (n, 1+4+4)
X = sm.add_constant(X)

ols = sm.OLS(Y, X).fit()

# ATE is average marginal effect of T:
# In this saturated interaction model with dummies, ATE = b + sum_q dq * E[Qq]
b = ols.params[1]
d = ols.params[1 + 4 + 1 : 1 + 4 + 4 + 1]  # interaction params (4 of them)
EQ = Q.mean(axis=0)
ate_reg = b + np.dot(d, EQ)

# ----------------------------
# Output table
# ----------------------------
results = pd.DataFrame({
    "Estimator": [
        "NN 1-to-1 (Inverse Variance distance on FFS quartiles)",
        "NN 1-to-1 (Mahalanobis distance on FFS quartiles)",
        "IPW (propensity from FFS quartiles)",
        "Linear regression (quartiles + interactions)"
    ],
    "ATE": [
        float(ate_nn_invvar),
        float(ate_nn_maha),
        float(ate_ipw),
        float(ate_reg)
    ]
})

print("\n=== Task 7 ATE Results (2018) ===")
print(results.to_string(index=False))

out_path = OUT / "task7_ate_results.csv"
results.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")