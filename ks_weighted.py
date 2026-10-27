"""The weighted KS p-value quoted in the README is low in one direction (p~0.002).

Weighted KS on a sample whose weights span 1085x has a small effective size, so its p-value can be
small while the *distribution* is unchanged. Test: bootstrap the null of the weighted KS statistic
and compare the observed value against it. If the observed D sits inside the null, the low p-value
is an artefact of the weight heterogeneity, not evidence of overtraining.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import BoostedDecisionTrees, to_Xyw

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
MASS, M = "signal_mass", 1000
rng = np.random.RandomState(11)


def load(ch, h):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{h}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{h}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


def weighted_ks(x, w, y, v):
    """Two-sample weighted KS statistic: max |CDF_x - CDF_y| with normalised weights."""
    order = np.argsort(np.concatenate([x, y]))
    xs = np.concatenate([x, y])[order]
    ws = np.concatenate([w, -v])[order]  # negative marks the second sample
    cdf1 = np.cumsum(np.where(ws > 0, ws, 0.0)) / w.sum()
    cdf2 = np.cumsum(np.where(ws < 0, -ws, 0.0)) / v.sum()
    return float(np.max(np.abs(cdf1 - cdf2)))


def null_weighted_ks(x, w, y, v, n=300):
    """Pool the two weighted samples and resample the pooled labels/weights at random."""
    pool_x = np.concatenate([x, y])
    pool_w = np.concatenate([w, v])
    n1, n2 = len(x), len(y)
    ds = []
    for _ in range(n):
        idx = rng.permutation(len(pool_x))
        a, b = idx[:n1], idx[n1:n1 + n2]
        ds.append(weighted_ks(pool_x[a], pool_w[a], pool_x[b], pool_w[b]))
    return np.array(ds)


for ch, h in [("3L_eee", 1), ("3L_eee", 2), ("3L_eeeJ", 1), ("3L_eeeJ", 2),
              ("4L_eeee", 1), ("4L_eeee", 2)]:
    sig, bkg, feats = load(ch, h)
    sig_te, bkg_te, _ = load(ch, 2 if h == 1 else 1)
    bdt = BoostedDecisionTrees(sig, bkg, feats + [MASS], weight_treatment="flatten")
    bdt.suggest_regularization()
    bdt.train()

    b_tr = bkg[bkg[MASS] == M]; b_te = bkg_te[bkg_te[MASS] == M]
    Xtr, _, wtr = to_Xyw(b_tr.iloc[:0], b_tr, bdt.features, "flat_weight")
    Xte, _, wte = to_Xyw(b_te.iloc[:0], b_te, bdt.features, "flat_weight")
    ptr, pte = bdt.predict(Xtr), bdt.predict(Xte)

    D = weighted_ks(ptr, wtr, pte, wte)
    null = null_weighted_ks(ptr, wtr, pte, wte, n=300)
    p95 = np.percentile(null, 95)
    frac = float((null >= D).mean())
    print(f"{ch} DF{h}: weighted bkg KS D={D:.4f}  null median={np.median(null):.4f} "
          f"95th={p95:.4f}  bootstrap p={frac:.3f}  "
          f"({'INSIDE null' if D <= p95 else 'EXCEEDS null'})  "
          f"N_eff(train)={wtr.sum()**2/(wtr**2).sum():.1f}", flush=True)
