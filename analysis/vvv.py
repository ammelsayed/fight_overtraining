"""Why does 4L_eeee DF1 train to AUC 0.997 but test at 0.876, while DF2 does the opposite?

Hypothesis: with N_eff ~3 the background is a handful of VVV events (one is 28-39% of the total
background weight). A model can achieve near-perfect *training* AUC by isolating those specific
events, which does not transfer to the other half.

Check: per-process, compare the fraction of the background weight that lands in the high-score
region (top 10% of the score range) in training vs in test.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import BoostedDecisionTrees, to_Xyw

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
MASS, M = "signal_mass", 1000


def load(ch, h):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{h}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{h}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


for h in (1, 2):
    sig, bkg, feats = load("4L_eeee", h)
    sig_te, bkg_te, _ = load("4L_eeee", 2 if h == 1 else 1)
    bdt = BoostedDecisionTrees(sig, bkg, feats + [MASS], weight_treatment="flatten")
    bdt.suggest_regularization()
    bdt.train()

    b_tr = bkg[bkg[MASS] == M].copy()
    b_te = bkg_te[bkg_te[MASS] == M].copy()
    b_tr["score"] = bdt.predict(b_tr[bdt.features].values.astype(np.float32))
    b_te["score"] = bdt.predict(b_te[bdt.features].values.astype(np.float32))

    print(f"\n=== 4L_eeee train=DF{h} ===")
    print(f"  bkg train n={len(b_tr)} sum_w={b_tr.flat_weight.sum():.4g} | "
          f"test n={len(b_te)} sum_w={b_te.flat_weight.sum():.4g}")
    for tag, d in (("train", b_tr), ("test ", b_te)):
        tot = d.flat_weight.sum()
        print(f"  {tag}: mean score (weighted) = {np.average(d.score, weights=d.flat_weight):.3f}")
        for proc, g in d.groupby("proc_name"):
            share = 100 * g.flat_weight.sum() / tot
            # fraction of this process' weight that lands above score 0.9
            hi = 100 * g.loc[g.score > 0.9, "flat_weight"].sum() / g.flat_weight.sum()
            print(f"      {proc:<7s} n={len(g):3d} weight share={share:5.1f}%  "
                  f"score>0.9 holds {hi:5.1f}% of its weight")
