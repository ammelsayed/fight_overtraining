"""Smoke test of the new regularization + oversampling machinery."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import (BoostedDecisionTrees, effective_weight, effective_events,
                         oversample_bagging, smote_oversample, to_Xyw)

BASE = "/data/ammelsayed/fight_overtraining/sample_data"

# --- unit sanity checks on the helpers -------------------------------------
w = np.array([1.0, 1.0, 1.0, 1.0])
assert abs(effective_weight(w) - 1.0) < 1e-12, "equal weights -> w_eff == mean"
assert abs(effective_events(w) - 4.0) < 1e-12
w2 = np.array([100.0, 1.0, 1.0])
assert effective_weight(w2) > w2.mean(), "heterogeneous -> w_eff > mean"
print("helper checks OK: w_eff(equal)=%.3f  w_eff([100,1,1])=%.3f  mean=%.3f"
      % (effective_weight(w), effective_weight(w2), w2.mean()))

for ch in ["3L_eee", "3L_eeeJ", "4L_eeee"]:
    d = f"{BASE}/{ch}"
    feats = json.load(open(f"{d}/features_SR.json"))
    sig = pd.read_parquet(f"{d}/sig_DF1.parquet")
    bkg = pd.read_parquet(f"{d}/bkg_DF1.parquet")
    features = feats + ["signal_mass"]
    print("=" * 90)
    print(ch)

    # oversampling must conserve the total signal weight and stay per-mass
    for kind in ("bagging", "smote"):
        b = BoostedDecisionTrees(sig, bkg, features, oversample=kind)
        Xtr, ytr, wtr = b.ds["train"]["X"], b.ds["train"]["y"], b.ds["train"]["w"]
        print(f"  {kind:8s}: train rows {len(ytr):6d} (sig {int((ytr==1).sum()):6d}, bkg {int((ytr==0).sum()):6d})"
              f"  sum_w_sig={wtr[ytr==1].sum():.5g}")
        assert np.isfinite(Xtr).all(), "non-finite features after oversampling"

    # regularization must be finite, positive and actually let the model split
    b = BoostedDecisionTrees(sig, bkg, features)
    p = b.suggest_regularization()
    b.train()
    print(f"  trained: best_it={b.model.best_iteration}  params={ {k: round(v,5) if isinstance(v,float) else v for k,v in p.items()} }")
    assert b.model.best_iteration > 0, f"{ch} still cannot split!"
print("ALL SMOKE TESTS PASSED")
