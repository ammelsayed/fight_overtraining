"""Does bounding max_depth by the effective number of minority-class events fix 4L_eeee DF1?

The current depth rule uses the *hessian* budget, h_tot/mcw, which for 4L_eeee gives 16 leaves.
But the background has N_eff ~ 3 effective events: a tree cannot populate 16 leaves with 3 events
without isolating individual events. Test whether depth <= log2(N_eff) removes the gap.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import BoostedDecisionTrees, to_Xyw, effective_events

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
MASS, M = "signal_mass", 1000


def load(ch, h):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{h}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{h}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


def auc(bdt, a, b):
    X, y, w = to_Xyw(a, b, bdt.features, "flat_weight")
    return float(roc_auc_score(y, bdt.predict(X), sample_weight=w))


ap = argparse.ArgumentParser()
ap.add_argument("--channel", required=True)
ap.add_argument("--half", type=int, required=True)
ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--out", required=True)
args = ap.parse_args()

sig, bkg, feats = load(args.channel, args.half)
sig_te, bkg_te, _ = load(args.channel, 2 if args.half == 1 else 1)
rows = []
for depth in args.depths:
    bdt = BoostedDecisionTrees(sig, bkg, feats + [MASS], weight_treatment="flatten")
    bdt.suggest_regularization()
    bdt.update_params(max_depth=depth)
    bdt.train()
    s_tr = sig[sig[MASS] == M]; b_tr = bkg[bkg[MASS] == M]
    s_te = sig_te[sig_te[MASS] == M]; b_te = bkg_te[bkg_te[MASS] == M]
    y, w = bdt.ds["train"]["y"], bdt.ds["train"]["w"]
    r = dict(channel=args.channel, half=args.half, depth=depth,
             neff_bkg=effective_events(w[y == 0]), neff_sig=effective_events(w[y == 1]),
             auc_tr=auc(bdt, s_tr, b_tr), auc_te=auc(bdt, s_te, b_te),
             auc_all_te=auc(bdt, sig_te, bkg_te), it=int(bdt.model.best_iteration or 0))
    r["gap"] = r["auc_tr"] - r["auc_te"]
    rows.append(r)
    print(f"{args.channel} DF{args.half} depth={depth} N_eff(bkg)={r['neff_bkg']:.1f} | "
          f"AUC {r['auc_tr']:.4f} -> {r['auc_te']:.4f} gap={r['gap']:+.4f} "
          f"all_test={r['auc_all_te']:.4f} it={r['it']}", flush=True)
pd.DataFrame(rows).to_json(args.out, orient="records", indent=1)
