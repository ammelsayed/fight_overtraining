"""Seed stability of the 4L_eeee train/test asymmetry.

The claim "the residual gap is fluctuation, not memorization" rests on the gap flipping sign
between the two DF directions. If the seed-to-seed spread is comparable to the between-direction
difference, that claim is supported; if not, it is not. Only the model seed is varied here
(subsample / colsample / split randomness); the data split is unchanged, so this isolates
optimization noise from data noise.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import BoostedDecisionTrees, to_Xyw

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="4L_eeee")
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sig, bkg, feats = load(args.channel, args.half)
    sig_te, bkg_te, _ = load(args.channel, 2 if args.half == 1 else 1)
    rows = []
    for seed in args.seeds:
        bdt = BoostedDecisionTrees(sig, bkg, feats + [MASS], weight_treatment="flatten")
        bdt.suggest_regularization()
        bdt.update_params(random_state=seed)
        bdt.train()
        s_tr = sig[sig[MASS] == M]; b_tr = bkg[bkg[MASS] == M]
        s_te = sig_te[sig_te[MASS] == M]; b_te = bkg_te[bkg_te[MASS] == M]
        r = dict(channel=args.channel, half=args.half, seed=seed,
                 auc_tr=auc(bdt, s_tr, b_tr), auc_te=auc(bdt, s_te, b_te),
                 auc_all_te=auc(bdt, sig_te, bkg_te),
                 it=int(bdt.model.best_iteration or 0))
        r["gap"] = r["auc_tr"] - r["auc_te"]
        rows.append(r)
        print(f"{args.channel} DF{args.half} seed={seed}: AUC {r['auc_tr']:.4f} -> {r['auc_te']:.4f} "
              f"gap={r['gap']:+.4f} all_test={r['auc_all_te']:.4f} it={r['it']}", flush=True)
    pd.DataFrame(rows).to_json(args.out, orient="records", indent=1)


if __name__ == "__main__":
    main()
