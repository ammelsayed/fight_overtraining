"""Run a single (mode, channel, half) study point and write one JSON row."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdt_methods import BoostedDecisionTrees, to_Xyw

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
MASS_CHECK = 1000
MASS = "signal_mass"

MODES = {
    "baseline":     dict(mcw=False, reg=False, over="none",    wt="none"),
    "mcw":          dict(mcw=True,  reg=False, over="none",    wt="none"),
    "reg":          dict(mcw=False, reg=True,  over="none",    wt="none"),
    "bagging":      dict(mcw=True,  reg=False, over="bagging", wt="none"),
    "smote":        dict(mcw=True,  reg=False, over="smote",   wt="none"),
    "reg_bagging":  dict(mcw=False, reg=True,  over="bagging", wt="none"),
    "reg_smote":    dict(mcw=False, reg=True,  over="smote",   wt="none"),
    "reg_flatten":  dict(mcw=False, reg=True,  over="none",    wt="flatten"),
    "reg_clip":     dict(mcw=False, reg=True,  over="none",    wt="clip"),
    "flat_bagging": dict(mcw=False, reg=True,  over="bagging", wt="flatten"),
    "flat_smote":   dict(mcw=False, reg=True,  over="smote",   wt="flatten"),
}


def load(ch, half):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{half}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{half}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


def auc(bdt, df_sig, df_bkg):
    X, y, w = to_Xyw(df_sig, df_bkg, bdt.features, "flat_weight")
    return float(roc_auc_score(y, bdt.predict(X), sample_weight=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--channel", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = MODES[args.mode]
    test_half = 2 if args.half == 1 else 1
    sig_tr, bkg_tr, feats = load(args.channel, args.half)
    sig_te, bkg_te, _ = load(args.channel, test_half)
    features = feats + [MASS]

    bdt = BoostedDecisionTrees(sig_tr, bkg_tr, features,
                               oversample=cfg["over"], weight_treatment=cfg["wt"])
    if cfg["mcw"]:
        w_bal = bdt.balance_weights(bdt.ds["train"]["y"], bdt.ds["train"]["w"])
        mcw, _ = bdt.suggest_min_child_weight(w_bal, bdt.ds["train"]["y"])
        bdt.update_params(min_child_weight=float(mcw))
    if cfg["reg"]:
        bdt.suggest_regularization()

    t0 = time.time()
    bdt.train()
    dt = time.time() - t0

    out_dir = f"results/{args.mode}/{args.channel}/DF{args.half}"
    os.makedirs(out_dir, exist_ok=True)

    s_tr = sig_tr[sig_tr[MASS] == MASS_CHECK]
    b_tr = bkg_tr[bkg_tr[MASS] == MASS_CHECK]
    s_te = sig_te[sig_te[MASS] == MASS_CHECK]
    b_te = bkg_te[bkg_te[MASS] == MASS_CHECK]

    ot = bdt.check_overtraining(df_sig_train=s_tr, df_bkg_train=b_tr,
                                df_sig_test=s_te, df_bkg_test=b_te,
                                output_dir=out_dir)

    row = dict(
        channel=args.channel, mode=args.mode, train_half=args.half,
        auc_slice_train=auc(bdt, s_tr, b_tr), auc_slice_test=auc(bdt, s_te, b_te),
        auc_all_train=auc(bdt, sig_tr, bkg_tr), auc_all_test=auc(bdt, sig_te, bkg_te),
        best_iteration=int(bdt.model.best_iteration or 0),
        n_train_rows=len(bdt.ds["train"]["y"]), train_time_s=round(dt, 1),
        **(ot or {}),
    )
    for k in ("min_child_weight", "max_depth", "reg_lambda", "learning_rate",
              "subsample", "colsample_bytree"):
        row[k] = bdt.params.get(k)
    row["auc_slice_gap"] = row["auc_slice_train"] - row["auc_slice_test"]
    row["auc_all_gap"] = row["auc_all_train"] - row["auc_all_test"]

    with open(args.out, "w") as fh:
        json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else
                       int(v) if isinstance(v, (np.integer, int)) else v)
                   for k, v in row.items()}, fh, indent=1)
    print(f">>> {args.mode:12s} {args.channel:9s} DF{args.half}: "
          f"slice AUC {row['auc_slice_train']:.3f} -> {row['auc_slice_test']:.3f} "
          f"(gap {row['auc_slice_gap']:+.3f}) | "
          f"KS p sig={row.get('ks_p_sig_weighted', float('nan')):.3g} "
          f"bkg={row.get('ks_p_bkg_weighted', float('nan')):.3g} | "
          f"all-AUC test={row['auc_all_test']:.3f} | it={row['best_iteration']}", flush=True)


if __name__ == "__main__":
    main()
