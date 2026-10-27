"""Study driver.

For each channel, train on DF1 and evaluate on DF2, then train on DF2 and evaluate on DF1.
The overtraining check is run on the M=1000 GeV hypothesis only, as requested.

Modes (everything is derived from the sample; no HPO, no per-channel constants):

  baseline        class defaults - documents the problem
  mcw             only the leaf floor is fixed (units fix), nothing else
  reg             full data-driven regularization
  bagging         mcw + bagging oversampling
  smote           mcw + SMOTE oversampling
  reg_bagging     full regularization + bagging oversampling
  reg_smote       full regularization + SMOTE oversampling
  reg_flatten     full regularization + per-process weight flattening
  reg_clip        full regularization + weight clipping at the 99.9% quantile
"""
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
CHANNELS = ["3L_eee", "3L_eeeJ", "4L_eeee"]
MASS_CHECK = 1000
MASS = "signal_mass"

# mode -> (fix mcw only, full regularization, oversampling, weight treatment)
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
}


def load(ch, half):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{half}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{half}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


def auc(bdt, df_sig, df_bkg):
    X, y, w = to_Xyw(df_sig, df_bkg, bdt.features, "flat_weight")
    return float(roc_auc_score(y, bdt.predict(X), sample_weight=w))


def run(ch, train_half, mode, out_root):
    cfg = MODES[mode]
    test_half = 2 if train_half == 1 else 1
    sig_tr, bkg_tr, feats = load(ch, train_half)
    sig_te, bkg_te, _ = load(ch, test_half)
    features = feats + [MASS]

    bdt = BoostedDecisionTrees(sig_tr, bkg_tr, features,
                               oversample=cfg["over"], weight_treatment=cfg["wt"])

    if cfg["mcw"]:
        # only the leaf floor, in effective-event currency
        w_bal = bdt.balance_weights(bdt.ds["train"]["y"], bdt.ds["train"]["w"])
        mcw, _ = bdt.suggest_min_child_weight(w_bal, bdt.ds["train"]["y"])
        bdt.update_params(min_child_weight=float(mcw))
    if cfg["reg"]:
        bdt.suggest_regularization()

    t0 = time.time()
    bdt.train()
    dt = time.time() - t0

    out_dir = f"{out_root}/{mode}/{ch}/DF{train_half}"
    os.makedirs(out_dir, exist_ok=True)

    s_tr = sig_tr[sig_tr[MASS] == MASS_CHECK]
    b_tr = bkg_tr[bkg_tr[MASS] == MASS_CHECK]
    s_te = sig_te[sig_te[MASS] == MASS_CHECK]
    b_te = bkg_te[bkg_te[MASS] == MASS_CHECK]

    ot = bdt.check_overtraining(df_sig_train=s_tr, df_bkg_train=b_tr,
                                df_sig_test=s_te, df_bkg_test=b_te,
                                output_dir=out_dir)

    row = dict(
        channel=ch, mode=mode, train_half=train_half,
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
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--halves", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/results.csv"
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []

    done = {(r["mode"], r["channel"], int(r["train_half"])) for r in rows}
    for mode in args.modes:
        for ch in args.channels:
            for half in args.halves:
                if (mode, ch, half) in done:
                    print(f"skip {mode} {ch} DF{half} (already done)", flush=True)
                    continue
                print(f"\n{'#'*100}\n# mode={mode} channel={ch} train=DF{half}\n{'#'*100}", flush=True)
                r = run(ch, half, mode, args.out)
                rows.append(r)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"\n>>> {mode:12s} {ch:9s} DF{half}: slice AUC {r['auc_slice_train']:.3f} -> "
                      f"{r['auc_slice_test']:.3f} (gap {r['auc_slice_gap']:+.3f}) | "
                      f"KS p sig={r.get('ks_p_sig_weighted', float('nan')):.3g} "
                      f"bkg={r.get('ks_p_bkg_weighted', float('nan')):.3g} | "
                      f"chi2 p sig={r.get('chi2_p_sig_weighted', float('nan')):.3g} "
                      f"bkg={r.get('chi2_p_bkg_weighted', float('nan')):.3g} | "
                      f"all-AUC test={r['auc_all_test']:.3f} | it={r['best_iteration']}", flush=True)

    df = pd.DataFrame(rows)
    cols = ["channel", "mode", "train_half", "auc_slice_train", "auc_slice_test", "auc_slice_gap",
            "ks_p_sig_weighted", "ks_p_bkg_weighted", "chi2_p_sig_weighted", "chi2_p_bkg_weighted",
            "ks_d_sig_weighted", "ks_d_bkg_weighted", "auc_all_test", "best_iteration",
            "min_child_weight", "max_depth", "reg_lambda", "subsample", "colsample_bytree"]
    print("\n================ SUMMARY ================")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
