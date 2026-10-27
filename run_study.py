import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bdt_methods import BoostedDecisionTrees, to_Xyw

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
CHANNELS = ["3L_eee", "3L_eeeJ", "4L_eeee"]
MASS_CHECK = 1000


def load(ch, half):
    d = f"{BASE}/{ch}"
    sig = pd.read_parquet(f"{d}/sig_DF{half}.parquet")
    bkg = pd.read_parquet(f"{d}/bkg_DF{half}.parquet")
    feats = json.load(open(f"{d}/features_SR.json"))
    return sig, bkg, feats


def auc_of(bdt, df_sig, df_bkg, weighted=True):
    X, y, w = to_Xyw(df_sig, df_bkg, bdt.features, weight_col="flat_weight")
    return float(roc_auc_score(y, bdt.predict(X), sample_weight=w if weighted else None))


def run(ch, train_half, mode, out_root, debug=False):
    test_half = 2 if train_half == 1 else 1
    sig_tr, bkg_tr, feats = load(ch, train_half)
    sig_te, bkg_te, _ = load(ch, test_half)
    features = feats + ["signal_mass"]

    kwargs = dict(apply_oversampling=True) if mode in ("smote", "both", "both2", "both3") else {}
    bdt = BoostedDecisionTrees(sig_tr, bkg_tr, features, **kwargs)
    # the server GPUs are shared and heavily contended -> run the whole matrix on CPU
    bdt.update_params(device="cpu", nthread=min(16, os.cpu_count() or 1))
    if mode in ("reg", "both", "reg2", "both2", "reg3", "both3"):
        bdt.debug = debug
        if mode in ("reg2", "both2"):
            # v2: winsorize extreme training weights at 10x the class mean before deriving/applying
            bdt.apply_weight_capping = True
        if mode in ("reg3", "both3"):
            # v3: winsorize at the class 95th-percentile weight (no multiplicative constant)
            bdt.apply_weight_capping = True
            bdt.weight_cap_mode = "percentile"
            bdt.weight_cap_factor = 95.0
        bdt.suggest_regularization(debug=debug)
    elif mode in ("baseline_mcw", "smote"):
        # the current method: XGBoost defaults + the ORIGINAL mean-based min_child_weight suggestion
        # (smote isolates the oversampling effect on top of it)
        bdt.mcw_strategy = "mean"
        bdt.use_min_child_weight_suggestion = True

    t0 = time.time()
    bdt.train()
    train_time = time.time() - t0

    out_dir = f"{out_root}/{mode}/{ch}/DF{train_half}"
    os.makedirs(out_dir, exist_ok=True)

    # official overtraining check for ONE signal mass hypothesis, train half vs test half
    s_tr = sig_tr[sig_tr["signal_mass"] == MASS_CHECK]
    b_tr = bkg_tr[bkg_tr["signal_mass"] == MASS_CHECK]
    s_te = sig_te[sig_te["signal_mass"] == MASS_CHECK]
    b_te = bkg_te[bkg_te["signal_mass"] == MASS_CHECK]
    ot = bdt.check_overtraining(df_sig_train=s_tr, df_bkg_train=b_tr,
                                df_sig_test=s_te, df_bkg_test=b_te,
                                output_dir=out_dir)

    row = dict(
        channel=ch, mode=mode, train_half=train_half,
        auc_slice_train=auc_of(bdt, s_tr, b_tr),
        auc_slice_test=auc_of(bdt, s_te, b_te),
        auc_slice_train_uw=auc_of(bdt, s_tr, b_tr, weighted=False),
        auc_slice_test_uw=auc_of(bdt, s_te, b_te, weighted=False),
        auc_all_train=auc_of(bdt, sig_tr, bkg_tr),
        auc_all_test=auc_of(bdt, sig_te, bkg_te),
        best_iteration=getattr(bdt.model, "best_iteration", None),
        n_features=len(features),
        n_train_rows=len(bdt.ds["train"]["y"]),
        train_time_s=round(train_time, 1),
        min_child_weight=bdt.params.get("min_child_weight"),
        max_depth=bdt.params.get("max_depth"),
        reg_lambda=bdt.params.get("reg_lambda"),
        learning_rate=bdt.params.get("learning_rate"),
        subsample=bdt.params.get("subsample"),
        colsample_bytree=bdt.params.get("colsample_bytree"),
        **ot,
    )
    row["auc_slice_gap"] = row["auc_slice_train"] - row["auc_slice_test"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "baseline_mcw", "reg", "smote", "both", "reg2", "both2", "reg3", "both3"],
                    choices=["baseline", "baseline_mcw", "reg", "smote", "both", "reg2", "both2", "reg3", "both3"])
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = []
    for mode in args.modes:
        for ch in args.channels:
            for half in (1, 2):
                print(f"\n{'#' * 100}\n# RUN mode={mode} channel={ch} train=DF{half}\n{'#' * 100}", flush=True)
                row = run(ch, half, mode, args.out, debug=args.debug)
                rows.append(row)
                print(f"\n>>> RESULT mode={mode} ch={ch} DF{half}: "
                      f"auc_slice_train={row['auc_slice_train']:.4f} auc_slice_test={row['auc_slice_test']:.4f} "
                      f"gap={row['auc_slice_gap']:+.4f} "
                      f"ks_p(sig/bkg)={row['ks_p_sig_weighted']:.3g}/{row['ks_p_bkg_weighted']:.3g} "
                      f"auc_all_test={row['auc_all_test']:.4f} best_it={row['best_iteration']}", flush=True)
                pd.DataFrame(rows).to_csv(f"{args.out}/results.csv", index=False)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["channel", "mode", "train_half", "auc_slice_train", "auc_slice_test", "auc_slice_gap",
            "ks_p_sig_weighted", "ks_p_bkg_weighted", "ks_p_sig_unweighted", "ks_p_bkg_unweighted",
            "auc_all_train", "auc_all_test", "best_iteration"]
    print("\n================ SUMMARY ================")
    print(df[cols].to_string(index=False))
    df.to_csv(f"{args.out}/results.csv", index=False)


if __name__ == "__main__":
    main()
