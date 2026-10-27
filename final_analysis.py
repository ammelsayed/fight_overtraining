"""Final analysis for the README: consolidate the grid, and test the hypothesis that the residual
train/test gap is governed by the effective (Kish) statistics rather than by the method."""
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import effective_events

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
pd.set_option("display.width", 260)

rows = [json.load(open(f)) for f in sorted(glob.glob("results_json/*.json"))]
df = pd.DataFrame(rows)

# effective statistics per (channel, half), from the training half
neff = {}
for ch in ["3L_eee", "3L_eeeJ", "4L_eeee"]:
    for h in (1, 2):
        b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF{h}.parquet")
        s = pd.read_parquet(f"{BASE}/{ch}/sig_DF{h}.parquet")
        ub = b.drop_duplicates("event_id")
        neff[(ch, h)] = (effective_events(ub.flat_weight.values),
                         effective_events(s.flat_weight.values),
                         ub.event_id.nunique())
df["neff_bkg"] = [neff[(r["channel"], int(r["train_half"]))][0] for r in rows]
df["neff_sig"] = [neff[(r["channel"], int(r["train_half"]))][1] for r in rows]
df["n_uniq_bkg"] = [neff[(r["channel"], int(r["train_half"]))][2] for r in rows]
df.to_csv("results/all_results.csv", index=False)

print("=== effective statistics of the training halves ===")
print(df.groupby("channel")[["n_uniq_bkg", "neff_bkg", "neff_sig"]].first().to_string(
    float_format=lambda x: f"{x:.3g}"))

print("\n=== averaged over the two DF1/DF2 directions ===")
g = df.groupby(["channel", "mode"]).agg(
    auc_tr=("auc_slice_train", "mean"), auc_te=("auc_slice_test", "mean"),
    gap=("auc_slice_gap", "mean"), gap_sd=("auc_slice_gap", "std"),
    ks_sig=("ks_p_sig_weighted", "mean"), ks_bkg=("ks_p_bkg_weighted", "mean"),
    ksd_sig=("ks_d_sig_weighted", "mean"), ksd_bkg=("ks_d_bkg_weighted", "mean"),
    chi2_sig=("chi2_p_sig_weighted", "mean"), chi2_bkg=("chi2_p_bkg_weighted", "mean"),
    auc_all=("auc_all_test", "mean"), it=("best_iteration", "mean")).reset_index()
print(g.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
g.to_csv("results/summary_by_channel_mode.csv", index=False)

print("\n=== is the residual gap governed by effective statistics? ===")
d = df[~df["mode"].isin(["baseline"])].dropna(subset=["neff_bkg", "auc_slice_gap"])
for label, x, y in [("N_eff(bkg)", d.neff_bkg, d.auc_slice_gap),
                    ("log N_eff(bkg)", np.log10(d.neff_bkg), d.auc_slice_gap)]:
    pr = pearsonr(x, y); sr = spearmanr(x, y)
    print(f"  gap vs {label:16s}: pearson r={pr[0]:+.3f} (p={pr[1]:.3g})  spearman={sr[0]:+.3f} (p={sr[1]:.3g})")

print("\n=== best mode per channel by (test AUC) with gap < 0.05 ===")
for ch, sub in g.groupby("channel"):
    ok = sub[sub.gap < 0.05].sort_values("auc_all", ascending=False)
    best = (ok if len(ok) else sub.sort_values("auc_all", ascending=False)).iloc[0]
    print(f"  {ch:9s}: {best['mode']:12s} test-AUC(all)={best['auc_all']:.4f} "
          f"slice {best['auc_tr']:.4f}->{best['auc_te']:.4f} gap={best['gap']:+.4f}")

print("\n=== weight-clipping is a no-op check ===")
for ch in ["3L_eee", "3L_eeeJ", "4L_eeee"]:
    b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF1.parquet").drop_duplicates("event_id")
    q = b.flat_weight.quantile(0.999)
    n_clipped = int((b.flat_weight > q).sum())
    print(f"  {ch:9s}: 99.9% quantile = {q:.4g}, events above it = {n_clipped} "
          f"(VVV share = {100*(b.flat_weight > 5e-3).mean():.2f}%)")
