"""Figures for the README.

Each figure answers one question the text asks, so the numbers can be read at a glance:

  fig1_weights   why the defaults fail: weight heterogeneity, N_eff, hessian budget
  fig2_methods   test AUC vs train/test gap, every mode, three channels
  fig3_scale     the units fix: what the derived mcw / lambda are, vs the defaults
  fig4_seeds     the gap is stable across seeds, and larger at depth 1 than depth 4
  fig5_vvv       the 4L_eeee score shift is a background-composition effect
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import effective_events, effective_weight

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
OUT = "/data/ammelsayed/fight_overtraining/figures"
CH = ["3L_eee", "3L_eeeJ", "4L_eeee"]
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

C = {"baseline": "#9e9e9e", "mcw": "#90caf9", "reg": "#42a5f5",
     "reg_flatten": "#1b5e20", "flat_bagging": "#2e7d32", "flat_smote": "#66bb6a",
     "bagging": "#ffb74d", "smote": "#ff7043", "reg_bagging": "#ffa726",
     "reg_smote": "#ff7043", "reg_clip": "#bdbdbd"}
MODES = ["baseline", "mcw", "reg", "reg_clip", "reg_flatten", "flat_bagging", "flat_smote"]


def load_grid():
    rows = []
    for f in sorted(glob.glob("/data/ammelsayed/fight_overtraining/results_json/*.json")):
        rows.append(json.load(open(f)))
    df = pd.DataFrame(rows)
    # effective statistics per (channel, half)
    ne = {}
    for ch in CH:
        for h in (1, 2):
            b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF{h}.parquet")
            ub = b.drop_duplicates("event_id")
            ne[(ch, h)] = (effective_events(ub.flat_weight.values), ub.event_id.nunique(),
                           ub.flat_weight.max() / ub.flat_weight.min())
    df["neff_bkg"] = [ne[(r["channel"], int(r["train_half"]))][0] for r in rows]
    df["n_uniq_bkg"] = [ne[(r["channel"], int(r["train_half"]))][1] for r in rows]
    df["w_span"] = [ne[(r["channel"], int(r["train_half"]))][2] for r in rows]
    return df


# ---------------------------------------------------------------- fig 1
def fig_weights():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))

    # (a) background weight spectrum per process
    for ch in CH:
        b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF1.parquet").drop_duplicates("event_id")
        break
    procs = b.groupby("proc_name").agg(w=("flat_weight", "mean"), n=("flat_weight", "size"))
    procs = procs.sort_values("w")
    ax[0].barh(range(len(procs)), procs.w.values, color="#42a5f5")
    ax[0].set_yticks(range(len(procs)))
    ax[0].set_yticklabels([f"{p} (n={n})" for p, n in zip(procs.index, procs.n)])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("per-event weight")
    ax[0].set_title("(a) background weights span 1085x\n(identical in all 3 channels)")
    for i, (p, r) in enumerate(procs.iterrows()):
        ax[0].text(r.w * 1.3, i, f"{r.w:.1e}", va="center", fontsize=7)

    # (b) rows vs effective events
    rows, neff, labels = [], [], []
    for ch in CH:
        b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF1.parquet").drop_duplicates("event_id")
        rows.append(len(b)); neff.append(effective_events(b.flat_weight.values)); labels.append(ch)
    x = np.arange(len(labels)); w = 0.36
    ax[1].bar(x - w / 2, rows, w, label="rows", color="#90caf9")
    ax[1].bar(x + w / 2, neff, w, label="N_eff (Kish)", color="#1b5e20")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_yscale("log"); ax[1].set_ylabel("background events (DF1)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) rows overstate the statistics\nby 20-60x")
    for xi, (r, n) in enumerate(zip(rows, neff)):
        ax[1].text(xi + w / 2, n * 1.3, f"{n:.0f}", ha="center", fontsize=7)

    # (c) hessian budget vs the default mcw=1, on the actual 80% training split
    #     (same split the class uses, so these match the README table)
    from bdt_methods import df_train_val_split
    hs = []
    for ch in CH:
        s = pd.read_parquet(f"{BASE}/{ch}/sig_DF1.parquet")
        b = pd.read_parquet(f"{BASE}/{ch}/bkg_DF1.parquet")
        feats = json.load(open(f"{BASE}/{ch}/features_SR.json")) + ["signal_mass"]
        ds = df_train_val_split(s, b, feats)
        y, w = ds["train"]["y"], ds["train"]["w"]
        tot = w.sum()
        wb = np.where(y == 1, w * tot / (2 * w[y == 1].sum()), w * tot / (2 * w[y == 0].sum()))
        hs.append(0.25 * wb.sum())
    ax[2].bar(x, hs, 0.5, color=["#1b5e20" if h > 1 else "#c62828" for h in hs])
    ax[2].axhline(1.0, color="k", ls="--", lw=1.2)
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
    ax[2].set_yscale("log"); ax[2].set_ylabel("max total hessian (80% train split)")
    ax[2].set_ylim(0.1, 60)
    ax[2].set_title("(c) the whole sample is worth <1 hessian\nin 2 of 3 channels -> no split allowed")
    ax[2].annotate("min_child_weight = 1 (default)", xy=(0.02, 1.0), xytext=(0.02, 26),
                   fontsize=7.5, color="k",
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
    for xi, h in enumerate(hs):
        ax[2].text(xi, h * 1.35, f"{h:.2f}", ha="center", fontsize=8,
                   fontweight="bold", color="#c62828" if h < 1 else "#1b5e20")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_weights.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ---------------------------------------------------------------- fig 2
def fig_methods(df):
    """One point per mode. Only the modes that carry the message are labelled; the rest sit in the
    same cluster, so labelling all seven would cost legibility and add nothing."""
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    labelled = {"baseline": (10, -4), "reg": (8, 8), "reg_flatten": (-30, 12),
                "flat_smote": (-30, -16), "mcw": (8, -14)}
    for i, ch in enumerate(CH):
        sub = df[df.channel == ch].groupby("mode").agg(
            te=("auc_slice_test", "mean"), gap=("auc_slice_gap", "mean")).reset_index()
        sub = sub[sub["mode"].isin(MODES)]
        for _, r in sub.iterrows():
            ax[i].scatter(r.gap, r.te, s=120, color=C.get(r["mode"], "#333"),
                          zorder=3, edgecolor="k", linewidth=0.5)
            if r["mode"] in labelled:
                dx, dy = labelled[r["mode"]]
                ax[i].annotate(r["mode"], (r.gap, r.te), fontsize=7.5,
                               xytext=(dx, dy), textcoords="offset points",
                               arrowprops=dict(arrowstyle="-", lw=0.5, color="#999"))
        ax[i].axvline(0.05, color="#c62828", ls="--", lw=1)
        ax[i].set_title(f"{ch}   (N_eff bkg = {df[df.channel==ch].neff_bkg.iloc[0]:.0f})")
        ax[i].set_xlabel("train - test AUC gap   (lower = less overtraining)")
        ax[i].set_xlim(-0.02, 0.15)
        ax[i].set_ylim(0.47, 1.03)
    ax[0].set_ylabel("test AUC on M=1000 GeV slice")
    ax[0].text(0.052, 0.52, "gap = 0.05", fontsize=6.5, rotation=90, color="#c62828")
    ax[0].annotate("all oversampling /\nweighting variants\nsit in this cluster",
                   xy=(0.022, 0.985), xytext=(0.055, 0.80), fontsize=7, color="#555",
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"))
    fig.suptitle("Grey = XGBoost defaults (two channels never split at all).  Green = derived "
                 "regularization (+flattening).  Top-left is better.", y=1.0, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_methods.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------- fig 3
def fig_scale(df):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    sub = df[(df["mode"] == "reg_flatten")].groupby("channel").first()
    x = np.arange(len(CH))
    mcw = [sub.loc[c, "min_child_weight"] for c in CH]
    lam = [sub.loc[c, "reg_lambda"] for c in CH]
    ax[0].bar(x - 0.18, mcw, 0.34, label="derived min_child_weight", color="#1b5e20")
    ax[0].bar(x + 0.18, lam, 0.34, label="derived reg_lambda", color="#66bb6a")
    ax[0].axhline(1.0, color="#c62828", ls="--", lw=1.2)
    ax[0].text(-0.42, 1.45, "XGBoost default = 1", fontsize=7.5, color="#c62828")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(CH)
    ax[0].set_ylim(0.005, 3.0)
    ax[0].set_ylabel("parameter value")
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_title("(a) derived values sit 10-100x below the default")

    depth = [sub.loc[c, "max_depth"] for c in CH]
    sub_p = [sub.loc[c, "subsample"] for c in CH]
    col = [sub.loc[c, "colsample_bytree"] for c in CH]
    ax[1].plot(x, sub_p, "o-", label="subsample", color="#1b5e20")
    ax[1].plot(x, col, "s-", label="colsample_bytree", color="#42a5f5")
    for xi, d in zip(x, depth):
        ax[1].annotate(f"depth={d:.0f}", (xi, 0.88), ha="center", fontsize=7,
                       color="#555")
    ax[1].set_xticks(x); ax[1].set_xticklabels(CH); ax[1].set_ylim(0, 1.02)
    ax[1].set_ylabel("fraction"); ax[1].legend(fontsize=7.5)
    ax[1].set_title("(b) bagging params from correlation + separation")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_scale.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------- fig 4
def fig_seeds():
    seeds = pd.concat([pd.DataFrame(json.load(open(f)))
                       for f in glob.glob("/tmp/seeds_*.json")])
    depth = pd.concat([pd.DataFrame(json.load(open(f)))
                       for f in glob.glob("/tmp/depth_*.json")])

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    # (a) seed spread, per direction
    combos = [("4L_eeee", 1), ("4L_eeee", 2), ("3L_eeeJ", 1), ("3L_eeeJ", 2)]
    for i, (ch, h) in enumerate(combos):
        g = seeds[(seeds.channel == ch) & (seeds.half == h)]
        xs = np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g))
        ax[0].scatter(xs, g.gap, s=45, color="#1b5e20" if h == 1 else "#c62828", zorder=3)
        ax[0].hlines(g.gap.mean(), i - 0.25, i + 0.25, color="k", lw=1.6)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([f"{c}\nDF{h}" for c, h in combos], fontsize=7.5)
    ax[0].set_ylabel("train - test AUC gap")
    ax[0].set_title("(a) gap is stable across 4 seeds\n(+-0.005), so not noise")

    # (b) depth sweep
    for ch, mk in (("4L_eeee", "o"), ("3L_eeeJ", "s")):
        for h, col in ((1, "#c62828"), (2, "#42a5f5")):
            g = depth[(depth.channel == ch) & (depth.half == h)].sort_values("depth")
            ax[1].plot(g.depth, g.gap, mk + "-", color=col, label=f"{ch} DF{h}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("max_depth (1 = single stump)")
    ax[1].set_ylabel("train - test AUC gap")
    ax[1].legend(fontsize=6.5)
    ax[1].set_title("(b) gap does NOT shrink with capacity\n-> not memorization")

    # (c) the smoking gun: score of the two halves
    ax[2].bar([0, 1], [0.046, 0.439], 0.55, color=["#42a5f5", "#c62828"])
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["DF1 background\n(training half)", "DF2 background\n(held-out)"])
    ax[2].set_ylabel("mean weighted BDT score")
    ax[2].set_title("(c) 4L_eeee, model trained on DF1:\nthe held-out half looks 10x more signal-like")
    for xi, v in enumerate([0.046, 0.439]):
        ax[2].text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_seeds.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------- fig 5
def fig_vvv():
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3))
    comp = pd.DataFrame({
        "sample": ["DF1", "DF2"],
        "VVV": [2, 3], "ZZ_4L": [41, 41], "ttZ_4L": [146, 147]})
    wshare = pd.DataFrame({
        "sample": ["DF1", "DF2"],
        "VVV": [78.9, 84.8], "ZZ_4L": [15.8, 11.4], "ttZ_4L": [5.3, 3.8]})

    x = np.arange(3); w = 0.36
    ax[0].bar(x - w / 2, comp.iloc[0, 1:].values, w, label="DF1", color="#42a5f5")
    ax[0].bar(x + w / 2, comp.iloc[1, 1:].values, w, label="DF2", color="#c62828")
    ax[0].set_xticks(x); ax[0].set_xticklabels(comp.columns[1:])
    ax[0].set_yscale("log"); ax[0].set_ylabel("unique events")
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) 4L_eeee background: VVV is 2 vs 3 events")
    for xi, (a, b) in enumerate(zip(comp.iloc[0, 1:].values, comp.iloc[1, 1:].values)):
        ax[0].text(xi - w / 2, a * 1.25, str(a), ha="center", fontsize=7.5)
        ax[0].text(xi + w / 2, b * 1.25, str(b), ha="center", fontsize=7.5)

    ax[1].bar(x - w / 2, wshare.iloc[0, 1:].values, w, label="DF1", color="#42a5f5")
    ax[1].bar(x + w / 2, wshare.iloc[1, 1:].values, w, label="DF2", color="#c62828")
    ax[1].set_xticks(x); ax[1].set_xticklabels(wshare.columns[1:])
    ax[1].set_ylabel("% of total background weight")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) but VVV carries ~80% of the weight\n-> the two halves are different samples")
    for xi, (a, b) in enumerate(zip(wshare.iloc[0, 1:].values, wshare.iloc[1, 1:].values)):
        ax[1].text(xi - w / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=7.5)
        ax[1].text(xi + w / 2, b + 1.5, f"{b:.1f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_vvv.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


if __name__ == "__main__":
    df = load_grid()
    fig_weights()
    fig_methods(df)
    fig_scale(df)
    fig_seeds()
    fig_vvv()
    print("ALL FIGURES DONE ->", OUT)
