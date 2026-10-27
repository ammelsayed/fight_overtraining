"""Two additional figures for the README.

  fig4_treatment    the method comparison: what actually moves test AUC
  fig7_overtraining the TMVA-style overtraining check itself, one panel per channel
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
from PIL import Image

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
ROOT = "/data/ammelsayed/fight_overtraining"
OUT = f"{ROOT}/figures"
CH = ["3L_eee", "3L_eeeJ", "4L_eeee"]

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


# ---------------------------------------------------------------- fig 6
def fig_treatment():
    rows = [json.load(open(f)) for f in glob.glob(f"{ROOT}/results_json/*.json")]
    df = pd.DataFrame(rows)

    arms = [("reg", "physical\nweights", "#90caf9"),
            ("reg_flatten", "flatten\nper process", "#1b5e20"),
            ("flat_bagging", "+ bagging", "#2e7d32"),
            ("flat_smote", "+ SMOTE", "#66bb6a")]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    # (a) test AUC, grouped by channel
    x = np.arange(len(CH)); w = 0.2
    for j, (mode, lab, col) in enumerate(arms):
        vals = [df[(df.channel == c) & (df["mode"] == mode)].auc_all_test.mean() for c in CH]
        ax[0].bar(x + (j - 1.5) * w, vals, w, label=lab, color=col)
        for xi, v in zip(x, vals):
            ax[0].text(xi + (j - 1.5) * w, v + 0.004, f"{v:.3f}", ha="center",
                       fontsize=5.8, rotation=90)
    ax[0].set_xticks(x); ax[0].set_xticklabels(CH)
    ax[0].set_ylim(0.90, 1.02)
    ax[0].set_ylabel("test AUC (all masses)")
    ax[0].legend(fontsize=7, ncol=2, loc="lower right")
    ax[0].set_title("(a) flattening is the jump; over-sampling is marginal")

    # (b) the same four arms on the M=1000 slice gap
    for j, (mode, lab, col) in enumerate(arms):
        vals = [df[(df.channel == c) & (df["mode"] == mode)].auc_slice_gap.mean() for c in CH]
        ax[1].bar(x + (j - 1.5) * w, vals, w, label=lab.replace("\n", " "), color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CH)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("train - test AUC gap")
    ax[1].set_title("(b) the gap does shrink, but see Fig. 5:\nit is composition, not memorization")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_treatment.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


# ---------------------------------------------------------------- fig 7
def fig_overtraining():
    """Stack the three check_overtraining outputs of the recommended configuration."""
    imgs = []
    for ch in CH:
        p = f"{ROOT}/results/reg_flatten/{ch}/DF1/score_distributions.png"
        if not os.path.exists(p):
            print("MISSING", p)
            return
        imgs.append(Image.open(p))
    w = min(i.width for i in imgs)
    imgs = [i.resize((w, int(i.height * w / i.width))) for i in imgs]
    total_h = sum(i.height for i in imgs)
    canvas = Image.new("RGB", (w, total_h), "white")
    y = 0
    for i in imgs:
        canvas.paste(i, (0, y)); y += i.height
    canvas.save(f"{OUT}/fig7_overtraining.png")
    print("fig7 done", canvas.size)


if __name__ == "__main__":
    fig_treatment()
    fig_overtraining()
    print("DONE")
