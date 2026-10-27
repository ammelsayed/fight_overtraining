"""Is the residual train/test KS tension real overtraining, or KS sensitivity at large N?

Control: compare the model's train-vs-test KS distance against the KS distance between two
statistically independent samples that the model has NEVER seen. If D(train, test) is compatible
with D(unseen, unseen), the classifier adds no train-specific structure and the p-value is driven
by the sample size, not by memorization.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, "/data/ammelsayed/fight_overtraining")
from bdt_methods import BoostedDecisionTrees, to_Xyw

BASE = "/data/ammelsayed/fight_overtraining/sample_data"
MASS, M = "signal_mass", 1000
rng = np.random.RandomState(7)


def load(ch, h):
    d = f"{BASE}/{ch}"
    return (pd.read_parquet(f"{d}/sig_DF{h}.parquet"),
            pd.read_parquet(f"{d}/bkg_DF{h}.parquet"),
            json.load(open(f"{d}/features_SR.json")))


def boot_D(a, b, n=400):
    """Bootstrap distribution of the KS statistic under the null (same parent distribution)."""
    pool = np.concatenate([a, b])
    ds = []
    for _ in range(n):
        rng.shuffle(pool)
        ds.append(ks_2samp(pool[:len(a)], pool[len(a):]).statistic)
    return np.array(ds)


for ch in ["3L_eee", "3L_eeeJ", "4L_eeee"]:
    sig, bkg, feats = load(ch, 1)
    sig_te, bkg_te, _ = load(ch, 2)
    bdt = BoostedDecisionTrees(sig, bkg, feats + [MASS], weight_treatment="flatten")
    bdt.suggest_regularization()
    bdt.train()

    b_tr = bkg[bkg[MASS] == M]
    b_te = bkg_te[bkg_te[MASS] == M]
    Xtr, _, _ = to_Xyw(b_tr.iloc[:0], b_tr, bdt.features, "flat_weight")
    Xte, _, _ = to_Xyw(b_te.iloc[:0], b_te, bdt.features, "flat_weight")
    ptr, pte = bdt.predict(Xtr), bdt.predict(Xte)

    D = ks_2samp(ptr, pte)
    # control: split the unseen test half in two, both never used in training
    idx = rng.permutation(len(pte))
    half = len(pte) // 2
    Dctl = ks_2samp(pte[idx[:half]], pte[idx[half:2 * half]])

    # scale the control to the train-vs-test sample sizes so the comparison is fair
    # (KS statistic scales like 1/sqrt(N_eff)); report both raw and size-matched
    n1, n2 = len(ptr), len(pte)
    n_ctl = half
    scale = np.sqrt((1.0 / n1 + 1.0 / n2) / (2.0 / n_ctl))
    null = boot_D(ptr, pte, n=200)

    print(f"{ch}: train n={n1} test n={n2}")
    print(f"   D(train,test)            = {D.statistic:.4f}  p={D.pvalue:.3g}")
    print(f"   D(testA,testB) [control] = {Dctl.statistic:.4f}  p={Dctl.pvalue:.3g}"
          f"   (size-matched ~{Dctl.statistic*scale:.4f})")
    print(f"   bootstrap null median D  = {np.median(null):.4f}"
          f"  95th pct = {np.percentile(null,95):.4f}")
    verdict = "NO evidence of train-specific structure" if D.statistic <= np.percentile(null, 95) \
        else "tension exceeds the null"
    print(f"   -> {verdict}\n")
