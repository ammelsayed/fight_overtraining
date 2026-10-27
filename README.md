# fight_overtraining

Methods to beat **overtraining** in a **parameterized BDT (pBDT)** trained with XGBoost when the
signal samples are small, using **data-driven regularization** and **over-sampling** — with no
hyper-parameter optimization and no per-channel hand tuning.

---

## 1. The idea

A search for a resonance of unknown mass normally trains one classifier per mass hypothesis. That
costs a separate training set per point and gives no interpolation between them. Instead, a single
**parameterized** classifier is trained (the parameterized-network idea of Baldi et al. [2]):

- `signal_mass` is added as an input feature; each signal event carries its true generated mass.
- The background sample is **duplicated once per mass hypothesis**, each copy labelled with that
  hypothesis' mass.

One model then interpolates over the whole mass range and is evaluated at any mass by setting
`signal_mass` to that value. This is the strategy used in published ATLAS and CMS searches
[3, 4].

## 2. The problem

With the near-default XGBoost hyper-parameters (`max_depth=6`, `learning_rate=0.3`, no
`min_child_weight`/`reg_lambda` restrictions) the pBDT **overtrains**: the training-sample score
distribution separates far better than the held-out one, so the classifier is memorizing its
training events rather than learning the likelihood ratio.

The per-mass signal statistics are the limiting factor. In `sample_data/`:

| channel   | features | signal rows / half | signal rows per mass | unique bkg events / half |
|-----------|---------:|-------------------:|---------------------:|-------------------------:|
| `3L_eee`  |       18 |        12 467 / 12 477 |  539 – 679 | 3 843 |
| `3L_eeeJ` |       30 |         7 816 /  7 825 |  312 – 425 |   591 |
| `4L_eeee` |       19 |         1 343 /  1 355 |   45 –  82 |   189 |

Each channel provides two statistically independent half-samples, `sig_DF1/DF2` and
`bkg_DF1/DF2`. **Model 1 trains on DF1 and is evaluated on DF2, and vice versa** — a genuine
held-out test, not a random split of one sample.

### 2.1 A second, less obvious problem: the regularization parameters are in the wrong units

XGBoost's structural parameters (`min_child_weight`, `reg_lambda`, `gamma`) are expressed in
**hessian** units, not event counts. For the logistic loss the hessian of an event with physical
weight `w` is at most `0.25 w`. The event weights in this analysis are O(10⁻⁴)–O(10⁻²), so the
XGBoost defaults are not "mild regularization" — they are an absolute veto on splitting.

Measured on the balanced training sample (`bdt_methods.suggest_min_child_weight` diagnostics):

| channel   | Σw (balanced) | max total hessian | can split at `min_child_weight=1`? |
|-----------|--------------:|------------------:|:----------------------------------:|
| `3L_eee`  |        34.18  |             8.545 | yes |
| `3L_eeeJ` |         6.59  |             1.646 | **no** |
| `4L_eeee` |         0.99  |             0.248 | **no** |

`3L_eeeJ` and `4L_eeee` therefore train to `best_iteration = 0`: AUC = 0.500, KS distance = 0.000,
a constant classifier. This is not overtraining — it is a classifier that never splits a single
node. Any method that is supposed to "beat overtraining" must first make the leaf floor
**scale-invariant**, i.e. expressed in effective events rather than raw hessian.

### 2.2 Background weights are not homogeneous

The background is a sum of six processes whose weights are constant within a process but differ by
up to **1085×** between processes (`VVV` 2.3e-2 vs `ttZ_4L` 2.1e-5). A mean weight is therefore not
a typical event. The Kish effective number of events,

    N_eff = (Σ w)² / Σ w² ,

measures the statistics that are actually there. For the unique background events:

| channel   | unique bkg | N_eff (Kish) | N_eff / rows |
|-----------|-----------:|-------------:|-------------:|
| `3L_eee`  |      3 843 |        198   |        5.2 % |
| `3L_eeeJ` |        591 |         13   |        2.2 % |
| `4L_eeee` |        189 |          3   |        1.7 % |

In `4L_eeee` the background is worth about **three** events. This must be accounted for when
deriving any leaf floor, otherwise the floor is either vacuous or impossible to satisfy.

## 3. Strategy

Two orthogonal directions. **No HPO, no per-channel hand tuning**: every parameter is derived from
the sample itself by one rule that is identical for all three channels, and that is therefore
applicable to a new channel without intervention.

### 3.1 Data-driven regularization

Everything is expressed in the same currency — **effective events** — using the Kish weight
`w_eff = Σw²/Σw` of each class, so the rule is invariant to the absolute normalization of the
weights and to the process mixture.

- **`min_child_weight`** — a leaf must hold at least `n_min` *effective* events:
  `mcw = 0.25 · n_min · w_eff`. This replaces the current `mean(w)`-based estimate, which is
  dominated by the highest-weight process and ignores the spread. A feasibility cap
  (`mcw ≤ 0.25 · W_min_class / n_leaves`) guarantees the minority class can still split and that
  the sample supports at least a minimum number of leaves.
- **`reg_lambda`** — the L2 term is an additive hessian, so it is expressed in the same currency:
  a leaf weight is shrunk as if backed by `n_pseudo` further effective events.
- **`max_depth`** — a backstop derived from the hessian budget,
  `max_depth = ceil(log2(H_total / mcw))`, i.e. the deepest *balanced* tree the sample can populate
  without violating the leaf floor.
- **`learning_rate` + early stopping** — a fixed moderate learning rate; the number of trees is
  decided by early stopping on the held-out validation split (already implemented).
- **`gamma`** — left at zero: it is a further absolute-hessian threshold and would duplicate
  `min_child_weight`.

### 3.2 Data-driven bagging parameters

- **`colsample_bytree`** — from the correlation structure: the features are strongly correlated
  (max |ρ| = 0.94), so the number of statistically independent directions is the **effective rank**
  of the correlation matrix (participation ratio of its eigenvalues). Each tree then sees as many
  features as there are independent directions, rather than an arbitrary fraction.
- **`subsample`** — from the separation power: the more cleanly the features separate signal from
  background, the more the model can memorize individual events, so the stronger the row
  sub-sampling.

### 3.3 Over-sampling

Over-sampling is applied to the **training half only** (never to DF2, never to the validation
split), **per truth-mass hypothesis**, so no unphysical cross-mass mixtures are created. Two
methods are implemented and compared:

- **Bagging (resampling with replacement)** — the resampling method documented for BDT training in
  TMVA [5]. This is the HEP-standard technique.
- **SMOTE (synthetic minority over-sampling)** [6] — synthetic events built as convex combinations
  of two real events of the *same* truth mass, found by k-nearest-neighbour search in the
  standardized feature space, never random noise.

**A note on what ATLAS and CMS actually do.** A full-text search of INSPIRE for `SMOTE` restricted
to ATLAS or to CMS returns **no records**; the documented practice is class balancing through event
weights and under-sampling, not synthetic minority generation [7, 8]. The HEP analysis that does use
SMOTE [9] is a simulation study, not a collaboration paper. We therefore treat SMOTE as the
*comparison* arm and bagging as the *baseline* HEP method, and we report which one actually reduces
the train/test gap rather than assuming either.

### 3.4 Background weight treatment

Because the background spans 1085× in weight (§2.2), three training-side treatments are compared:
keep the physical weights (only the leaf floor absorbs the heterogeneity); flatten each process to
an equal total weight during training; and clip extreme weights. This is an empirical question and
the answer is reported, not assumed.

## 4. Validation protocol

For every channel, every direction (DF1→DF2 and DF2→DF1) and every configuration:

1. train on one half, evaluate on the other;
2. run the TMVA-style overtraining check [5] on **one signal-mass hypothesis** (M = 1000 GeV) only:

   ```python
   bdt1.check_overtraining(
       df_sig_train=sig_DF1.loc[sig_DF1[new_col] == 1000],
       df_bkg_train=bkg_DF1.loc[bkg_DF1[new_col] == 1000],
       df_sig_test=sig_DF2.loc[sig_DF2[new_col] == 1000],
       df_bkg_test=bkg_DF2.loc[bkg_DF2[new_col] == 1000],
   )
   ```

   which reports the Kolmogorov–Smirnov probability and distance and the χ² probability between the
   train and test score distributions, separately for signal and background, weighted and unweighted;
3. record the train/test AUC gap on the mass slice, the KS p-values and the **test AUC**. Killing
   overtraining by destroying the classifier is not a result: the test AUC must survive.

## 5. References

1. T. Chen, C. Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016, [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
2. P. Baldi, K. Cranmer, T. Faucett, P. Sadowski, D. Whiteson, *Parameterized neural networks for high-energy physics*, Eur. Phys. J. C 76 (2016) 235, [arXiv:1601.07913](https://arxiv.org/abs/1601.07913)
3. ATLAS Collaboration, *Search for Higgs boson decays into a Z boson and a light hadronically decaying resonance*, [arXiv:2405.04914](https://arxiv.org/abs/2405.04914)
4. CMS Collaboration, *Search for a pseudoscalar boson decaying to four photons*, [arXiv:2208.01469](https://arxiv.org/abs/2208.01469)
5. A. Hoecker et al., *TMVA — Toolkit for Multivariate Data Analysis*, CERN-OPEN-2007-007, [arXiv:physics/0703039](https://arxiv.org/abs/physics/0703039)
6. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, *SMOTE: Synthetic Minority Over-sampling Technique*, J. Artif. Intell. Res. 16 (2002) 321, [arXiv:1106.1813](https://arxiv.org/abs/1106.1813)
7. C. W. Murphy, *Class Imbalance Techniques for High Energy Physics*, SciPost Phys. 7 (2019) 076, [arXiv:1905.00339](https://arxiv.org/abs/1905.00339)
8. ATLAS Collaboration, *Dijet resonance search with weak supervision using √s = 13 TeV pp collisions*, Phys. Rev. Lett. 125 (2020) 131801, [arXiv:2005.02983](https://arxiv.org/abs/2005.02983)
9. R. Sahoo, K. Goswami, S. Prasad, *Machine learning driven identification of heavy flavor decay leptons in pp collisions at the LHC*, Phys. Rev. D 113 (2026) 094025, [arXiv:2509.00712](https://arxiv.org/abs/2509.00712)
10. L. Kish, *Survey Sampling*, Wiley (1965) — the effective sample size used in §2.2 and §3.1
11. ATLAS Collaboration, *Fiducial and differential cross-sections for VBF H→WW\*→eνμν*, Phys. Rev. D 108 (2023) 072003, [arXiv:2304.03053](https://arxiv.org/abs/2304.03053) — example of the KS train/test overtraining check

## 6. Status

Results and findings will be added here after the study.
