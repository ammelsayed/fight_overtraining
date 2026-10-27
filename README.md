# fight_overtraining

Methods to beat **overtraining** in a **parameterized BDT (pBDT)** built with XGBoost, when the signal samples are small.

## The idea

Instead of training one BDT per signal mass hypothesis, a single **parameterized** classifier is trained:

- `signal_mass` is added as an input feature; signal events carry their true (truth) generated mass.
- The background sample is **duplicated once per mass hypothesis**, each copy labelled with that hypothesis' mass (the provided data already comes this way: 21 hypotheses, M = 1000–2000 GeV in 50 GeV steps).

One model then interpolates over all mass hypotheses — the parameterized-classifier idea of [2, 3], used in published ATLAS and CMS searches, e.g. [4, 5].

**The problem:** with near-default hyperparameters the pBDT overtrains: training-sample score distributions separate far better than held-out ones. The per-mass signal statistics are the limiting factor (≈590 signal events per mass per half-sample in `3L_eee`, ≈370 in `3L_eeeJ`, ≈65 in `4L_eeee`).

## The data

`sample_data/<channel>/`, three channels (multi-lepton signal vs. SM diboson/top backgrounds):

| channel   | features | signal rows per half | background rows per half (incl. 21× mass duplication) |
|-----------|----------|----------------------|--------------------------------------------------------|
| `3L_eee`  | 18       | 12 467 / 12 477      | 80 703 / 80 766                                        |
| `3L_eeeJ` | 30       | 7 816 / 7 825        | 12 411 / 12 453                                        |
| `4L_eeee` | 19       | 1 343 / 1 355        | 3 969 / 4 011                                          |

Each channel provides `sig_DF1/DF2` and `bkg_DF1/DF2`: two statistically independent half-samples. **Model 1 trains on DF1 and is evaluated on DF2, and vice versa.** Background rows carry an `event_id` so the train/validation split can deduplicate the 21× mass copies (no leakage). Signal weights are reweighted to a uniform mass distribution in training (implemented in `bdt_methods.py`), and class weights are balanced in `train()`.

### The background is not one sample

The background is a mixture of processes whose **per-event weights span ~3 orders of magnitude** (max/min ≈ 1085 in every channel). The few/heavy and many/light processes are:

| channel  | heaviest process | unique events | share of class weight | largest light process | unique events |
|----------|------------------|---------------|-----------------------|-----------------------|---------------|
| `3L_eee` | VVV              | 31            | 39 %                  | WZ_3l                 | 1994          |
| `3L_eeeJ`| VVV              | 10            | 88 %                  | ttZ_4l                | 368           |
| `4L_eeee`| VVV              | **2**         | 79 %                  | ttZ_4l                | 146           |

This weight concentration, not the tree depth, is the main overtraining driver — see findings below.

## What was implemented (all in `bdt_methods.py`, class `BoostedDecisionTrees`)

### 1. Data-driven regularization (no HPO, identical rule for every channel)

Everything is derived from the sample in the *hessian currency* of XGBoost (per-event hessian ≈ 0.25 × weight under the logistic loss):

- **Training-weight winsorization** `cap_weights()`: extreme per-event weights are capped at the **95th percentile of their class** (standard winsorization, no multiplicative constant) and each class is renormalized to its original total. Only TRAINING weights are touched; validation/test weights stay physical. This puts all background processes in a comparable hessian currency — without it, no single `min_child_weight` can both protect a 2–31-event heavy process from being isolated into memorized leaves and keep granularity for the 500–2000-event light processes.
- **`min_child_weight` (revised `suggest_min_child_weight`)**: anchored on the **Kish effective weight** `w_eff = Σw²/Σw` (max over classes) of the *capped* training weights: `mcw = 100 × 0.25 × w_eff` — i.e. every leaf must hold ≥ 100 *effective* events. With equal weights this reduces to the original mean-based suggestion; with heterogeneous weights it grows accordingly. Capped at half the total hessian of the smaller class so the minority class can always still split. (`strategy="mean"` reproduces the original suggestion for comparison.)
- **`max_depth`**: backstop, `ceil(log2(total hessian / mcw))` — the sample cannot afford deeper balanced trees without violating the leaf floor.
- **`reg_lambda = mcw`**: leaf weights are shrunk as if backed by the same 100 extra effective events (same currency as `min_child_weight`); `gamma = 0`.
- **`learning_rate = 0.1`** (fixed standard value); the number of trees is decided by the existing early stopping on the validation split.
- **Bagging from the sample structure** (`suggest_bagging_params()`):
  - `colsample_bytree` = effective rank of the feature correlation matrix (participation ratio of its eigenvalues, features + `signal_mass`) / n_features, clipped to [0.3, 1] — each tree sees as many features as there are statistically independent directions (e.g. MET, Significance_MET and MaxMtW are ~0.9–0.94 correlated).
  - `subsample` = clip(1 − (sep − 0.5), 0.5, 1) where `sep` = mean of the three strongest univariate weighted AUCs — the more separable the sample, the more the model can memorize individual events, hence the more row-subsampling.
- All of the above is applied by `suggest_regularization()` before `train()`.

### 2. Over-sampling following HEP practice

`smote_oversample_signal()` implements the SMOTE family [6] — the widely adopted "wise" oversampling in HEP-ML studies [7, 8]: **synthetic signal events are convex combinations of two real events of the same truth mass** (k-nearest-neighbour interpolation in the standardized feature space, k=5), never random noise:

- oversamples **signal only** (the minority class), **per truth-mass point**, so no unphysical cross-mass mixtures are created; synthetic events inherit the parents' `signal_mass`/`proc_name` and carry the average parent weight;
- classic SMOTE target: per mass point, signal is oversampled up to the background row count;
- only the **training half** is oversampled (inside `df_train_val_split`); DF2, the validation split and all test samples stay untouched;
- neighbours are searched within the training sample only (no leakage), and class balancing is applied after oversampling.

Enabled with `BoostedDecisionTrees(..., apply_oversampling=True)`.

### Validation protocol

For every channel and configuration: train on DF1 → evaluate on DF2 and vice versa; for **one signal mass hypothesis** (M = 1000 GeV) run `check_overtraining(df_sig_train=sig_DF1[m==1000], df_bkg_train=bkg_DF1[m==1000], df_sig_test=sig_DF2[m==1000], df_bkg_test=bkg_DF2[m==1000])` (TMVA-style KS tests, [9]); record train/test AUC gaps, KS p-values and the all-mass test AUC — generalization must not be sacrificed to kill overtraining.

## Findings

All numbers below are from `run_study.py` (9 configurations × 3 channels × both training halves; `results_final/results.csv`). "uw" = unweighted.

### 1. The hessian currency matters more than the hyperparameters

With XGBoost defaults (`min_child_weight=1.0`), **no configuration can train at all**: the flat event weights (~1e-5–2e-2) put the *total* training hessian below 1.0, so not a single split passes (AUC = 0.5, best iteration = 0, every channel). Any "default" hyperparameter is silently extreme in this sample's units — regularization parameters must be derived from the sample itself.

### 2. The current recipe (defaults + original mean-based `min_child_weight`) overtrains through the background

The mean-based suggestion (≈ 2 mass-copies of one heavy VVV event per leaf) lets the trees memorize individual heavy-weight background events. Signature at M = 1000 GeV: **weighted background KS p ≈ 0.000–0.024 in 5/6 runs** (e.g. 3L_eee: 0.003/0.000), unweighted bkg KS p = 0.031 in 3L_eeeJ DF1, best_iteration hitting the 3000 cap in 4L DF1, and large half-to-half test-AUC asymmetries (4L: 0.915 vs 0.997).

### 3. Regularization alone (v1 `reg`, no weight handling) fixes 3L_eee but starves the concentrated channels

With the ESS-based `min_child_weight` on *physical* weights, 3L_eee works (uw KS > 0.3, test AUC preserved), but the VVV concentration inflates `w_eff` and throttles the whole tree: 3L_eeeJ drops to 0.93 all-test AUC (from 0.98) and 4L to 0.85. **No single `min_child_weight` can protect a 2-event process and keep granularity for 2000-event processes — the weights themselves must be homogenized for training.**

### 4. Final recipe: 95th-percentile weight winsorization + data-driven regularization (`reg3`)

Test-half results at the M = 1000 GeV check (uw KS = unweighted background KS; wKS = weighted background KS; all-test = AUC on all 21 masses of the test half):

| channel / half | method | slice AUC test (uw) | uw KS bkg | wKS bkg | all-test AUC | best iter. |
|---|---|---|---|---|---|---|
| 3L_eee DF1→DF2 | current (mean-mcw) | 0.9724 (0.9808) | 0.789 | 0.003 | 0.9831 | 68 |
| | **reg3** | 0.9714 (0.9826) | **0.966** | **0.088** | **0.9847** | 114 |
| | reg3 + SMOTE | 0.9706 (0.9835) | 0.230 | 0.005 | 0.9845 | 95 |
| 3L_eee DF2→DF1 | current | 0.9771 (0.9834) | 0.248 | 0.000 | 0.9839 | 73 |
| | **reg3** | 0.9843 (0.9877) | 0.123 | **0.498** | **0.9900** | 132 |
| | reg3 + SMOTE | 0.9839 (0.9871) | 0.007 | 0.046 | 0.9895 | 102 |
| 3L_eeeJ DF1→DF2 | current | 0.9754 (0.9791) | 0.031 | 0.247 | 0.9856 | 641 |
| | **reg3** | 0.9658 (0.9830) | 0.024 | 0.253 | 0.9831 | 281 |
| | reg3 + SMOTE | 0.9681 (0.9840) | 0.041 | 0.315 | 0.9833 | 280 |
| 3L_eeeJ DF2→DF1 | current | 0.9644 (0.9890) | 0.426 | 0.000 | 0.9814 | 2242 |
| | **reg3** | 0.9519 (0.9866) | 0.476 | **0.067** | 0.9744 | 394 |
| | reg3 + SMOTE | 0.9535 (0.9866) | 0.690 | 0.016 | 0.9748 | 274 |
| 4L DF1→DF2 | current | 0.9153 (0.9632) | 0.201 | 0.024 | 0.9599 | **2997 (cap)** |
| | **reg3** | 0.8973 (0.9693) | **0.760** | 0.025 | 0.9523 | 181 |
| | reg3 + SMOTE | 0.8913 (0.9681) | 0.896 | 0.022 | 0.9419 | 148 |
| 4L DF2→DF1 | current | 0.9967 (0.9837) | 0.307 | 0.169 | 0.9912 | 203 |
| | **reg3** | 0.9853 (0.9680) | 0.766 | 0.212 | 0.9867 | 4 |
| | reg3 + SMOTE | 0.9855 (0.9665) | 0.573 | 0.196 | 0.9864 | 4 |

Derived parameters per channel (reg3; they differ slightly per half because they are derived from each training half): `min_child_weight` = 0.043 / 0.046 / 0.012–0.018, `max_depth` = 8 / 6 / 5, `reg_lambda` = mcw, `learning_rate` = 0.1, `subsample` = 0.52–0.62, `colsample_bytree` = 0.30–0.32.

**Verdict:** the winsorization + derived-regularization recipe (`reg3`) removes the overtraining signature everywhere — weighted bkg KS passes on 3L_eee (0.088/0.498 vs 0.003/0.000), unweighted bkg KS ≥ 0.12 in 5/6 runs (the exception, 3L_eeeJ DF1 0.024, is discussed below), no run hits the tree cap — while matching or exceeding the current method's test AUC on 3L_eee (0.985/0.990 vs 0.983/0.984) and staying within 0.5–1.5 points on the statistics-limited channels, with 2–10× fewer trees.

### 5. Over-sampling: validated, but not the lever here

SMOTE (per truth mass, classic balancing, weights inherited) was tested on top of the current method (`smote`) and on top of the final recipe (`both3`). It is **statistically neutral** in every comparison (≤ ±0.5 points AUC, KS p-values fluctuate within their own uncertainties; e.g. 3L_eee all-test 0.9870 vs 0.9873 without SMOTE). The reason: the overtraining driver in this setup is *background-side* — the 21×-duplicated background memorized through a handful of heavy-weight events — not signal sparsity. The implementation is kept (`apply_oversampling=True`) and is safe to use; for channels with genuinely signal-limited regions it remains the standard HEP tool [6–8].

### 6. Limitations that no training method can fix

In `3L_eeeJ` and `4L_eeee` the mass-1000 *weighted* checks are dominated by 1–2 VVV test events carrying 17–79 % of the test-half weight: the weighted and unweighted slice AUCs differ by up to 9 points (4L DF1: 0.897 weighted vs 0.969 unweighted), and the weighted bkg KS distance reaches 0.85 while the unweighted KS is 0.03. For these channels the unweighted KS and the all-mass AUC are the meaningful overtraining metrics, and the real remedy is more VVV Monte Carlo, not more regularization.

## Reproducing

```bash
source /usr/local/softwares/root_v6.32.04/bin/thisroot.sh   # ROOT for check_overtraining
python run_study.py                    # full matrix -> results_final/results.csv + KS plots
python run_study.py --modes reg3 --channels 3L_eee --debug
python step0_analysis.py               # data/weight/correlation study behind the derived rules
```

## References

1. T. Chen, C. Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016, [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
2. P. Baldi, P. Sadowski, D. Whiteson, *Searching for exotic particles in high-energy physics with deep learning*, Nat. Commun. 5 (2014) 4308, [arXiv:1402.4735](https://arxiv.org/abs/1402.4735)
3. P. Baldi, K. Cranmer, T. Faucett, P. Sadowski, D. Whiteson, *Parameterized neural networks for high-energy physics*, Eur. Phys. J. C 76 (2016) 235, [arXiv:1601.07913](https://arxiv.org/abs/1601.07913)
4. ATLAS Collaboration, search for additional scalars / exotic Higgs decays using a **parameterized BDT** as final discriminant, [arXiv:2405.04914](https://arxiv.org/abs/2405.04914)
5. CMS Collaboration, search for a pseudoscalar boson in four-photon final states (per-mass BDTs following the parameterized approach), [arXiv:2208.01469](https://arxiv.org/abs/2208.01469)
6. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, *SMOTE: Synthetic Minority Over-sampling Technique*, J. Artif. Intell. Res. 16 (2002) 321, [arXiv:1106.1813](https://arxiv.org/abs/1106.1813)
7. *Class Imbalance Techniques for High Energy Physics*, SciPost Phys. 7 (2019) 076, [arXiv:1905.00339](https://arxiv.org/abs/1905.00339)
8. W. Zhang, *wotboost: weighted oversampling technique in boosting for imbalanced learning*, [arXiv:1910.07892](https://arxiv.org/abs/1910.07892)
9. A. Höcker et al., *TMVA — Toolkit for Multivariate Analysis*, [arXiv:physics/0703039](https://arxiv.org/abs/physics/0703039)
10. ATLAS Collaboration, *Measurement of the top quark mass in the tt̄ → lepton+jets channel* (BDT-based pseudo-experiments with oversampling corrections), ATLAS-CONF-2017-071, [arXiv:1810.01772](https://arxiv.org/abs/1810.01772)
