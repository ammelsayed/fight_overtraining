# fight_overtraining

Methods to beat **overtraining** in a **parameterized BDT (pBDT)** built with XGBoost, when the signal samples are small.

## The idea

Instead of training one BDT per signal mass hypothesis, a single **parameterized** classifier is trained:

- `signal_mass` is added as an input feature; signal events carry their true (truth) generated mass.
- The background sample is **duplicated once per mass hypothesis**, each copy labelled with that hypothesis' mass (the provided data already comes this way: 21 hypotheses, M = 1000–2000 GeV in 50 GeV steps).

One model then interpolates over all mass hypotheses — the parameterized-classifier idea of [2, 3], used in published ATLAS and CMS searches, e.g. [4, 5].

**The problem:** with near-default hyperparameters (`max_depth=6`, `learning_rate=0.3`, no `min_child_weight` / `reg_lambda` / `subsample` restrictions), the pBDT overtrains: training-sample score distributions separate far better than held-out ones. The per-mass signal statistics are the limiting factor (≈590 signal events per mass per half-sample in `3L_eee`, ≈370 in `3L_eeeJ`, ≈65 in `4L_eeee`).

## The data

`sample_data/<channel>/`, three channels (multi-lepton signal vs. SM diboson/top backgrounds):

| channel   | features | signal rows per half | background rows per half (incl. 21× mass duplication) |
|-----------|----------|----------------------|--------------------------------------------------------|
| `3L_eee`  | 18       | 12 467 / 12 477      | 80 703 / 80 766                                        |
| `3L_eeeJ` | 30       | 7 816 / 7 825        | 12 411 / 12 453                                        |
| `4L_eeee` | 19       | 1 343 / 1 355        | 3 969 / 4 011                                          |

Each channel provides `sig_DF1/DF2` and `bkg_DF1/DF2`: two statistically independent half-samples. **Model 1 trains on DF1 and is evaluated on DF2, and vice versa.** Background rows carry an `event_id` so the train/validation split can deduplicate the 21× mass copies (no leakage). Signal weights are reweighted to a uniform mass distribution in training (already implemented in `bdt_methods.py`), and class weights are balanced in `train()`.

## Strategy

Two orthogonal directions. **No HPO, no per-channel hand tuning**: every hyperparameter is derived from the sample itself by the same rule for all channels.

### 1. Data-driven regularization

- **`min_child_weight`** — extend the existing `suggest_min_child_weight()` logic: a leaf must represent a fixed minimum number of equivalent background events (`mcw = min_events_in_leaf × 0.25 × ⟨w⟩`, the maximum Hessian per event under the logistic loss), so trees cannot split on statistical noise.
- **`max_depth`** — bounded by statistics instead of being free: `max_depth = floor(log2(W_train / mcw))`, so even a fully grown tree cannot create leaves below the statistical floor.
- **L2 regularization (`reg_lambda`)** — expressed in the same "pseudo-event" currency: `reg_lambda = 0.25 × ⟨w⟩ × n_pseudo`, i.e. leaf weights are shrunk as if backed by at least `n_pseudo` events.
- **`learning_rate` + early stopping** — a fixed moderate learning rate; the number of trees is decided by early stopping on the held-out validation split (already in the class), not by hand.
- **`subsample` / `colsample_bytree`** — fixed standard values (0.8) for stochastic regularization.

### 2. Over-sampling (following HEP practice)

Naive duplication or adding random noise is not acceptable. The method follows the SMOTE family [6] — the widely adopted "wise" oversampling in HEP-ML studies [7, 8]: **synthetic signal events are created as convex combinations of two real events of the same truth mass** (k-nearest-neighbour interpolation in feature space), never as random noise:

- oversample **signal only** (the minority class), **per truth-mass point**, so no unphysical cross-mass mixtures are created; synthetic events inherit the parents' `signal_mass` and an averaged weight;
- only the **training half** is oversampled; DF2 and all test samples stay untouched;
- neighbours are searched within the training sample only (no leakage), and class balancing (`balance_weights()`) is applied after oversampling.

For completeness: ATLAS and CMS most commonly treat class imbalance through event reweighting and under-sampling, and use resolution-smearing in fast simulation [10]; SMOTE-type interpolation is the oversampling method established in the HEP-ML literature [7, 8]. Both it and the reweighting-based approach will be compared.

### Validation protocol

For every channel and every configuration (baseline / regularized / oversampled / combined):

1. train `bdt1` on DF1 and evaluate on DF2; train `bdt2` on DF2 and evaluate on DF1;
2. for **one signal mass hypothesis** (M = 1000 GeV) run the TMVA-style overtraining check [9]:
   ```python
   bdt1.check_overtraining(
       df_sig_train=sig_DF1.loc[sig_DF1["signal_mass"] == 1000],
       df_bkg_train=bkg_DF1.loc[bkg_DF1["signal_mass"] == 1000],
       df_sig_test=sig_DF2.loc[sig_DF2["signal_mass"] == 1000],
       df_bkg_test=bkg_DF2.loc[bkg_DF2["signal_mass"] == 1000],
   )
   ```
3. record the train/test AUC gap on the mass slice, the KS p-values, and the test AUC — generalization must not be sacrificed in order to kill overtraining;
4. the same code path and the same data-driven formulas run unchanged on all three channels.

Results and findings will be added here after the study.

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
