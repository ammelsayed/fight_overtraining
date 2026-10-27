import os
import sys
import ROOT
import joblib
import pickle
import json
import shap
import optuna
import scienceplots
import subprocess
import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import make_scorer
from scipy.stats import ks_2samp
from tabulate import tabulate
from scipy import stats
# plt.style.use('science')
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")

# Optional
from metrics import expected_significance_binned
from plotting import plot_feature

# ---------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------- #

def setup_logging(level=logging.INFO, log_file=None):
    logger = logging.getLogger("stat_inference")
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

logger = setup_logging(level=logging.INFO)

# ---------------------------------------------------------------- #
# Helper methods
# ---------------------------------------------------------------- #

# Check if cuda is supported
def cuda_is_available():
    """Run nvidia-smi and hide the output, return True if nvidia-smi exited with code 0 (success)"""
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
        return result.returncode == 0
    except FileNotFoundError:
        return False

def get_weights_info(w):
    if len(w) == 0: return "empty"
    else: return f"total={np.sum(w):.6e}, min={np.min(w):.6e}, max={np.max(w):.6e}, mean={np.mean(w):.6e}, std={np.std(w):.6e}"

def to_Xyw(df_sig, df_bkg, features, weight_col = "weight"):
    X_sig = df_sig[features].values.astype(np.float32)
    X_bkg = df_bkg[features].values.astype(np.float32)
    w_sig = df_sig[weight_col].values.astype(np.float64)
    w_bkg = df_bkg[weight_col].values.astype(np.float64)
    X = np.vstack([X_sig, X_bkg])
    y = np.array([1] * len(X_sig) + [0] * len(X_bkg), dtype=np.float32)
    w = np.concatenate([w_sig, w_bkg])
    return X, y, w

def split_df(df, first_fraction=0.5, *, correct_weights = True, group_col="proc_name", weight_col="flat_weight", random_state=42):

    # basic checks
    if not 0.0 < first_fraction < 1.0:
        raise ValueError("first_fraction must be strictly between 0.0 and 1.0")
    if group_col not in df.columns:
        raise KeyError(f"Required stratification column '{group_col}' is missing")
    if weight_col not in df.columns:
        raise KeyError(f"Required weight column '{weight_col}' is missing")
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    rng = np.random.RandomState(random_state)
    fst_parts, snd_parts = [],  []
    for proc, group_df in df.groupby(group_col):

        if len(group_df) < 2:
            logger.warning(f"Process {proc!r} has only 1 event; removing it from the dataframe.")
            df = df[df[group_col] != proc]
            continue

        group_seed = int(rng.randint(0, np.iinfo(np.int32).max))
        df1, df2 = train_test_split(
            group_df, 
            train_size=first_fraction, 
            random_state=group_seed, 
            shuffle=True
        )

        # Correct weights of the first and second part
        if correct_weights:
            df1 = df1.copy()
            corr = len(group_df) / len(df1) 
            df1[weight_col] *= corr

            df2 = df2.copy()
            corr = len(group_df) / len(df2) 
            df2[weight_col] *= corr
            
        fst_parts.append(df1)
        snd_parts.append(df2)

    df_fst = pd.concat(fst_parts, ignore_index=True)
    df_snd = pd.concat(snd_parts, ignore_index=True)

    return df_fst, df_snd
		
def reweight_to_uniform_distributions(df, group_col="proc_name", weight_col="flat_weight"):
    """
    Notice if the signal dataset is composed of only one group,
    then this weights modification is not applied. Hence, this 
    method is effictive only if we are using a pBDT
    """
    target = df.groupby(group_col)[weight_col].sum().mean()
    parts = []
    for proc, group in df.groupby(group_col):
        group = group.copy()                                   
        w_tot = group[weight_col].sum()
        group[weight_col] *= target / w_tot                   
        parts.append(group)
    df_uniform = pd.concat(parts, ignore_index=True)
    return df_uniform

def smote_oversample_signal(df_sig, features, n_synth_per_mass, k=5, random_state=42,
                            weight_col="flat_weight", mass_col="signal_mass"):
    """
    SMOTE-style over-sampling of the signal (minority) class, applied per truth-mass point:
    synthetic events are convex combinations of two real events of the SAME truth mass
    (k-nearest-neighbour interpolation in the standardized feature space, Chawla et al. 2002),
    never random noise. Synthetic events inherit the parents' signal_mass, proc_name and carry
    the average parent weight. Non-feature discriminator columns are copied from the first parent.
    Only the caller decides which samples are oversampled (training samples only, never test).
    """
    rng = np.random.RandomState(random_state)
    kin_features = [f for f in features if f != mass_col]
    other_cols = [c for c in df_sig.columns if c not in kin_features + [mass_col, weight_col]]

    X = df_sig[kin_features].values.astype(float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    pos = {idx: i for i, idx in enumerate(df_sig.index)}

    synth = []
    for mass, grp in df_sig.groupby(mass_col):
        rows = [pos[i] for i in grp.index]
        n = len(rows)
        if n < 2:
            continue
        k_g = min(k, n - 1)
        n_target = int(n_synth_per_mass.get(mass, np.mean(list(n_synth_per_mass.values())))) \
            if isinstance(n_synth_per_mass, dict) else int(n_synth_per_mass)
        if n_target <= 0:
            continue
        # nearest neighbours within the same truth mass (no cross-mass interpolation)
        d = np.sqrt(((Xz[rows][:, None, :] - Xz[rows][None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = np.argsort(d, axis=1)[:, :k_g]
        for _ in range(n_target):
            a = rng.randint(n)
            b = nn[a, rng.randint(k_g)]
            lam = rng.uniform(0.0, 1.0)
            xz_new = lam * Xz[rows[a]] + (1.0 - lam) * Xz[rows[b]]
            row = df_sig.iloc[rows[a]].copy()
            row[kin_features] = xz_new * sd + mu
            row[mass_col] = mass
            row[weight_col] = 0.5 * (df_sig[weight_col].iloc[rows[a]] + df_sig[weight_col].iloc[rows[b]])
            synth.append(row)

    if not synth:
        return df_sig
    df_synth = pd.DataFrame(synth).reset_index(drop=True)
    return pd.concat([df_sig, df_synth], ignore_index=True)


def df_train_val_split(df_sig, df_bkg, features, train_size=0.80, weight_col = "flat_weight",
                       oversample_signal=False, smote_k=5, smote_target="balance"):
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1")

    df_sig_train, df_sig_val = split_df(df_sig, first_fraction=train_size)

    if 'event_id' in df_bkg.columns:
        # Split based on unique event IDs to prevent leakage from duplicated events
        unique_ids = df_bkg['event_id'].unique()

        # Use a fixed seed so the split is reproducible
        rng = np.random.RandomState(42)
        rng.shuffle(unique_ids)

        n_train = int(len(unique_ids) * train_size)
        train_ids = unique_ids[:n_train]

        df_bkg_train = df_bkg[df_bkg['event_id'].isin(train_ids)].copy()
        df_bkg_val = df_bkg[~df_bkg['event_id'].isin(train_ids)].copy()
    else:
        # Fallback for standard (non-parametrized) training
        df_bkg_train, df_bkg_val = split_df(df_bkg, first_fraction=train_size)

    # Reweight all the signal samples to have a uniform truth mass distribution in the training.
    df_sig_train = reweight_to_uniform_distributions(df_sig_train)

    # Over-sample the training signal only (never the validation/test samples): per truth mass,
    # synthetic SMOTE events are added. Default target: classic SMOTE balancing, i.e. the number
    # of background rows per mass hypothesis.
    if oversample_signal:
        if "signal_mass" in df_sig_train.columns and "signal_mass" in df_bkg_train.columns:
            n_synth = (df_bkg_train.groupby("signal_mass").size() - df_sig_train.groupby("signal_mass").size()).fillna(0).clip(lower=0)
            n_synth_per_mass = {m: int(v) for m, v in n_synth.items()}
        else:
            n_synth_per_mass = max(len(df_bkg_train) - len(df_sig_train), 0)
        if smote_target == "balance":
            target = n_synth_per_mass
        else:
            f = float(smote_target)
            target = {m: int(f * g) for m, g in df_sig_train.groupby("signal_mass").size().items()} \
                if isinstance(n_synth_per_mass, dict) else int(f * len(df_sig_train))
        df_sig_train = smote_oversample_signal(df_sig_train, features, target, k=smote_k)

    # Convert
    X_all, y_all, w_all = to_Xyw(df_sig, df_bkg, features, weight_col)
    X_train, y_train, w_train = to_Xyw(df_sig_train, df_bkg_train, features, weight_col)
    X_val, y_val, w_val = to_Xyw(df_sig_val, df_bkg_val, features, weight_col)
    
    # Prepare an organized output
    dataset = {"all" :{}, "train": {}, "val": {}}
    dataset["all"]["X"], dataset["all"]["y"], dataset["all"]["w"] = X_all, y_all, w_all
    dataset["train"]["X"], dataset["train"]["y"], dataset["train"]["w"] = X_train, y_train, w_train
    dataset["val"]["X"], dataset["val"]["y"], dataset["val"]["w"] = X_val, y_val, w_val
    dataset["all"]["w_model"] = None
    dataset["train"]["w_model"] = None
    dataset["val"]["w_model"] = None
    return dataset

# ---------------------------------------------------------------- #
# Asimov Calculation Methods
# ---------------------------------------------------------------- #

def asimov_significance(s, b, b_err=0.0):
    """Discovery significance with background uncertainty."""
    if s <= 0.0:
        return 0.0
    if b <= 0.0:
        b = 1e-10  # small placeholder to avoid division by zero
        b_err = max(b_err, 1e-10)

    # No background uncertainty: Cowan et al. asymptotic formula
    if b_err <= 0.0:
        val = 2.0 * ((s + b) * math.log1p(s / b) - s)
        return math.sqrt(max(val, 0.0))

    sb2 = b_err * b_err
    denom = b * b + (s + b) * sb2
    if denom <= 0.0:
        return 0.0

    term1 = (s + b) * math.log(((s + b) * (b + sb2)) / denom)
    term2 = (b * b / sb2) * math.log(1.0 + (sb2 * s) / (b * (b + sb2)))
    val = 2.0 * (term1 - term2)
    return math.sqrt(max(val, 0.0))

vec_asimov = np.vectorize(asimov_significance)

def get_counts_and_errors(model, X, y, w=None, nbins = 30):
    y_pred_prob = model.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1.0, nbins + 1)
    if w is None:
        w = np.array([1] * len(X), dtype=np.float32)
    counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
    counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
    sum_w2_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0]**2)
    errors_bkg = np.sqrt(sum_w2_bkg)
    return counts_sig, counts_bkg, errors_bkg, _

def expected_significance_binned(model, X, y, w, nbins = 30):
    counts_sig, counts_bkg, errors_bkg, _ = get_counts_and_errors(model, X, y, w, nbins = nbins)
    return np.sqrt(np.sum(vec_asimov(s=counts_sig, b=counts_bkg, b_err=errors_bkg)**2))

def expected_significance_unbinned(model, X, y, w):
    """
    Calculates the unbinned (max-cut) Asimov significance.
    Evaluates the Cowan significance for every possible BDT score threshold
    and returns the maximum.
    """
    y_pred = model.predict_proba(X)[:, 1]
    
    # Sort events by BDT score in descending order
    sort_idx = np.argsort(-y_pred)
    y_sorted = y[sort_idx]
    w_sorted = w[sort_idx]
    
    # Cumulative sums from the highest score downwards
    cum_s = np.cumsum(np.where(y_sorted == 1, w_sorted, 0.0))
    cum_b = np.cumsum(np.where(y_sorted == 0, w_sorted, 0.0))
    
    # Optional: Cumulative statistical error on background
    cum_b_err = np.sqrt(np.cumsum(np.where(y_sorted == 0, w_sorted**2, 0.0)))
    
    # Calculate Asimov significance for every threshold
    Z = vec_asimov(s=cum_s, b=cum_b, b_err=cum_b_err)
    
    # Return the best significance
    return np.max(Z)


# ---------------------------------------------------------------- #
# BDT Class
# ---------------------------------------------------------------- #

class BoostedDecisionTrees:

    def __init__(self, df_sig, df_bkg, features, apply_oversampling=False, smote_k=5, smote_target="balance"):

        self.df_sig = df_sig
        self.df_bkg = df_bkg
        self.features = features
        # Oversampling options (SMOTE on the training signal only; must be set before the split is built)
        self.apply_oversampling = apply_oversampling
        self.smote_k = smote_k
        self.smote_target = smote_target
        self.ds = df_train_val_split(df_sig, df_bkg, features,
                                     oversample_signal=apply_oversampling,
                                     smote_k=smote_k, smote_target=smote_target)
        
        # Model options
        # see https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
        # see https://xgboost.readthedocs.io/en/latest/parameter.html
        # see https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
        self.params = {
            "device"                : "cuda" if cuda_is_available() else "cpu", 
            "tree_method"           : "hist",
            "verbosity"             : 0,  # default: 1, 0 (silent), 1 (warning), 2 (info), 3 (debug)         
            "random_state"          : 42,
            "nthread"               : os.cpu_count(),

            "objective"             : "binary:logistic",

            "enable_categorical"    : True,

            # early stopping is informed by area under precision-recall, which directly measures 
            # signal-class precision = background rejection at the high-score end where the leak occurs.  
            # (Note: XGBoost uses the last listed metric for early stopping.)
            "n_estimators"          : int(3000),  # default=100, range: [1,∞]
            "early_stopping_rounds" : 20,
            "eval_metric"           : ["logloss"], 

            # "max_depth"             : 3,     # default=6, range: [0,∞], 0 indicates no limit on depth
            # "learning_rate"         : 0.03,   # default=0.3, range: [0,1]

            # "min_child_weight"      : 1.0,   # default=1, range: (0,∞]
            # "scale_pos_weight"      : 1.0,

            # "subsample"             : 0.8,   # default=1, range: (0,1]
            # "colsample_bytree"      : 0.8,   # default=1, range: (0,1]

            # # L1 and L2 regularization terms on weights.
            # # Increasing these value will make model more conservative.
            # "reg_lambda"            : 10.0,   # default=1, range: [0, ∞]
            # "reg_alpha"             : 0.0,   # default=0, range: [0, ∞]
            # "gamma"                 : 1.0,    # default=0, range: [0,∞]

        }

        self.model = xgb.XGBClassifier(**self.params)

        # standarization optins
        # see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        self.apply_standardization = False
        self.scaler = StandardScaler()

        # Weights balancing options
        self.apply_weights_balancing = True

        # hop
        self.hpo_study = None

        # Other options
        self.debug = False
        self.use_min_child_weight_suggestion = False

        from xgboost import __version__ as xgb_version
        print(f"Using XGBoost Version {xgb_version}")

    def get_model(self):
        return self.model

    def update_params(self,**kwargs):
        self.params.update(kwargs)
        self.model = xgb.XGBClassifier(**self.params)
        return self

    def avg_std_min_max(self, arr, text=True):
        Avg, Std, Min, Max = np.mean(arr), np.std(arr), np.min(arr), np.max(arr)
        if text:
            return f"min={Min:.6e}, max={Max:.6e}, mean={Avg:.6e}, std={Std:.6e}"
        else:
            return Avg, Std, Min, Max

    def print_features(self):
        table = [[f, self.avg_std_min_max(self.df_sig[f]), self.avg_std_min_max(self.df_bkg[f])] for f in self.features]
        print(f"Using the following {len(self.features)} features for training:")
        print(tabulate(table, headers=["Name", "Signal", "Background"], tablefmt="fancy_grid"))
        return None

    def print_params(self):
        table = [[k, "-", v] for k, v in self.params.items()]
        print("Using the following XGBClassifier paramaters:")
        print(tabulate(table, headers=["Param", "Default", "Employed"], tablefmt="fancy_grid"))
        return None

    def update_features_names(self):
        # make sure self.features is a list of (str) names
        self.model.get_booster().feature_names = self.features
        return None

    def balance_weights(self, y, w, weighting_strategy = "physical-weights"):
        """
        Two weighting strategies are employed. 
        In the weighted scheme, each event is assigned its generator-level physical weight, 
        and these weights are subsequently scaled to achieve class balance. 
        In the unweighted scheme, the physical weights are discarded and each event receives a uniform weight of unity. 
        In both strategies, class balancing is enforced, 
        thereby guaranteeing that the total sum of weights for signal events equals that for background events.
        """
        sum_sig = np.sum(w[y == 1])
        sum_bkg = np.sum(w[y == 0])
        n_sig = len(w[y == 1])
        n_bkg = len(w[y == 0])

        if sum_sig == 0 or sum_bkg == 0:
            raise ValueError("One class has zero total physics weight.")

        if weighting_strategy == "class-only":
            w_tot = compute_sample_weight(class_weight="balanced", y=y)

        if weighting_strategy == "physical-weights":
            factor_sig = (sum_sig + sum_bkg) / (2 * sum_sig)
            factor_bkg = (sum_sig + sum_bkg) / (2 * sum_bkg)
            w_class = np.where(y == 1, factor_sig, factor_bkg)
            w_tot = w * w_class

        if self.debug:
            print("After balancing weights:")
            print(f"  Signal weights: {get_weights_info(w_tot[y == 1])}")
            print(f"  Background weights: {get_weights_info(w_tot[y == 0])}")

        # Optional sanity check
        if not np.isclose(np.sum(w_tot[y == 1]), np.sum(w_tot[y == 0]), rtol=1e-8):
            raise RuntimeError("Balanced sums are not equal -> numerical issue.")

        return w_tot

    def suggest_min_child_weight(self, w, y, min_events_in_leaf=100, debug = False):
        """
        Revised for heterogeneous per-process weights. The per-event weights span ~3 orders of
        magnitude across the background processes, so a mean-weight-based threshold allowed leaves
        holding only a couple of heavy-weight (e.g. VVV) events -> memorization -> overtraining.
        Instead the threshold is anchored on the Kish effective weight w_eff = sum(w^2)/sum(w):
        - with equal weights, w_eff == mean(w) (the old behaviour);
        - with heterogeneous weights, w_eff > mean(w) and a leaf at the threshold carries on
          average at least min_events_in_leaf *effective* (Kish) events of MC statistics.
        The suggestion is capped at half the total hessian of the smaller class, so the minority
        class can always still split.
        """
        w = np.asarray(w, dtype=float)
        w_eff = 0.0
        for c in (0, 1):
            wc = w[y == c]
            if wc.sum() > 0:
                w_eff = max(w_eff, (wc ** 2).sum() / wc.sum())

        mcw_suggested = min_events_in_leaf * 0.25 * w_eff

        # Absolute maximum safe limit: total hessian of the smaller class
        min_class_weight = min(np.sum(w[y == 1]), np.sum(w[y == 0]))
        mcw_max_safe_limit = 0.25 * min_class_weight
        mcw_suggested = min(mcw_suggested, 0.5 * mcw_max_safe_limit)

        if self.debug or debug:
            print(f"Diagnostics for the 'min_child_weight' paramater (using all samples):")
            print(f"  > Min/Max/Average event weight  : {np.min(w):.2e} / {np.max(w):.2e} / {np.mean(w):.2e}")
            print(f"  > Effective weight (Kish)       : {w_eff:.2e} (sum(w^2)/sum(w), max over classes)")
            print(f"  > Suggested min_child_weight    : {mcw_suggested:.2e} (>= {min_events_in_leaf} effective events per leaf, capped at 0.5 * safe limit)")
            print(f"  > Absolute Maximum safe limit   : {mcw_max_safe_limit:.2e} (never exceed this, otherwise minority class can NEVER split!)")

        return mcw_suggested, mcw_max_safe_limit

    def suggest_bagging_params(self, sep_topk=3, subsample_floor=0.5, colsample_floor=0.3, debug=False):
        """
        Data-driven bagging (no HPO), derived from the sample itself:
        - colsample_bytree: the correlation structure of the (kinematic features + signal_mass) is
          compressed to its effective rank (participation ratio of the correlation-matrix
          eigenvalues); each tree then sees as many features as there are statistically
          independent directions in the sample.
        - subsample: the stronger the kinematic features separate signal from background (mean of
          the sep_topk strongest univariate weighted AUCs), the more the model can memorize
          individual events, hence the more row-subsampling is applied:
          subsample = clip(1 - (sep - 0.5), subsample_floor, 1).
        """
        kin = [f for f in self.features if f != "signal_mass"]

        # effective rank of the correlation matrix (features as used by the model, i.e. incl. signal_mass)
        X = pd.concat([self.df_sig[self.features], self.df_bkg[self.features]]).astype(float)
        good = X.std()[(X.std() > 0)].index
        C = np.corrcoef(X[good].values.T)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        n_eff = ev.sum() ** 2 / (ev ** 2).sum()
        colsample_bytree = float(np.clip(n_eff / len(self.features), colsample_floor, 1.0))

        # univariate separation (kinematic features only; unique background events)
        ub = self.df_bkg.drop_duplicates(subset="event_id") if "event_id" in self.df_bkg.columns else self.df_bkg
        y = np.r_[np.ones(len(self.df_sig)), np.zeros(len(ub))]
        wv = np.r_[self.df_sig["flat_weight"].values, ub["flat_weight"].values]
        aucs = []
        for f in kin:
            a = roc_auc_score(y, np.r_[self.df_sig[f].values, ub[f].values], sample_weight=wv)
            aucs.append(max(a, 1.0 - a))
        sep = float(np.mean(sorted(aucs)[-sep_topk:]))
        subsample = float(np.clip(1.0 - (sep - 0.5), subsample_floor, 1.0))

        if debug:
            print("Diagnostics for the bagging parameters:")
            print(f"  > Features (n={len(self.features)}): effective rank {n_eff:.2f} -> colsample_bytree = {colsample_bytree:.3f}")
            print(f"  > Univariate separation (top-{sep_topk} mean weighted AUC) = {sep:.3f} -> subsample = {subsample:.3f}")

        return {"colsample_bytree": colsample_bytree, "subsample": subsample, "sep_auc": sep, "n_eff_features": n_eff}

    def suggest_regularization(self, min_events_in_leaf=100, learning_rate=0.1, debug=False):
        """
        Derives the full regularization set from the sample itself (no HPO, identical rule for
        every channel). Everything is expressed in the same 'pseudo-event' currency:
          min_child_weight : ESS-based leaf statistical floor, see suggest_min_child_weight
          max_depth        : backstop, ceil(log2(total hessian / mcw)); the sample cannot afford
                             deeper balanced trees without violating the leaf statistical floor
          reg_lambda       : leaf weights are shrunk as if backed by min_events_in_leaf extra
                             effective events (same currency as min_child_weight)
          learning_rate    : fixed standard value; the number of trees is decided by early stopping
          subsample / colsample_bytree : from feature separation / correlations,
                             see suggest_bagging_params
        """
        y_train, w_train = self.ds["train"]["y"], self.ds["train"]["w"]
        w_bal = self.balance_weights(y_train, w_train)

        mcw, mcw_safe = self.suggest_min_child_weight(w_bal, y_train, min_events_in_leaf=min_events_in_leaf, debug=debug)
        h_tot = 0.25 * w_bal.sum()
        max_depth = max(3, int(np.ceil(np.log2(max(h_tot / mcw, 2.0)))))
        bag = self.suggest_bagging_params(debug=debug)

        params = {
            "min_child_weight" : float(mcw),
            "max_depth"        : int(max_depth),
            "reg_lambda"       : float(mcw),
            "gamma"            : 0.0,
            "learning_rate"    : float(learning_rate),
            "subsample"        : bag["subsample"],
            "colsample_bytree" : bag["colsample_bytree"],
        }
        self.update_params(**params)

        print("Derived regularization parameters (data-driven, no HPO):")
        print(f"  > min_child_weight = {mcw:.4g} (leaf floor: {min_events_in_leaf} effective events, safe limit {mcw_safe:.4g})")
        print(f"  > max_depth = {max_depth} (hessian budget {h_tot:.4g} / mcw = {h_tot/mcw:.1f} leaves)")
        print(f"  > reg_lambda = {mcw:.4g}, gamma = 0.0, learning_rate = {learning_rate}")
        print(f"  > subsample = {bag['subsample']:.3f}, colsample_bytree = {bag['colsample_bytree']:.3f} "
              f"(sep AUC {bag['sep_auc']:.3f}, effective rank {bag['n_eff_features']:.2f})")

        # Per-process consequence of the leaf floor (background processes carry very different weights)
        if debug:
            factor_bkg = w_bal[y_train == 0].sum() / self.ds["train"]["w"][y_train == 0].sum()
            rows = []
            for proc, grp in self.df_bkg.groupby("proc_name"):
                w_p = grp["flat_weight"].mean() * factor_bkg
                rows.append((proc, mcw / (0.25 * w_p)))
            print("  > Implied minimum rows per leaf, per background process:")
            for proc, n_rows in rows:
                print(f"      {proc:<10s}: {n_rows:,.0f} rows ({n_rows/21:.0f} unique events)")

        return params

    def train(self):

        X_train, y_train, w_train = self.ds["train"]["X"], self.ds["train"]["y"], self.ds["train"]["w"]
        X_val, y_val, w_val = self.ds["val"]["X"], self.ds["val"]["y"], self.ds["val"]["w"]

        if self.apply_standardization:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        # Keep validation weights physical/representative. Class balancing is
        # part of the training objective, not a correction to yields.
        if self.apply_weights_balancing:
            w_train = self.balance_weights(y_train, w_train)
            w_val = self.balance_weights(y_val, w_val)
        
        if self.use_min_child_weight_suggestion:
            self.update_params(min_child_weight = self.suggest_min_child_weight(w_train, y_train)[0])

        self.print_features()
        self.print_params()

        self.model.fit(
            X_train, y_train,
            sample_weight = w_train,
            eval_set = [(X_train, y_train), (X_val, y_val)],
            sample_weight_eval_set = [w_train, w_val],
            verbose = self.params["early_stopping_rounds"]
        )

        best_it = getattr(self.model, "best_iteration", None)
        if self.params.get("early_stopping_rounds") is not None and best_it is not None:
            print(f'Best score {self.model.best_score}, Best iteration {best_it}')
        else:
            print(f'Final fit used {self.params["n_estimators"]} estimators, with early stopping disabled.')
        
        self.update_features_names()
        return self

    def _hpo_evaluate(self, model, X_val, y_val, w_val, metric="auc"):
        if metric == "auc":
            y_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred, sample_weight=w_val)

        if metric.startswith("significance"):
            binning = metric.split("_")[1]  # "5bins", "3bins", or "unbinned"
            if binning == "unbinned":
                return float(expected_significance_unbinned(model, X_val, y_val, w_val))

            n_bins = int(binning.replace("bins", ""))
            return float(expected_significance_binned(model, X_val, y_val, w_val, nbins = n_bins))

        raise ValueError(f"Unknown metric: {metric}")
   
    def train_HPO_CV(self, n_trials=30, n_folds=3, metric="auc", direction="maximize"):

        print("Starting hyperparameter optimisation (HOP) ...")
        print(f" > Optimization Objective : {metric}.")
        print(f" > Optimization Direction : {direction}.") 
        print(f" > Optimization Trials    : {n_trials}.") 
        print(f" > Evaluation Strategy    : {n_folds}-fold CV approach.") 

        X_all, y_all, w_all = self.ds["all"]["X"], self.ds["all"]["y"], self.ds["all"]["w"]

        # Tune the scale-sensitive min_child_weight using the same balanced
        # training-weight convention used inside each fold, while retaining
        # physical weights for every validation fold.
        w_all_for_training = self.balance_weights(y_all, w_all) if self.apply_weights_balancing else w_all
    
        if self.use_min_child_weight_suggestion:
            mcw_suggested, mcw_max_safe = self.suggest_min_child_weight(w_all_for_training, y_all)
        else:
            mcw_suggested, mcw_max_safe = 1e-8, 1e2

        def objective(trial):
            params = {
                'verbosity': 0, # scilent wile tuning'
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 10.0),
            }
            
            kf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=42)
            fold_scores = []
            fold_n_estimators = []
            for train_idx, val_idx in kf.split(X_all, y_all):
                X_fold_train, X_fold_val = X_all[train_idx], X_all[val_idx]
                y_fold_train, y_fold_val = y_all[train_idx], y_all[val_idx]
                w_fold_train_physical = w_all[train_idx]
                w_fold_val = w_all[val_idx]

                w_fold_train = (
                    self.balance_weights(y_fold_train, w_fold_train_physical)
                    if self.apply_weights_balancing
                    else w_fold_train_physical
                )

                # Important: Standardization must be done INSIDE the fold to avoid data leakage
                if self.apply_standardization:
                    scaler = StandardScaler()
                    X_fold_train = scaler.fit_transform(X_fold_train)
                    X_fold_val = scaler.transform(X_fold_val)

                model = xgb.XGBClassifier(**{**self.params, **params})
                model.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=w_fold_train,
                    eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
                    sample_weight_eval_set=[w_fold_train, w_fold_val],
                    verbose=False
                )
                score = self._hpo_evaluate(model, X_fold_val, y_fold_val, w_fold_val, metric)
                fold_scores.append(score)
                best_iteration = getattr(model, "best_iteration", None)
                if best_iteration is None:
                    best_iteration = self.params["n_estimators"] - 1
                fold_n_estimators.append(int(best_iteration) + 1)
                
            trial.set_user_attr("fold_n_estimators", fold_n_estimators)
            trial.set_user_attr("n_estimators", int(np.rint(np.mean(fold_n_estimators))))
            return np.mean(fold_scores)
        
        self.hpo_study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        self.hpo_study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
       
        best_params = self.hpo_study.best_params
        best_n_estimators = self.hpo_study.best_trial.user_attrs["n_estimators"]
        self.update_params(
            **best_params,
            n_estimators=int(best_n_estimators),
            early_stopping_rounds=None,
        )
        self.train()
        return self
    
    def train_HPO(self, n_trials=30, metric="auc", direction="maximize"):

        print("Starting hyperparameter optimisation (HOP) ...")
        print(f" > Optimization Objective : {metric}.")
        print(f" > Optimization Direction : {direction}.") 
        print(f" > Optimization Trials    : {n_trials}.") 

        X_train, y_train, w_train = self.ds["train"]["X"], self.ds["train"]["y"], self.ds["train"]["w"]
        X_val, y_val, w_val = self.ds["val"]["X"], self.ds["val"]["y"], self.ds["val"]["w"]

        if self.apply_standardization:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        if self.apply_weights_balancing:
            w_train = self.balance_weights(y_train, w_train)
    
        if self.use_min_child_weight_suggestion:
            mcw_suggested, mcw_max_safe = self.suggest_min_child_weight(w_train, y_train)
        else:
            mcw_suggested, mcw_max_safe = 1e-8, 1e2

        def objective(trial):
            params = {
                'verbosity': 0, # scilent wile tuning'
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 10.0),
            }
            
            model = xgb.XGBClassifier(**{**self.params, **params})
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                sample_weight_eval_set=[w_train, w_val],
                verbose=False
            )
            score = self._hpo_evaluate(model, X_val, y_val, w_val, metric)
            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is None:
                best_iteration = self.params["n_estimators"] - 1
            trial.set_user_attr("n_estimators", int(best_iteration) + 1)
            return score
        
        self.hpo_study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        self.hpo_study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
       
        best_params = self.hpo_study.best_params
        best_n_estimators = self.hpo_study.best_trial.user_attrs["n_estimators"]
        self.update_params(
            **best_params,
            n_estimators=int(best_n_estimators),
            early_stopping_rounds=None,
        )
        self.train()
        return self

    def train_HPO_NoEarlyStopping(self, n_trials=30, metric="auc", direction="maximize"):

        print("Starting hyperparameter optimisation (HOP) with no early stopping ...")
        print(f" > Optimization Objective : {metric}.")
        print(f" > Optimization Direction : {direction}.") 
        print(f" > Optimization Trials    : {n_trials}.")

        self.update_params(early_stopping_rounds=None)

        X_train, y_train, w_train = self.ds["train"]["X"], self.ds["train"]["y"], self.ds["train"]["w"]
        X_val, y_val, w_val = self.ds["val"]["X"], self.ds["val"]["y"], self.ds["val"]["w"]

        if self.apply_standardization:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        if self.apply_weights_balancing:
            w_train = self.balance_weights(y_train, w_train)

        def objective(trial):
            params = {
                'verbosity': 0, # scilent wile tuning'
                'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 10.0),
            }
            
            model = xgb.XGBClassifier(**{**self.params, **params})
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                sample_weight_eval_set=[w_train, w_val],
                verbose=False
            )
            score = self._hpo_evaluate(model, X_val, y_val, w_val, metric)
            return score
        
        self.hpo_study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        self.hpo_study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
       
        best_params = self.hpo_study.best_params
        self.update_params(**best_params)
        self.train()
        return self
    
    def save_hpo_visualizations(self, output_dir):
        """Save Optuna's standard interactive Plotly diagnostics as standalone HTML."""
        if self.hpo_study is None:
            print("[WARNING] HPO was never run. Please run .train_HPO() before calling this method.")
            return None
                    
        from optuna import visualization as vis
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = {
            "optimization_history": vis.plot_optimization_history(self.hpo_study),
            "parameter_importances": vis.plot_param_importances(self.hpo_study),
            "parallel_coordinate": vis.plot_parallel_coordinate(self.hpo_study),
            "slice": vis.plot_slice(self.hpo_study),
            "contour": vis.plot_contour(self.hpo_study),
        }
        for name, figure in plots.items():
            path = output_dir / f"{name}.html"
            figure.write_html(str(path), include_plotlyjs=True, full_html=True)
  
        with (output_dir / "hpo_study.pkl").open("wb") as handle:
            pickle.dump(self.hpo_study, handle)
   
        return None

    def save_hyperparameters(self, path):
        """Save the booster's effective config. `path` is the .joblib path."""
        path = Path(path)
        json_file = path.with_name(path.stem + "_hyperparamaters.json")
        config = json.loads(self.model.get_booster().save_config())
        with open(json_file, "w") as f:
            json.dump(config, f, indent=2)

    def save(self, path):
        path = Path(path)
        if path.suffix != ".joblib":
            raise ValueError(f"Path must end with '.joblib', got: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(path))
        self.model.save_model(str(path.with_suffix(".ubj")))
        self.save_hyperparameters(path)
        print(f"Model saved to {path}")

    def load(self, path):
        path = Path(path)
        if path.suffix != ".joblib":
            raise ValueError(f"Path must end with '.joblib', got: {path}")
        if not path.exists():
            raise FileNotFoundError(f"No model file at {path}")
        self.model = joblib.load(path)
        return self

    @classmethod
    def load_from(cls, path):
        """Prediction-only constructor: skip __init__ (which requires dataframes)."""
        obj = cls.__new__(cls)
        obj.apply_standardization = False
        obj.scaler = StandardScaler()
        obj.features = None          # caller must set before predict()
        obj.load(path)
        return obj

    def predict(self, X):
        if self.apply_standardization:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]
    
    def predict_class(self, X):
        y_pred_prob = self.predict(X)
        return (y_pred_prob >= 0.5).astype(int)

    def evaluate_AUC(self, X, y, w):
        # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        y_pred = self.predict(X)
        return roc_auc_score(y, y_pred, sample_weight = w)

    def evaluate_significance(self, X, y, w):
        y_pred = self.predict(X)
        return significance_vscore(y, y_pred, sample_weight = w)

    def evaluate_asimov_significance(self, df_sig, df_bkg):
        """Max Asimov significance for given signal/background dataframes."""
        X, y, w = to_Xyw(df_sig, df_bkg, self.features)
        return float(np.max(self.evaluate_significance(X, y, w)))

    def get_score_histogram_counts_and_errors(self, X = None, y = None, w = None, nbins = 30, *, df_sig=None, df_bkg=None):
        bins = np.linspace(0, 1.0, nbins + 1)
        if all(v is None for v in (X, y, w)) and all(v is not None for v in (df_sig, df_bkg)):
            X, y, w = to_Xyw(df_sig, df_bkg, self.features)
        if w is None:
            w = np.array([1] * len(X), dtype=np.float32)

        y_pred_prob = self.predict(X)
        counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
        sum_w2_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1]**2)
        errors_sig    = np.sqrt(sum_w2_sig)
        counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
        sum_w2_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0]**2)
        errors_bkg    = np.sqrt(sum_w2_bkg)
        return {
            "counts_sig": counts_sig, "errors_sig": errors_sig,
            "counts_bkg": counts_bkg, "errors_bkg": errors_bkg,
            "bin_edges": _
        }
    
    def get_ps_value(self, X_test = None, y_test = None, w_test = None):
        """
        if X_test not given, return p-values for the Train vs Val distributions
        for the signal and background
        if X_test is given, return p-values for the Train vs X_test
        """
        # KS test (independent of weights)
        # Use Kolmogorov-Smirnov test to compare train and val distributions
        # For alternative='two-sided', the null hypothesis is that the two distributions are identical,
        # the alternative is that they are not identical. 
        # The statistic is the maximum absolute difference between the empirical distribution functions of the samples.
        # if ks_sig.pvalue < 0.05, the the two distributions are not identical (at least not drawn from the same distribution)

        # probabily should consider a weighted ks test or the Anderson-Darling test
        
        if all(v is not None for v in (X_test, y_test)):
            y_pred_prob = self.predict(X_test)
            y = y_test
        else:
            y_pred_prob = self.predict(self.ds["val"]["X"])
            y = self.ds["val"]["y"]

        y_pred_prob_train = self.predict(self.ds["train"]["X"])
        y_train = self.ds["train"]["y"]

        ks_sig = ks_2samp(y_pred_prob_train[y_train==1], y_pred_prob[y==1], alternative='two-sided', method='exact')
        ks_bkg = ks_2samp(y_pred_prob_train[y_train==0], y_pred_prob[y==0], alternative='two-sided', method='exact')
        pvalue_sig = ks_sig.pvalue
        pvalue_bkg = ks_bkg.pvalue
        # fpvalue_sig = f"{pvalue_sig:.2e}" if pvalue_sig < 0.01 else f"{pvalue_sig:.3f}"
        # fpvalue_bkg = f"{pvalue_bkg:.2e}" if pvalue_bkg < 0.01 else f"{pvalue_bkg:.3f}"
        # ks_results = f"Signal (Background) KS p-value: {fpvalue_sig} ({fpvalue_bkg})"
        return pvalue_sig, pvalue_bkg

    def print_features_ranking(self, importance_type="gain"):
        print(f"Top ranking features by {importance_type.title()}")
        results = self.model.get_booster().get_score(importance_type=importance_type)
        df = pd.DataFrame(list(results.items()), columns=['Feature Name', 'Importance'])
        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
        return None

    def plot_metric_curves(self, output_dir):
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        results = self.model.evals_result()
        metrics = set().union(*[list(d.keys()) for d in results.values()])
        for metric in metrics:
            try:
                plt.figure(figsize=(6, 5))
                if len(results) > 1:
                    plt.plot(results['validation_0'][metric], label=f'Train {metric.title()}')
                    plt.plot(results['validation_1'][metric], label=f'Validation {metric.title()}')
                else:
                    plt.plot(results['validation_0'][metric], label=f'Validation {metric.title()}')
                plt.xlabel("Boosting Rounds")
                plt.ylabel(metric.title())
                plt.legend()
                plt.grid(True)
                safe_name = metric.replace('/', '_')
                plt.savefig(os.path.join(output_dir, f"{safe_name}_curve.png"), dpi=600)
                plt.close()
            except Exception as e:
                print(f"Error: {e}. Skip plotting.")

    def evaluate_model(self, X_test = None, y_test = None, w_test = None, *, df_sig=None, df_bkg=None):
        """
        Preform internal eveluation using both the training and validation datasets
        If provided, report eveluation for a test dataset as well.
        """
        print("Eveluation report for the trained model.")

        # provide possible ways to get the test dataset
        if all(v is None for v in (X_test, y_test, w_test)) and all(v is not None for v in (df_sig, df_bkg)):
            X_test, y_test, w_test = to_Xyw(df_sig, df_bkg, self.features)
        
        # (1) print the classification_report from sklearn.metrics for each dataset
        # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        def short_report(X, y, w, *, label):
            print(f"Classification report for the '{label}' dataset:")
            y_pred = self.predict_class(X)
            print(classification_report(y, y_pred, labels = [1,0], target_names = ["Signal", "Background"], sample_weight=w, digits=4))
            print("AUC-ROC: ", f"{self.evaluate_AUC(X, y, w):.04f}")
            # for n in range(5, 105, 5):
            #     z_exp = expected_significance_binned(self.model, X, y, w, nbins = n)
            #     print(f"Expected Cowan Significance ({n:>3} bins): ", f"{z_exp:.04f}", "σ")
            
        for k in self.ds.keys():
            print("-"*70)
            X, y, w, w_model = self.ds[k].values()
            short_report(X, y, w, label = k)
        print("-"*70)
        if all(v is not None for v in (X_test, y_test, w_test)):
            short_report(X_test, y_test, w_test, label = "Test")
            print("-"*70)

        # (2) calculate the ks test p-values
        print("Kolmogorov-Smirnov (KS) test results:")
        table  = [["Train", "Val", *self.get_ps_value()]]
        if all(v is not None for v in (X_test, y_test)):
            table.append(["Train", "Test", *self.get_ps_value(X_test, y_test)])
        print(tabulate(table, headers=["Distr. 1", "Distr. 2", "Signal", "Background"], tablefmt="fancy_grid"))

        # (3) features ranking
        self.update_features_names()
        self.print_features_ranking(importance_type="gain")
        self.print_features_ranking(importance_type="weight")

        return self.model

    def read_model_structure(self, output_dir = "./", draw_flowcharts=False):
        output_dir = Path(output_dir).resolve() / "model_structure"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Tree structure, dataframes
        booster = self.model.get_booster()
        tree_df = booster.trees_to_dataframe()
        tree_df.to_csv(output_dir / "tree_structure.csv", index=False)

        # Tree structure, raw text representation
        tree_dump = booster.get_dump(with_stats=True)
        with open(output_dir / "tree_structure.txt", "w") as f:
            for i in range(len(tree_dump)):
                f.write(f"Structure of Tree {i}:\n")
                f.write(tree_dump[i])
                f.write("\n\n")
        
        # Visual flowcharts
        if draw_flowcharts:
            flowcharts_dir = output_dir / "visual_flowcharts"
            flowcharts_dir.mkdir(parents=True, exist_ok=True)
            num_trees = len(tree_dump)
            tree_indices = [0]
            if num_trees > 1:
                # Always draw tree 0, then pick up to 5 others at random
                n_extra = min(5, num_trees - 1)
                extra = np.random.choice(np.arange(1, num_trees), size=n_extra, replace=False)
                tree_indices.extend(sorted(int(i) for i in extra))

            print(f"Plotting visual flowcharts for {len(tree_indices)} randomly chosen trees.")
            for tree_idx in tqdm(tree_indices):
                plt.figure(figsize=(12, 8))
                xgb.plot_tree(self.model, tree_idx=tree_idx, rankdir='TB')
                plt.title(f"Visual representation of Tree {tree_idx}")
                plt.tight_layout()
                plt.savefig(flowcharts_dir / f"tree_{tree_idx}.png", dpi=600)
                plt.close()
        
        return None

    def get_shap_summary(self, *, 
        X_test = None, 
        df_sig = None, df_bkg=None, weight_col = "flat_weight",
        output_dir="./",       
        get_beeswarm_plot=True,
        get_bar_plot=False, 
        get_waterfall_plots=False,
        n_special_events=10, 
        get_dependance_plots=False,
        n_dependence=None, 
        dpi=300,
        prefix=""
        ):

        if all(x is None for x in (X_test, df_sig, df_bkg)):
            X = self.ds["train"]["X"]
        elif X_test is not None:
            X = X_test
        elif X_test is None and all(x is not None for x in (df_sig, df_bkg)):
            X, y, w = to_Xyw(df_sig, df_bkg, self.features, weight_col)

        output_dir = Path(output_dir).resolve() / f"shap{prefix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        explainer = shap.Explainer(self.model)
        shap_values = explainer(X)

        # Mean absolute SHAP values
        importance = pd.DataFrame({"Feature": self.features, "Mean |SHAP|": np.abs(shap_values.values).mean(axis=0)})
        importance = importance.sort_values("Mean |SHAP|", ascending=False) #.reset_index(drop=True)
        importance.to_csv(output_dir / "shap_importance.csv", index=False)

        # Beeswarm plot
        if get_beeswarm_plot:
            print("Plotting the beeswarm plot")
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=self.features, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_beeswarm.png", dpi=dpi)
            plt.close()

        # bar plot
        if get_bar_plot:
            print("Plotting the bar plot")
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=self.features, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_importance_bar.png", dpi=dpi)
            plt.close()

        # Waterfall plots for highest and lowest scoring events
        if get_waterfall_plots:
            waterfall_dir = output_dir / "waterfall_plots"
            waterfall_dir.mkdir(parents=True, exist_ok=True)
            scores = self.predict(X)
            highest_events = np.argsort(scores)[-1*n_special_events:][::-1]
            lowest_events = np.argsort(scores)[:n_special_events]
            print(f"Plotting waterfall plots for the {len(highest_events)} highest scoring features")
            for i, event in tqdm(enumerate(highest_events)):
                plt.figure()
                explanation = shap.Explanation(values=shap_values.values[event], base_values=shap_values.base_values[event], data=X.iloc[event].values, feature_names=self.features,)
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                plt.savefig(waterfall_dir / f"waterfall_highest_{i+1}_score_{scores[event]:.4f}.png", dpi=dpi)
                plt.close()
            print(f"Plotting waterfall plots for the {len(lowest_events)} lowest scoring features")
            for i,event in tqdm(enumerate(lowest_events)):
                plt.figure()
                explanation = shap.Explanation(values=shap_values.values[event], base_values=shap_values.base_values[event], data=X.iloc[event].values, feature_names=self.features,)
                shap.plots.waterfall(explanation, show=False)
                plt.tight_layout()
                plt.savefig(waterfall_dir / f"waterfall_lowest_{i+1}_score_{scores[event]:.4f}.png", dpi=dpi)
                plt.close()

        # Dependance plots
        if get_dependance_plots:
            dependence_dir = output_dir / "dependence_plots"
            dependence_dir.mkdir(parents=True, exist_ok=True)
            features = importance["Feature"][:n_dependence] if (n_dependence!=None and n_dependence <= len(self.features)) else importance["Feature"]
            print(f"Plotting dependance plots for {len(features)} features")
            for feature in tqdm(features):
                plt.figure()
                shap.dependence_plot(feature, shap_values.values, X, interaction_index="auto", show=False, feature_names=self.features,)
                plt.tight_layout()
                safe_name = feature.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                plt.savefig(dependence_dir / f"shap_dependence_{safe_name}.png", dpi=dpi)
                plt.close()
            
        return None

    def plot_permutation_importance(self, X, y, metric='roc_auc', n_repeats=10, output_dir="./"):

        os.makedirs(os.path.abspath(output_dir), exist_ok=True)

        result = permutation_importance(self.model, X, y, scoring=metric, n_repeats=n_repeats, random_state=42, n_jobs=-1)

        imp = pd.DataFrame({'feature': self.features, 'importance': result.importances_mean, 'std': result.importances_std}).sort_values('importance')
        #identify harmful features
        harmful = imp[imp['importance'] < 0]['feature'].tolist()
        if harmful:
            print(f"⚠️ Harmful features (negative importance) that will drop your {metric}:")
            for i, hf in enumerate(harmful, start = 1):
                print(f"{i:<3d} : {hf}")
        else:
            print("✅ No harmful features detected.")
        
        # Plot everything
        colors = ['red' if x < 0 else 'steelblue' for x in imp['importance']]
        plt.figure(figsize=(8, max(4, len(imp)*0.3)))
        plt.barh(imp['feature'], imp['importance'], color=colors,alpha=0.8)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.xlabel(f'Permutation importance (drop in {metric})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"permutation_importance_{metric}.png"), dpi=600)
        plt.close()
        return imp

    def check_overtraining(self, 
        df_sig_train=None, df_bkg_train=None, 
        df_sig_test=None, df_bkg_test=None, 
        nBins=30, xMin=0.0, xMax=1.0, output_dir="./", weight_col="flat_weight"
        ):

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if df_sig_train is None and df_bkg_train is None:
            X_train, y_train, w_train = self.ds["train"]["X"], self.ds["train"]["y"], self.ds["train"]["w"]
        else:
            X_train, y_train, w_train = to_Xyw(df_sig_train, df_bkg_train, self.features, weight_col)
        y_pred_prob_train = self.predict(X_train)

        if df_sig_test is None and df_bkg_test is None:
            X_test, y_test, w_test = self.ds["val"]["X"], self.ds["val"]["y"], self.ds["val"]["w"]
        else:
            X_test, y_test, w_test = to_Xyw(df_sig_test, df_bkg_test, self.features, weight_col)
        y_pred_prob_test = self.predict(X_test)

        # Book and fill the histograms
        weighted_hists, unweighted_hists = [], []
        def make_hist(name, x, w=None):
            n=name.split("_")
            isSig = "sig" in n
            isTrain = "t" in n
            isWeigthed = "w" in n
            yTitle = ("" if isWeigthed else "Un-") + f"Weighted Events / {nBins} bins"

            h = ROOT.TH1D(name, f"{name}; BDT Score; {yTitle}", nBins, xMin, xMax)
            h.SetDirectory(0)
            h.Sumw2()
            x = np.asarray(x, dtype="double")
            if w is None: 
                w = np.ones_like(x)
            else: 
                w = np.asarray(w, dtype="double")
            h.FillN(int(x.size), x, w)

            # Reverted colors: Signal = Red, Background = Blue
            color = ROOT.kRed if isSig else ROOT.kBlue
            h.SetLineColor(color)
            h.SetMarkerColor(color)
            h.SetMarkerStyle(21)
            h.SetMarkerSize(0.8)
            h.SetLineWidth(1)

            if isTrain:
                # Training is filled drawn as hist
                if isSig:
                    # Signal is filled with hatched fill style
                    h.SetFillStyle(3004) 
                    h.SetFillColor(color)
                else:
                    # Background is filled with solid color with Alpha(0.8)
                    h.SetFillStyle(1001)
                    h.SetFillColorAlpha(color, 0.4)
            else:
                # Validation is not filled, drawn as error bars
                h.SetFillStyle(0)
            
            if isWeigthed:
                weighted_hists.append(h)
            else:
                unweighted_hists.append(h)

            return h

        # Background distributions
        x, w = y_pred_prob_train[y_train==0], w_train[y_train==0]
        h_bkg_t_w = make_hist("h_bkg_t_w", x, w)
        h_bkg_t_uw = make_hist("h_bkg_t_uw", x, None)

        x, w = y_pred_prob_test[y_test==0], w_test[y_test==0]
        h_bkg_v_w = make_hist("h_bkg_v_w", x, w)
        h_bkg_v_uw = make_hist("h_bkg_v_uw", x, None)

        # signal distributions
        x, w = y_pred_prob_train[y_train==1], w_train[y_train==1]
        h_sig_t_w = make_hist("h_sig_t_w", x, w)
        h_sig_t_uw = make_hist("h_sig_t_uw", x, None)

        x, w = y_pred_prob_test[y_test==1], w_test[y_test==1]
        h_sig_v_w = make_hist("h_sig_v_w", x, w)
        h_sig_v_uw = make_hist("h_sig_v_uw", x, None)

        # Use KS test p-value to check for overtraining
        bkg_w_ks = h_bkg_t_w.KolmogorovTest(h_bkg_v_w, "UONX")
        sig_w_ks = h_sig_t_w.KolmogorovTest(h_sig_v_w, "UONX")
        bkg_uw_ks = h_bkg_t_uw.KolmogorovTest(h_bkg_v_uw, "UONX")
        sig_uw_ks = h_sig_t_uw.KolmogorovTest(h_sig_v_uw, "UONX")

        # Use KS test p-value to check for overtraining
        bkg_w_ks_d = h_bkg_t_w.KolmogorovTest(h_bkg_v_w, "M")
        sig_w_ks_d = h_sig_t_w.KolmogorovTest(h_sig_v_w, "M")
        bkg_uw_ks_d = h_bkg_t_uw.KolmogorovTest(h_bkg_v_uw, "M")
        sig_uw_ks_d = h_sig_t_uw.KolmogorovTest(h_sig_v_uw, "M")

        # --- Chi-Square Test (binned) ---
        # "P" prints chi2 and ndf internally; the function returns the p-value.
        # "WW" for weighted histograms, "UU" for unweighted histograms.
        chi2_p_bkg_w = h_bkg_t_w.Chi2Test(h_bkg_v_w, "WW P")
        chi2_p_sig_w = h_sig_t_w.Chi2Test(h_sig_v_w, "WW P")
        chi2_p_bkg_uw = h_bkg_t_uw.Chi2Test(h_bkg_v_uw, "UU P")
        chi2_p_sig_uw = h_sig_t_uw.Chi2Test(h_sig_v_uw, "UU P")

        # Helper to compute the integral and its statistical error 
        def get_int_and_err(h):
            err = np.zeros(1, dtype="double")
            i = h.IntegralAndError(0, h.GetNbinsX() + 1, err)
            return f"{i:.3f} #pm {err[0]:.3f}"

        # Pre-compute strings before normalization
        ns_w_str = f"#it{{N_{{s}}}} = {get_int_and_err(h_sig_t_w)} (training) / {get_int_and_err(h_sig_v_w)} (test)"
        nb_w_str = f"#it{{N_{{b}}}} = {get_int_and_err(h_bkg_t_w)} (training) / {get_int_and_err(h_bkg_v_w)} (test)"
        ns_uw_str = f"#it{{N_{{s}}}} = {get_int_and_err(h_sig_t_uw)} (training) / {get_int_and_err(h_sig_v_uw)} (test)"
        nb_uw_str = f"#it{{N_{{b}}}} = {get_int_and_err(h_bkg_t_uw)} (training) / {get_int_and_err(h_bkg_v_uw)} (test)"

        # Normalize the histograms before drawing
        # in fact we can use the .DrawNormalized(), but usage is not flexiable
        for h in (weighted_hists + unweighted_hists):
            if h.Integral() > 0:
                h.Scale(1.0 / h.Integral())

        # Draw
        ROOT.gStyle.SetEndErrorSize(3)
        canvas = ROOT.TCanvas("c", "c", 1400, 600)
        canvas.Divide(2,1)
        
        canvas.cd(1)

        ymax1 = max([h.GetMaximum() for h in weighted_hists]) * 2.0
        h_bkg_t_w.SetMaximum(ymax1)
        h_bkg_t_w.SetMinimum(1e-4)
        h_bkg_t_w.Draw("hist")
        h_bkg_v_w.Draw("same e1")
        h_sig_t_w.Draw("same hist")
        h_sig_v_w.Draw("same e1")

        leg1 = ROOT.TLegend(0.19, 0.78, 0.88, 0.92)
        leg1.SetNColumns(2)
        leg1.SetBorderSize(0)
        leg1.SetFillStyle(0)
        leg1.SetTextSize(0.03)
        leg1.AddEntry(h_sig_t_w, "Signal (training)", "f")
        leg1.AddEntry(h_bkg_t_w, "Background (training)", "f")
        leg1.AddEntry(h_sig_v_w, "Signal (test)", "lep")
        leg1.AddEntry(h_bkg_v_w, "Background (test)", "lep")
        leg1.Draw()
               
        tex1 = ROOT.TLatex()
        tex1.SetNDC()
        tex1.SetTextSize(0.028)
        tex1.DrawLatex(0.2, 0.73, f"Kolmogorov-Smirnov test: signal (background) #it{{p}}-value = {sig_w_ks:.3f} ({bkg_w_ks:.3f})")
        tex1.DrawLatex(0.2, 0.70, f"Kolmogorov-Smirnov test: signal (background) distance = {sig_w_ks_d:.3f} ({bkg_w_ks_d:.3f})")
        tex1.DrawLatex(0.2, 0.67, f"Chi-Square test: signal (background) #it{{p}}-value = {chi2_p_sig_w:.3f} ({chi2_p_bkg_w:.3f})")
        tex1.DrawLatex(0.2, 0.64, ns_w_str)
        tex1.DrawLatex(0.2, 0.61, nb_w_str)

        canvas.cd(2)

        ymax2 = max([h.GetMaximum() for h in unweighted_hists]) * 2.0
        h_bkg_t_uw.SetMaximum(ymax2) 
        h_bkg_t_uw.SetMinimum(1e-4)
        h_bkg_t_uw.Draw("hist")
        h_bkg_v_uw.Draw("same e1")
        h_sig_t_uw.Draw("same hist")
        h_sig_v_uw.Draw("same e1")

        leg2 = ROOT.TLegend(0.19, 0.78, 0.88, 0.92)
        leg2.SetNColumns(2)
        leg2.SetBorderSize(0)
        leg2.SetFillStyle(0)
        leg2.SetTextSize(0.03)
        leg2.AddEntry(h_sig_t_uw, "Signal (training)", "f")
        leg2.AddEntry(h_bkg_t_uw, "Background (training)", "f")
        leg2.AddEntry(h_sig_v_uw, "Signal (test)", "lep")
        leg2.AddEntry(h_bkg_v_uw, "Background (test)", "lep")
        leg2.Draw()
        
        tex2 = ROOT.TLatex()
        tex2.SetNDC()
        tex2.SetTextSize(0.028)
        tex2.DrawLatex(0.2, 0.73, f"Kolmogorov-Smirnov test: signal (background) #it{{p}}-value = {sig_uw_ks:.3f} ({bkg_uw_ks:.3f})")
        tex2.DrawLatex(0.2, 0.70, f"Kolmogorov-Smirnov test: signal (background) distance = {sig_uw_ks_d:.3f} ({bkg_uw_ks_d:.3f})")
        tex2.DrawLatex(0.2, 0.67, f"Chi-Square test: signal (background) #it{{p}}-value = {chi2_p_sig_uw:.3f} ({chi2_p_bkg_uw:.3f})")
        tex2.DrawLatex(0.2, 0.64, ns_uw_str)
        tex2.DrawLatex(0.2, 0.61, nb_uw_str)

        canvas.Update()
        canvas.SaveAs(str(output_dir / "score_distributions.png"))
        canvas.SaveAs(str(output_dir / "score_distributions.pdf"))
        canvas.Close()

        return {
            "ks_p_sig_weighted": sig_w_ks, "ks_p_bkg_weighted": bkg_w_ks,
            "ks_d_sig_weighted": sig_w_ks_d, "ks_d_bkg_weighted": bkg_w_ks_d,
            "ks_p_sig_unweighted": sig_uw_ks, "ks_p_bkg_unweighted": bkg_uw_ks,
            "ks_d_sig_unweighted": sig_uw_ks_d, "ks_d_bkg_unweighted": bkg_uw_ks_d,
            "chi2_p_sig_weighted": chi2_p_sig_w, "chi2_p_bkg_weighted": chi2_p_bkg_w,
            "chi2_p_sig_unweighted": chi2_p_sig_uw, "chi2_p_bkg_unweighted": chi2_p_bkg_uw,
        }

    def draw_score_histogram(self, df_sig, df_bkg, plotter = None, nbins = 20, xMin = 0.0, xMax = 1.0, output_dir = "./", weight_col="flat_weight"):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if plotter is None:
            plotter = plot_feature(df_sig, df_bkg)      
        X_sig = df_sig[self.features].values.astype(np.float32) 
        X_bkg = df_bkg[self.features].values.astype(np.float32)
        scores_sig = self.predict(X_sig)
        scores_bkg = self.predict(X_bkg)
        df_sig['bdt_score'] = scores_sig
        df_bkg['bdt_score'] = scores_bkg
        # Save score-only DataFrames to parquet
        df_sig[['bdt_score', weight_col, 'proc_name']].to_parquet(output_dir / f"sig.parquet", engine='pyarrow', compression='snappy', index=False)
        df_bkg[['bdt_score', weight_col, 'proc_name']].to_parquet(output_dir / f"bkg.parquet", engine='pyarrow', compression='snappy', index=False)
        plotter.df_sig = df_sig
        plotter.df_bkg = df_bkg
        plotter.plot('bdt_score', nbins, xMin=xMin, xMax=xMax)
