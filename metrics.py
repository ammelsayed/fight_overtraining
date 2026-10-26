import math
import numpy as np

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

## The expected signficance
## If you use the binned one then we are in trouble
## The binned expected signficance should only be used for
## getting a rough estimate of the final significance of the scores distributions
## and not be used as an optimization objective
def get_counts_and_errors(model, X, y, w, nbins = 30, weighted = True):
    y_pred_prob = model.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1.0, nbins + 1)
    if weighted == False:
        w = np.array([1] * len(X), dtype=np.float32)
    counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
    counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
    sum_w2_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0]**2)
    errors_bkg = np.sqrt(sum_w2_bkg)
    return counts_sig, counts_bkg, errors_bkg, _

def expected_significance_binned(model, X, y, w, nbins = 30):
    counts_sig, counts_bkg, errors_bkg, _ = get_counts_and_errors(model, X, y, w, nbins = nbins)
    return np.sqrt(np.sum(vec_asimov(s=counts_sig, b=counts_bkg, b_err=errors_bkg)**2))

def print_bdt_histogram_bins(model, X, y, w, nbins = 30):
    np.set_printoptions(precision=2, suppress=False)
    y_pred_prob = model.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1.0, nbins + 1)
    counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
    counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
    print(f"  > Binned signal histogram values:")
    print(counts_sig)
    print(f"  > Binned background histogram values")
    print(counts_bkg)
    print(f"  > Bin edges:")
    print(_)

## A stable version of the expted signficance
def amsasimov(s_in, b_in):
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
    """
    s=np.copy(s_in)
    b=np.copy(b_in)
    s=np.where( (b_in == 0) , 0., s_in)
    b=np.where( (b_in == 0) , 1., b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)  # nan if unphysical values
    if np.isscalar(s_in):
        return float(ams)
    else:
        return  ams

def significance_vscore(y_true, y_score, sample_weight=None):
    """
    Calculate the significance using the Asimov method.
    """
    if sample_weight is None:
        sample_weight = np.full(len(y_true), 1.0)

    # Define bins for y_score (100 bins from 0 to 1.0)
    bins = np.linspace(0, 1.0, 101)

    # Fills s and b weighted binned distributions
    s_hist, bin_edges = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, bin_edges = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Compute cumulative sums (from the right!)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    # Compute significance
    significance = amsasimov(s_cumul, b_cumul)

    # Find the bin with the maximum significance
    max_value = np.max(significance)

    return significance


def significance_reduction_ratio(weights):
    """
    Compute the ratio R = sqrt( sum(w) / (sum(w) + sum(w^2)) )
    which quantifies how much the significance is reduced when
    using weighted events compared to unweighted events (w=1).

    Parameters:
        weights : array-like or float
            If array-like: per‑event weights.
            If float: assumed constant weight for all events (flat weights).

    Returns:
        R : float
            The reduction factor (0 < R ≤ 1).
            Multiply the naive significance s/sqrt(b) by R to get
            the approximate significance with weight‑induced variance.

    For flat weights (all w_i = w):
        R = 1 / sqrt(1 + w)
    """
    if np.isscalar(weights):
        w = float(weights)
        return 1.0 / np.sqrt(1.0 + w)
    else:
        w = np.asarray(weights)
        sum_w = np.sum(w)
        sum_w2 = np.sum(w**2)
        return np.sqrt(sum_w / (sum_w + sum_w2))

def significance_vscore_scorer(y_true, y_score, sample_weight=None):
    """
    Returns the maximum significance score for the given true labels and predicted scores.
    This function is designed to be used as a scorer in model evaluation.
    """
    return np.max(significance_vscore(y_true, y_score, sample_weight=sample_weight))


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)



def bkg_rejection_rate(sig_x, sig_w, bkg_x, bkg_w, sig_eff=0.7, cut_direction=">="):
    # basic onput validation
    if not 0.0 <= sig_eff <= 1.0:
        raise ValueError(f"sig_eff must be between 0 and 1, got {sig_eff}")

    if cut_direction not in (">=", "<="):
        raise ValueError(f"cut_direction must be '>=' or '<=', got '{cut_direction}'")

    # Sort signal and compute cumulative weights
    idx = np.argsort(sig_x)
    sorted_sig_x = sig_x[idx]
    cumw = np.cumsum(sig_w[idx])
    total = cumw[-1]
    
    # Target cumulative weight depending on cut direction
    if cut_direction == ">=":
        target_cum = (1 - sig_eff) * total   
    else:  # cut_direction == "<="
        target_cum = sig_eff * total        
    
    # Find position where cumulative weight reaches or exceeds target
    pos = np.searchsorted(cumw, target_cum)
    
    # If exactly at a point, return that score
    # else interpolate threshold linearly between adjacent points
    # if you do not want to liner interpolation,
    # you can just use thr = sorted_sig_x[pos]
    if pos == 0:
        thr = sorted_sig_x[0]
    elif pos >= len(sorted_sig_x):
        thr = sorted_sig_x[-1]
    else:
        # w_prev: cumulative weight just below target
        # w_curr: cumulative weight just above or equal to target
        w_prev, w_curr = cumw[pos-1], cumw[pos]
        if w_curr == w_prev:
            thr = sorted_sig_x[pos]
        else:
            frac = (target_cum - w_prev) / (w_curr - w_prev)
            thr = sorted_sig_x[pos-1] + frac * (sorted_sig_x[pos] - sorted_sig_x[pos-1])
    
    # Apply cut to background and compute rejection
    if cut_direction == ">=":
        pass_bkg = bkg_x >= thr
    else:
        pass_bkg = bkg_x <= thr
    
    bkg_eff = np.sum(bkg_w[pass_bkg]) / np.sum(bkg_w)
    bkg_rej = 1 - bkg_eff
    return bkg_rej, thr


def threshold_from_signal_rejection(scores, weights, rejection):
    """
    Returns threshold T such that:  fraction of weighted signal with score <= T  = rejection.
    
    Parameters
    ----------
    scores : array-like, signal scores
    weights : array-like, signal weights
    rejection : float, desired signal rejection rate (e.g., 0.0005 for 0.05%)
    
    Returns
    -------
    threshold : float
    """
    idx = np.argsort(scores)
    sorted_scores = scores[idx]
    cumw = np.cumsum(weights[idx])
    target = rejection * cumw[-1]
    pos = np.searchsorted(cumw, target)
    # If exactly at a point, return that score; else linearly interpolate
    if pos == 0:
        return sorted_scores[0]
    if pos >= len(sorted_scores):
        return sorted_scores[-1]
    # Linear interpolation between sorted_scores[pos-1] and sorted_scores[pos]
    w_prev = cumw[pos-1]
    w_curr = cumw[pos]
    if w_curr == w_prev:
        return sorted_scores[pos]
    frac = (target - w_prev) / (w_curr - w_prev)
    return sorted_scores[pos-1] + frac * (sorted_scores[pos] - sorted_scores[pos-1])

def threshold_from_signal_efficiency(scores, weights, efficiency):
    """
    efficiency = fraction of signal that passes cut (score > threshold)
    """
    rejection = 1 - efficiency
    return threshold_from_signal_rejection(scores, weights, rejection)


def significance_with_uncertainty(S, B, sigma_B):
    return S / np.sqrt(B + sigma_B**2) if (B + sigma_B**2) > 0 else 0.0

def significance_above_cut(y_true, bdt_scores, weights=None, cut=0.90, sigma_B=10.0):
    """
    Returns significance (S/√(B+σ_B²)) for events with bdt_scores >= cut.

    Parameters:
    - bdt_scores : array-like, BDT output scores (e.g., probability)
    - y_true     : array-like, 1 for signal, 0 for background
    - cut        : float, threshold on BDT score (e.g., 0.90)
    - weights    : array-like (optional), per-event weights. If None, all events have weight 1.
    - sigma_B    : systematic uncertainty on the total background estimate (scalar)
    """
    pass_cut = np.array(bdt_scores) >= cut
    y = np.array(y_true)
    
    if weights is None:
        weights = np.ones_like(bdt_scores, dtype=float)
    else:
        weights = np.array(weights)
    
    S = np.sum(weights[(y == 1) & pass_cut])
    B = np.sum(weights[(y == 0) & pass_cut])
    
    return significance_with_uncertainty(S, B, sigma_B)
