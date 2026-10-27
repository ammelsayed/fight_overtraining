import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

pd.set_option('display.width', 220)

BASE = '/data/ammelsayed/fight_overtraining/sample_data'

for ch in ['3L_eee', '3L_eeeJ', '4L_eeee']:
    d = f'{BASE}/{ch}'
    feats = json.load(open(f'{d}/features_SR.json'))
    sig = pd.read_parquet(f'{d}/sig_DF1.parquet')
    bkg = pd.read_parquet(f'{d}/bkg_DF1.parquet')
    print('=' * 110)
    print(f'CHANNEL {ch}  (features={len(feats)})')

    # duplication check
    if 'event_id' in bkg.columns:
        n_uniq = bkg['event_id'].nunique()
        print(f'bkg rows={len(bkg)}, unique event_id={n_uniq}, dup factor={len(bkg)/n_uniq:.1f}')
        ub = bkg.drop_duplicates('event_id')
    else:
        ub = bkg

    # per-process background weight stats
    rows = []
    for proc, gdf in bkg.groupby('proc_name'):
        w = gdf['flat_weight'].values
        nu = gdf['event_id'].nunique() if 'event_id' in gdf.columns else len(w)
        rows.append(dict(proc=proc, n_rows=len(w), n_unique=nu, sum_w=w.sum(),
                         mean_w=w.mean(), sum_w2=(w ** 2).sum()))
    t = pd.DataFrame(rows)
    t['N_eff'] = t['sum_w'] ** 2 / t['sum_w2']
    t['w_eff'] = t['sum_w2'] / t['sum_w']
    print('\nBackground per-process weight stats (DF1 rows, incl. mass duplication):')
    print(t.to_string(index=False, float_format=lambda x: f'{x:.4g}'))

    w = bkg['flat_weight'].values
    print(f'\nBKG class total: rows={len(w)}, sum_w={w.sum():.4g}, mean_w={w.mean():.4g}, '
          f'N_eff_kish={(w.sum() ** 2)/(w ** 2).sum():.4g}, w_eff={(w ** 2).sum()/w.sum():.4g}, '
          f'w_max={w.max():.4g}, w_min={w.min():.4g}, max/min ratio={w.max()/w.min():.4g}')

    # signal per-mass stats
    rows = []
    for m, gdf in sig.groupby('signal_mass'):
        wt = gdf['flat_weight'].values
        rows.append(dict(mass=int(m), n=len(wt), sum_w=wt.sum(), N_eff=(wt.sum() ** 2)/(wt ** 2).sum()))
    ts = pd.DataFrame(rows)
    print(f'\nSignal: masses={len(ts)}, n/mass min={ts["n"].min()} max={ts["n"].max()}, '
          f'sum_w/mass min={ts["sum_w"].min():.4g} max={ts["sum_w"].max():.4g}, N_eff/mass min={ts["N_eff"].min():.1f}')

    # feature correlation effective rank (features + signal_mass)
    cols = feats + ['signal_mass']

    def eff_rank(Xdf, tag):
        nan_total = int(Xdf.isna().sum().sum())
        stds = Xdf.std()
        good = stds[stds > 0].index
        dropped = [c for c in Xdf.columns if c not in good]
        C = np.corrcoef(Xdf[good].values.T)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        n_eff = ev.sum() ** 2 / (ev ** 2).sum()
        print(f'{tag}: NaNs={nan_total}, kept={len(good)}/{len(Xdf.columns)} dropped={dropped}, '
              f'eff_rank(participation ratio)={n_eff:.2f} -> colsample candidate={n_eff/len(Xdf.columns):.3f}')
        return C, good, n_eff

    C, good_cols, n_eff = eff_rank(pd.concat([sig[cols], ub[cols]]), 'corr(sig+bkg unique)')
    Cb, good_cols_b, n_eff_b = eff_rank(ub[cols], 'corr(bkg unique only)')

    def top_pairs(C, names, k=5):
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j], C[i, j]))
        pairs.sort(key=lambda p: -abs(p[2]))
        return [(a, b, round(c, 2)) for a, b, c in pairs[:k]]

    print('top-5 |corr| pairs (sig+bkg):', top_pairs(C, good_cols))

    # univariate separation (weighted AUC), signal vs unique bkg
    y = np.r_[np.ones(len(sig)), np.zeros(len(ub))]
    wv = np.r_[sig['flat_weight'].values, ub['flat_weight'].values]
    aucs = {}
    for f in feats:
        v = np.r_[sig[f].values, ub[f].values]
        a = roc_auc_score(y, v, sample_weight=wv)
        aucs[f] = max(a, 1 - a)
    s = pd.Series(aucs).sort_values(ascending=False)
    print(f'\nUnivariate weighted AUC (sig vs unique bkg): max={s.iloc[0]:.3f}, '
          f'top3 mean={s.head(3).mean():.3f}, top5 mean={s.head(5).mean():.3f}, mean={s.mean():.3f}')
    print(s.round(3).to_dict())

    # --- min_child_weight: current suggestion vs ESS-revised (on raw bkg rows, pre-balancing)
    N_MIN = 100
    mcw_current = N_MIN * 0.25 * w.mean()
    mcw_ess = N_MIN * 0.25 * ((w ** 2).sum() / w.sum())
    print(f'\nmin_child_weight on bkg (N_min={N_MIN}, pre-balancing):')
    print(f'  current (mean_w based) : {mcw_current:.4g}  -> min VVV events/leaf = {mcw_current/(0.25*0.02327):.1f}, '
          f'min WZ rows/leaf = {mcw_current/(0.25*0.0005059):.0f}')
    print(f'  ESS-revised (w_eff)    : {mcw_ess:.4g}  -> min VVV events/leaf = {mcw_ess/(0.25*0.02327):.1f}, '
          f'min WZ rows/leaf = {mcw_ess/(0.25*0.0005059):.0f}')
    print(f'  safe limit 0.25*sum_w  : {0.25*w.sum():.4g}')
