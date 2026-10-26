import os
import uuid
import ROOT
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from metrics import weighted_mean, bkg_rejection_rate
from metrics import asimov_significance
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
from gc import collect

# plt.style.use('science')
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("ATLAS")

## =============================
## Run options
## =============================

debug = 0

def get_counts_and_errors(hist):
    n = hist.GetNbinsX()
    counts = np.array([hist.GetBinContent(i) for i in range(1, n+1)], dtype=np.float64)
    errors = np.array([hist.GetBinError(i)   for i in range(1, n+1)], dtype=np.float64)
    return counts, errors

def debug_histogram(counts_sig, counts_bkg, errors_bkg, print_debug=True):
    df = pd.DataFrame({
        'bin':range(len(counts_sig)),
        's':counts_sig,
        'b':counts_bkg,
        'bkg_e':errors_bkg
    })
    
    b_ok = df.b > 0
    df = df.assign(
        b_n_eff = np.where(b_ok, ( df.b ** 2 ) / ( df.bkg_e ** 2), np.nan),
        b_rel_stat_unc = np.where(b_ok, (df.bkg_e / df.b) * 100, np.nan), 
        Z_Asimov = [ROOT.RooStats.AsimovSignificance(s,b,e) if b>0 else np.nan for s,b,e in zip(df.s,df.b,df.bkg_e)],
        Z_NoError = np.where(b_ok, df.s / np.sqrt(df.b.where(b_ok)), np.nan)
    )
    
    s_tot, b_tot = df.s.sum(), df.b.sum()
    e_tot = np.sqrt((df.bkg_e ** 2).sum())
    b_tot_ok = b_tot > 0
    df = pd.concat([df, pd.DataFrame([{
        'bin': 'total',
        's': s_tot,
        'b': b_tot,
        'bkg_e': e_tot,
        'b_n_eff': (b_tot ** 2) / (e_tot ** 2) if b_tot_ok else np.nan,
        'b_rel_stat_unc': (e_tot / b_tot) * 100 if b_tot_ok else np.nan,
        'Z_Asimov': ROOT.RooStats.AsimovSignificance(s_tot, b_tot, e_tot) if b_tot_ok else np.nan,
        'Z_NoError': s_tot / np.sqrt(b_tot) if b_tot_ok else np.nan,
    }])], ignore_index=True)
    
    if print_debug:
        print(df.to_string(index=False, formatters={
            'bin':'{:>5}'.format,
            's':'{:8.3f}'.format,
            'b':'{:8.3f}'.format,
            'bkg_e':'{:8.3f}'.format, 
            'b_n_eff':'{:8.3f}'.format, 
            'b_rel_stat_unc':'{:6.2f}%'.format, 
            'Z_Asimov':'{:6.3f}'.format,
            'Z_NoError':'{:6.3f}'.format}
        ))
    return df

def save_debug_report(df_sig, df_bkg, feature, output_dir, *, bin_edges, weight_col='flat_weight', group_col='proc_name'):
    """
    Assume we have one signal processes in df_sig
    and multiple background processes in df_bkg
    we want to get a detiled report like that of debug_histogram for each background processes
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nbins = len(bin_edges) - 1
    sections = [
        f"# Histogram debug report\n",
        f"- feature: `{feature}`",
        f"- bins:    {nbins}",
        f"- signal rows:     {len(df_sig)}",
        f"- background rows: {len(df_bkg)}",
    ]

    # get signal stat
    array_S       = df_sig[feature].values.astype(np.float64)
    weight_S      = df_sig[weight_col].values.astype(np.float64)
    counts_sig, _ = np.histogram(array_S, bins=bin_edges, density=False, weights=weight_S)
    sum_w2_sig, _ = np.histogram(array_S, bins=bin_edges, density=False, weights=weight_S**2)
    errors_sig    = np.sqrt(sum_w2_sig)

    # get total background stat
    array_B       = df_bkg[feature].values.astype(np.float64)
    weight_B      = df_bkg[weight_col].values.astype(np.float64)
    counts_bkg, _ = np.histogram(array_B, bins=bin_edges, density=False, weights=weight_B)
    sum_w2_bkg, _ = np.histogram(array_B, bins=bin_edges, density=False, weights=weight_B**2)
    errors_bkg    = np.sqrt(sum_w2_bkg)

    Z_exp = np.sqrt(np.sum([ROOT.RooStats.AsimovSignificance(s,b,e)**2 if b>0 else np.nan for s,b,e in zip(counts_sig, counts_bkg, errors_bkg)]))
    Z_exp_noerr = np.sqrt(np.sum([ROOT.RooStats.AsimovSignificance(s,b,0)**2 if b>0 else np.nan for s,b,e in zip(counts_sig, counts_bkg, errors_bkg)]))
    sections.append(f"- Asimov signifiance: {Z_exp}")
    sections.append(f"- Asimov signifiance (No Error): {Z_exp_noerr}\n")

    # write into the report
    df_tot = debug_histogram(counts_sig, counts_bkg, errors_bkg, print_debug=False)
    sections.append("## Total background\n\n" + df_tot.to_markdown(index=False))

    # get stat of each background process
    for proc, group_df in df_bkg.groupby(group_col):
        array       = group_df[feature].values.astype(np.float64)
        weight      = group_df[weight_col].values.astype(np.float64)
        counts, _   = np.histogram(array, bins=bin_edges, density=False, weights=weight)
        sum_w2, _   = np.histogram(array, bins=bin_edges, density=False, weights=weight**2)
        errors      = np.sqrt(sum_w2)

        df = debug_histogram(counts_sig, counts, errors, print_debug=False)
        sections.append(f"## Background: `{proc}`\n\n" + df.to_markdown(index=False))

    # write to disk
    path = output_dir / "histogram_debug_report.md"
    path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Saved debug report at '{path.resolve()}'.")
    return path

def draw_label(x,y,line,text_size=0.04, setDNC = True):
    x = np.double(x); y = np.double(y)
    label = ROOT.TLatex()
    label.SetNDC(setDNC)
    label.SetTextSize(text_size)
    label.DrawLatex(x, y, line)
    return label

def plot_bkg_rejection_curve(sig_x, sig_w, bkg_x, bkg_w, title, targets=[0.7, 0.8, 0.9, 0.95], output_dir="./bkg_rejection_rate.png"):
    mean_sig = weighted_mean(sig_x, sig_w)
    mean_bkg = weighted_mean(bkg_x, bkg_w)
    if mean_sig < mean_bkg:
        cut_direction = "<="
    else:
        cut_direction = ">="

    sig_effs = np.linspace(0.01, 0.99, 100) 
    bkg_rejs = []
    thrs = []
    
    for eff in sig_effs:
        rej, _ = bkg_rejection_rate(sig_x, sig_w, bkg_x, bkg_w, sig_eff=eff, cut_direction=cut_direction)
        bkg_rejs.append(rej)
        thrs.append(_)
    
    plt.figure(figsize=(6,5))
    plt.plot(bkg_rejs, sig_effs, 'b-', lw=2)
    plt.xlabel('Background Rejection')
    plt.ylabel('Signal Efficiency')
    plt.grid(True, alpha=0.3)
    
    for target in targets:
        idx = np.argmin(np.abs(sig_effs - target))
        eff_actual = sig_effs[idx]
        rej_actual = bkg_rejs[idx]
        thr_actual = thrs[idx]
        plt.axhline(y=eff_actual, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=rej_actual, color='r', linestyle='--', alpha=0.5)
        plt.plot(rej_actual, eff_actual, 'ro', markersize=6, label = rf'$\epsilon_s$ = {int(target*100)}% @ {cut_direction} {thr_actual:.02f}')
        plt.annotate(f'{int(target*100)}%', (rej_actual, eff_actual), textcoords="offset points", xytext=(5,5), fontsize=9)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir, dpi=300)

class plot_feature:
    def __init__(self, df_sig, df_bkg):
        self.df_sig = df_sig
        self.df_bkg = df_bkg
        self.output_dir = Path(".").resolve()
        self.draw_histograms_formats = ["pdf", "png"]
        self.stack_background_processes = True
        self.show_auc_and_bestcut = True
        self.show_event_count_on_legends = True
        self.apply_smoothing = False
        self.sqrt_s = 13.6
        self.luminosity = 500
        self.weight_col = "flat_weight"
        self._custom_labels = None
        self._draw_significance_pad = True
        self.debug_histogram = False
        self.include_overflow = False
        self.regroup_bkg_proc_names = False      
        self.regroup_sig_proc_names = False 
        self.sig_plot_scale = 1.0
        self.style = self.update_style()
        self._set_color_palette = ROOT.kSunset
        self.labels = [
            "#scale[1.1]{#bf{#sqrt{s} = %s TeV, %s fb^{-1}}}" % (self.sqrt_s, self.luminosity),
            "#it{B_{e} = B_{#mu} = B_{#tau} = 1/3}",
        ]

    @property
    def draw_significance_pad(self):
        return self._draw_significance_pad

    @draw_significance_pad.setter
    def draw_significance_pad(self, value):
        self._draw_significance_pad = value
        self.update_style()

    @property
    def set_color_palette(self):
        return self._set_color_palette

    @set_color_palette.setter
    def set_color_palette(self, value):
        self._set_color_palette = value
        ROOT.gStyle.SetPalette(value)
        ROOT.gStyle.SetNumberContours(10)
            
    def update_style(self):
        gStyle = ROOT.gStyle
        self.style = {
            "colors" : {
                "sig" : [ROOT.kRed, ROOT.kMagenta, ROOT.kSpring],
                "bkg" : [ROOT.TColor.GetColor("#3f90da"), ROOT.TColor.GetColor("#ffa90e"), ROOT.TColor.GetColor("#bd1f01"), ROOT.TColor.GetColor("#94a4a2"), ROOT.TColor.GetColor("#832db6"), ROOT.TColor.GetColor("#a96b59"), ROOT.TColor.GetColor("#e76300"), ROOT.TColor.GetColor("#b9ac70"), ROOT.TColor.GetColor("#717581"), ROOT.TColor.GetColor("#92dadd")]
            },
            "upper_pad" : {
                "yAxis" : {
                    "LabelSize"   : ( 1.0 if self.draw_significance_pad else gStyle.GetLabelSize("Y") ),
                    "TitleSize"   : ( 1.0 if self.draw_significance_pad else gStyle.GetTitleSize("Y") ),
                    "LabelOffset" : ( 1.0 if self.draw_significance_pad else gStyle.GetLabelOffset("Y") ),
                    "TitleOffset" : ( 1.2 if self.draw_significance_pad else gStyle.GetTitleOffset("Y") )
                },
                "xAxis" : {
                    "LabelSize"   : ( 0.0 if self.draw_significance_pad else gStyle.GetLabelSize("X") ),
                    "TitleSize"   : ( 0.0 if self.draw_significance_pad else gStyle.GetTitleSize("X") ),
                    "LabelOffset" : ( 1.0 if self.draw_significance_pad else gStyle.GetLabelOffset("X") ),
                    "TitleOffset" : ( 1.0 if self.draw_significance_pad else gStyle.GetTitleOffset("X") )
                }
            },
            "lower_pad" : {
                "yAxis" : {
                    "LabelSize"   : ( 0.15 if self.draw_significance_pad else 1.0 ),
                    "TitleSize"   : ( 0.15 if self.draw_significance_pad else 1.0 ),
                    "LabelOffset" : ( 1.0  if self.draw_significance_pad else 1.0 ),
                    "TitleOffset" : ( 0.38 if self.draw_significance_pad else 1.0 )
                },
                "xAxis" : {
                    "LabelSize"   : ( 0.15 if self.draw_significance_pad else 1.0 ),
                    "TitleSize"   : ( 0.15 if self.draw_significance_pad else 1.0 ),
                    "LabelOffset" : ( 1.0   if self.draw_significance_pad else 1.0 ),
                    "TitleOffset" : ( 0.7   if self.draw_significance_pad else 1.0 )
                }
            },
            "legends": {
                "TextFont"  : 42,
                "TextSize"  : 0.04,
                "X1"        : 0.58,  # left edge
                "Y1"        : 0.45,  # bottom edge
                "X2"        : 0.9,   # right edge
                "Y2"        : 0.91    # top edge
            },
            "canvas": {
                "log_y"         : True,
                "min_y"         : 1e-1,
                "max_y_scale"   : 1e5,
                "width"         : 900,
                "height"        : 800,
                "TopMargin"     : 0.0,
                "BottomMargin"  : 0.0,
                "LeftMargin"    : 0.0,
                "RightMargin"   : 0.0
            }
        }
        return self.style

    def set_output_dir(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        return self

    def set_draw_histograms_formats(self, formats):
        self.draw_histograms_formats = formats
        return self

    def add_legend_entry(self, legend, hist, legend_name, legend_type, corresponding_hist_obj=None):
        if self.show_event_count_on_legends:
            legend.SetTextFont(102)
            legend.SetTextSize(0.029)
            legend.SetX1(0.45)
            if isinstance(hist, ROOT.TH1):
                counts   = int(round(hist.GetEntries()))
                weighted = hist.Integral(0, hist.GetNbinsX() + 1)
                stat_unc = float(np.sqrt(sum(hist.GetBinError(i) ** 2 for i in range(1, hist.GetNbinsX() + 1))))
            elif corresponding_hist_obj is not None:
                counts   = int(round(corresponding_hist_obj.GetEntries()))
                weighted = corresponding_hist_obj.Integral(0, corresponding_hist_obj.GetNbinsX() + 1)
                stat_unc = float(np.sqrt(sum(corresponding_hist_obj.GetBinError(i) ** 2 for i in range(1, corresponding_hist_obj.GetNbinsX() + 1))))
            else:
                print("Wrong object passed.")
                counts, weighted, stat_unc = -0.0, -0.0, -0.0

            new_name = f"{legend_name:<18}{counts:<10}{weighted:<10.4f}{stat_unc:<10.4f}"

        else:
            new_name = legend_name
        
        legend.AddEntry(hist, new_name, legend_type)
        return None
    
    def regroup_df_bkg_proc_names(self, df, proc_prefixes=None, col="proc_name"):
        if proc_prefixes is None:
            proc_prefixes = {
                "#it{t#bar{t}}, #it{tW^{-}}, #it{#bar{t}W^{+}}":
                    ["tW_1L", "tW_2L", "tt_1L", "tt_2L"],
                "#it{t#bar{t}W^{#pm}}, #it{t#bar{t}Z}":
                    ["ttW_1L", "ttW_2L", "ttW_3L",  "ttZ_1L", "ttZ_2L", "ttZ_3L", "ttZ_4L"],
                "#it{W^{#pm}W^{#pm}jj} (QCD, EW)":
                    ["SS-WW_1L", "SS-WW_2L"],
                "#it{W^{#pm}W^{#mp}}, #it{W^{#pm}Z}, #it{ZZ}":
                    ["OS-WW_1L", "OS-WW_2L", "WZ_1L", "WZ_2L", "WZ_3L", "ZZ_2L", "ZZ_4L"],
                "#it{VVV} (#it{V = W^{#pm}, Z})":
                    ["VVV"],
                "#it{W^{#pm}} + jets":
                    ["W + jets"],
                "#it{Z} + jets":
                    ["Z + jets"],
            }

        mapping = {name: group for group, names in proc_prefixes.items() for name in names}
        present = set(df[col].unique())

        used = {k: v for k, v in mapping.items() if k in present}
        for name in mapping.keys() - used.keys():
            print(f"[regroup_proc_names] '{name}' not found in df['{col}'], skipping (keeping as-is).")

        out = df.copy()
        out[col] = out[col].replace(used)
        return out

    def regroup_df_sig_proc_names(self, df, proc_prefixes=None, col="proc_name"):
        if proc_prefixes is None:
            proc_prefixes = {}
            scaler = f"#times {self.sig_plot_scale}" if self.sig_plot_scale != 1.0 else ""
            for m in range(1000, 2050, 50):
                proc_prefixes[f"#it{{m_{{N}}}} = {m/1000:.2f} TeV {scaler}"] = [f"M{int(m)}"]

        mapping = {name: group for group, names in proc_prefixes.items() for name in names}
        present = set(df[col].unique())

        used = {k: v for k, v in mapping.items() if k in present}
        for name in mapping.keys() - used.keys():
            print(f"[regroup_proc_names] '{name}' not found in df['{col}'], skipping (keeping as-is).")

        out = df.copy()
        out[col] = out[col].replace(used)
        return out

    def _merge_overflow(self, hist, nBins):
        """Merge the overflow bin (nBins+1) into the last regular bin (nBins)."""
        if not self.include_overflow:
            return
        overflow = hist.GetBinContent(nBins + 1)
        overflow_err = hist.GetBinError(nBins + 1)
        if overflow != 0 or overflow_err != 0:
            last_content = hist.GetBinContent(nBins)
            last_err = hist.GetBinError(nBins)
            hist.SetBinContent(nBins, last_content + overflow)
            hist.SetBinError(nBins, math.sqrt(last_err**2 + overflow_err**2))
            hist.SetBinContent(nBins + 1, 0.0)
            hist.SetBinError(nBins + 1, 0.0)

    def plot(self, feature, nBins=30, bin_edges = None, xMin=None, xMax=None, units=None, xTitle = None, yTitle = None, sufx=None):

        # Check the required columns exist or not
        for _ in [feature, self.weight_col, "proc_name"]:
            if _ not in self.df_sig.columns:
                print(f"[plot_feature] SKIP: '{_}' not in df_sig columns.")
                return None
            if _ not in self.df_bkg.columns:
                print(f"[plot_feature] SKIP: '{_}' not in df_bkg columns.")
                return None

        df_sig = self.df_sig
        df_bkg = self.df_bkg
        output_dir = self.output_dir
        labels = self.labels
        os.makedirs(output_dir, exist_ok=True)

        # auto finx xMin and xMax
        if xMin is None:
            xMin = min(np.nanmin(self.df_sig[feature]), np.nanmin(self.df_bkg[feature]))
        if xMax is None:
            xMax = max(np.nanmax(self.df_sig[feature]), np.nanmax(self.df_bkg[feature]))
            # ROOT puts x >= xMax into the overflow bin. Expand slightly so the
            # auto-detected maximum doesn't spike into the last bin when include_overflow=True.
            xMax += (xMax - xMin) * 1e-6 if xMax != xMin else 1.0
       
       # binning
        if bin_edges is None:
            xRange = np.linspace(xMin, xMax, int(nBins) + 1)
            bin_width = f'{abs(xMax-xMin)/nBins:.02f}'
        else:
            xRange = np.array(bin_edges, dtype=np.float64)
            nBins = len(bin_edges) - 1
            bin_width = f'{abs(xMax-xMin)/nBins:.02f} (avg)'
        
        xRange = np.ascontiguousarray(xRange, dtype=np.float64)

        # Generate unique IDs for histograms to avoid name conflicts
        unique_id = str(uuid.uuid4())[:8]

        xTitle = (xTitle or str(feature)) + (f" [{units}]" if units else "")
        yTitle = yTitle or (f"Events / {bin_width} {units}" if units else f"Events / {bin_width}")
     
        # Simple message
        print(f"\nDrawing histogram for the feature: {feature}")
        if debug:
            print(f"    > nbins = {nBins}, x = ({xMin},{xMax}) {units}, bin_width = {bin_width} {units}")
            if self.sig_plot_scale > 1.0:
                print(f"    > Scaling signal events by: {self.sig_plot_scale}")
            
    
        # Calculate AUC, and find best threshold
        bestcut_labels = []
        if self.show_auc_and_bestcut:
            os.makedirs(os.path.join(output_dir,"bkg_rejection_rates"), exist_ok=True)
            array_S = df_sig[feature].values.astype(np.float32)
            array_B = df_bkg[feature].values.astype(np.float32)
            weight_S = df_sig[self.weight_col].values.astype(np.float64)
            weight_B = df_bkg[self.weight_col].values.astype(np.float64)
            y_true = np.concatenate([np.zeros(len(array_B)), np.ones(len(array_S))])
            y_score = np.concatenate([array_B, array_S])
            w = np.concatenate([weight_B, weight_S])
            auc_score = roc_auc_score(y_true, y_score, sample_weight = w)
            auc_score = max(auc_score, 1 - auc_score) # direction independent
            fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight = w)
            best_idx = np.argmax(tpr - fpr)
            best_thr = thresholds[best_idx] # best threshold using Youden's J statistic
            sig_eff_youden = tpr[best_idx]   
            bkg_rej_youden = 1 - fpr[best_idx]
            print(f"AUC (direction independent): {auc_score:.4f}")
            print(f"Youden's J threshold: >= {best_thr:.4f}")
            print(f"  -> Signal efficiency (acceptance): {sig_eff_youden:.4f} ({sig_eff_youden*100:.2f}%)")
            print(f"  -> Background rejection: {bkg_rej_youden:.4f} ({bkg_rej_youden*100:.2f}%)")
            print("Thresholds for fixed signal efficiencies:")
            mean_sig = weighted_mean(array_S, weight_S)
            mean_bkg = weighted_mean(array_B, weight_B)
            print(f"Weighted mean for signal: {mean_sig}")
            print(f"Weighted mean for background: {mean_bkg}")
            plot_bkg_rejection_curve(
                array_S, weight_S, array_B, weight_B, title=feature,
                targets=[0.7, 0.8, 0.9, 0.95], 
                output_dir=f"{output_dir}/bkg_rejection_rates/{feature}.png"
            )
            for eff in [0.999, 0.99, 0.95, 0.90]:
                bkg_rej, thr = bkg_rejection_rate(array_S, weight_S, array_B, weight_B, sig_eff=eff, cut_direction=">=") 
                print(f"(>=) ε_s = {eff*100:.1f}% -> threshold >= {thr:.4f}, background rejection = {bkg_rej:.4f} ({bkg_rej*100:.2f}%)")
                cut_direction=">="
                if mean_sig < mean_bkg:
                    cut_direction="<="
                    print(f"       *Suggested cut direction based on weighted means: <=")
                    bkg_rej, thr = bkg_rejection_rate(array_S, weight_S, array_B, weight_B, sig_eff=eff, cut_direction="<=") 
                    print(f"       *ε_s = {eff*100:.1f}% -> threshold <= {thr:.4f}, background rejection = {bkg_rej:.4f} ({bkg_rej*100:.2f}%)")

                bestcut_labels.append(f"(#epsilon_{{s}}={eff*100:.1f}%)(#epsilon^{{rej}}_{{b}}={bkg_rej*100:.1f}%){cut_direction}{thr:.2f}")


        # Book and fill signal histograms
        # ----------------------------------------

        if self.regroup_sig_proc_names: df_sig = self.regroup_df_sig_proc_names(df_sig)
        sig_list = df_sig['proc_name'].unique().tolist()
        hist_sig_dict = {}
        for idx, sig_name in enumerate(sig_list):
            X_sig = df_sig[df_sig['proc_name'] == str(sig_name)][feature].values
            w_sig = df_sig[df_sig['proc_name'] == str(sig_name)][self.weight_col].values
            hist_sig_name = f"hist_sig_{sig_name}_{unique_id}"
            hist_sig_titles = f"{hist_sig_name}; {xTitle}; {yTitle}"
            hist_sig = ROOT.TH1D(hist_sig_name, hist_sig_titles, nBins, xRange)
            color_idx = idx % len(self.style["colors"]["sig"]) 
            hist_sig.SetLineColor(self.style["colors"]["sig"][color_idx])
            hist_sig.SetLineWidth(4)
            hist_sig.Sumw2()
            for x, w in zip(X_sig, w_sig):
                if np.isfinite(x): hist_sig.Fill(x, w)
            self._merge_overflow(hist_sig, nBins)
            hist_sig.Scale(self.sig_plot_scale)
            hist_sig_dict[sig_name] = hist_sig
            if debug:
                print(f"Booked and filled signal histogram: {hist_sig_name}")
            if self.apply_smoothing: hist_sig.Smooth()
        hist_sig_dict_sorted = dict(sorted(hist_sig_dict.items(), key=lambda item: item[1].Integral(), reverse=False))
        counts_sig, sig_unc = get_counts_and_errors(list(hist_sig_dict_sorted.values())[0])

        # Book and fill background histograms
        # ----------------------------------------

        if self.regroup_bkg_proc_names: df_bkg = self.regroup_df_bkg_proc_names(df_bkg)
        bkg_list = df_bkg['proc_name'].unique().tolist()
        hist_bkg_dict = {}
        for idx, bkg_name in enumerate(bkg_list):
            X_bkg = df_bkg[df_bkg['proc_name'] == str(bkg_name)][feature].values
            w_bkg = df_bkg[df_bkg['proc_name'] == str(bkg_name)][self.weight_col].values
            hist_bkg_name = f"hist_bkg_{bkg_name}_{unique_id}"
            hist_bkg_titles = f"{hist_bkg_name}; {xTitle}; {yTitle}"
            hist_bkg = ROOT.TH1D(hist_bkg_name, hist_bkg_titles, nBins, xRange)
            if self.set_color_palette is None:
                color_idx = idx % len(self.style["colors"]["bkg"]) 
                hist_bkg.SetFillColorAlpha(self.style["colors"]["bkg"][color_idx], 0.9)
                hist_bkg.SetFillColorAlpha(self.style["colors"]["bkg"][color_idx], 0.9)
            hist_bkg.SetLineWidth(1)
            hist_bkg.Sumw2()
            for x, w in zip(X_bkg, w_bkg):
                if np.isfinite(x): hist_bkg.Fill(x, w)
            self._merge_overflow(hist_bkg, nBins) 
            hist_bkg_dict[bkg_name] = hist_bkg
            if debug:
                print(f"Booked and filled background histogram: {hist_bkg_name}")
            if self.apply_smoothing: hist_bkg.Smooth()
        hist_bkg_dict_sorted = dict(sorted(hist_bkg_dict.items(), key=lambda item: item[1].Integral(), reverse=False))

        # Book stacked histogram
        stack_name = f"stack_{unique_id}"
        stack_titles = f"Background Stack; {xTitle}; {yTitle}"
        stack  = ROOT.THStack(stack_name, stack_titles)
        for name, hist in hist_bkg_dict_sorted.items():
            stack.Add(hist)
        if debug:
            print(f"Booked and prepared stacked histogram: {stack_name}")

        # Book and fill total background histogram
        hist_bkg_total_name = f"hist_bkg_total_{unique_id}"
        hist_bkg_total = ROOT.TH1D(hist_bkg_total_name, stack_titles, nBins, xRange)
        hist_bkg_total.SetLineColor(ROOT.kBlack)
        hist_bkg_total.SetLineWidth(2)
        hist_bkg_total.Sumw2()
        for name, hist in hist_bkg_dict_sorted.items():
            hist_bkg_total.Add(hist)
        if self.apply_smoothing:
            hist_bkg_total.Smooth()
        counts_bkg, bkg_unc = get_counts_and_errors(hist_bkg_total)
        if debug:
            print(f"Booked and prepared total background histogram: {hist_bkg_total_name}")
    
        # Book and fill total background statistical uncertanity band
        bkg_uncertainty_band_name = f"bkg_uncertainty_band_{unique_id}"
        bkg_uncertainty_band = ROOT.TGraphAsymmErrors(hist_bkg_total)
        bkg_uncertainty_band.SetName(bkg_uncertainty_band_name)
        bkg_uncertainty_band.SetFillColor(ROOT.kBlack)
        bkg_uncertainty_band.SetFillStyle(3004)  # Hatched pattern
        bkg_uncertainty_band.SetLineColor(ROOT.kBlack)
        bkg_uncertainty_band.SetLineWidth(2)
        bkg_uncertainty_band.SetMarkerSize(0)
        bkg_uncertainty_band_solid_fill = bkg_uncertainty_band.Clone(f"bkg_uncertainty_band_solid_fill{unique_id}")
        bkg_uncertainty_band_solid_fill.SetFillColorAlpha(ROOT.kBlack, 0.05)
        bkg_uncertainty_band_solid_fill.SetFillStyle(1001)
        if debug:
            print(f"Booked and created uncertainty band for total background: {bkg_uncertainty_band_name}")


        # Book Legends and add the names
        # ----------------------------------------

        legend = ROOT.TLegend()
        legend.SetFillStyle(0)
        legend.SetFillColorAlpha(ROOT.kWhite, 0.0)
        legend.SetBorderSize(0)
        legend.SetTextFont(self.style["legends"]["TextFont"])
        legend.SetTextSize(self.style["legends"]["TextSize"])
        legend.SetX1(self.style["legends"]["X1"])
        legend.SetY1(self.style["legends"]["Y1"])
        legend.SetX2(self.style["legends"]["X2"])
        legend.SetY2(self.style["legends"]["Y2"])
        for name, hist in hist_bkg_dict_sorted.items():
            if hist.Integral() > 0 :
                self.add_legend_entry(legend, hist, name, 'f')
        self.add_legend_entry(legend, bkg_uncertainty_band, "MC Stat. Unc.", 'f', corresponding_hist_obj=hist_bkg_total)
        for name, hist in hist_sig_dict_sorted.items():
            self.add_legend_entry(legend, hist, name, 'l')
    
        # Save histograms to a ROOT file
        # ----------------------------------------

        if "root" in self.draw_histograms_formats:
            root_file_path = (output_dir / f"{feature}_histograms.root") if sufx is None else (output_dir / f"{sufx}_{feature}_histograms.root") 
            f_out = ROOT.TFile(str(root_file_path), "RECREATE")
            f_out.cd()
            hist_bkg_total.Write("hist_bkg")
            for idx, (name, hist) in enumerate(hist_sig_dict_sorted.items(), start=1):
                hist_sig.Write(f"hist_sig_{idx}")
            f_out.Close()
            if debug:
                print(f"Saved root file containing histograms at: {root_file_path}")
        else:
            root_file_path = None
        
        # Create significance histogram for the lower pad (same binning as main)
        # ----------------------------------------
        if self.draw_significance_pad:
            hist_sig = list(hist_sig_dict_sorted.values())[0] ## temporary fix
            hist_significance_name = f"hist_significance_{unique_id}"
            hist_significance_titles = f"Significance; {xTitle}; Z_{{exp}} (#sigma)"
            hist_significance = ROOT.TH1D(hist_significance_name, hist_significance_titles, nBins, xRange)
            hist_significance.SetLineColor(ROOT.kBlack)
            hist_significance.SetMarkerStyle(20)
            hist_significance.SetMarkerSize(0.8)
            hist_significance.GetXaxis().SetLabelSize(self.style["lower_pad"]["xAxis"]["LabelSize"])
            hist_significance.GetXaxis().SetTitleSize(self.style["lower_pad"]["xAxis"]["TitleSize"])
            hist_significance.GetXaxis().SetRangeUser(xMin, xMax)
            hist_significance.GetYaxis().SetLabelSize(self.style["lower_pad"]["yAxis"]["LabelSize"])
            hist_significance.GetYaxis().SetTitleSize(self.style["lower_pad"]["yAxis"]["TitleSize"])
            hist_significance.GetYaxis().SetTitleOffset(self.style["lower_pad"]["yAxis"]["TitleOffset"])
            hist_significance.Sumw2()
            z_total2 = 0.0
            for i in range(1, nBins + 1):
                s = hist_sig.GetBinContent(i)
                b = hist_bkg_total.GetBinContent(i)
                b_err = hist_bkg_total.GetBinError(i)
                if b > 0 and np.isfinite(b_err) and b_err > 0:
                    z = ROOT.RooStats.AsimovSignificance(s, b, b_err)
                    if not np.isfinite(z):
                        z = 0.0
                else:
                    z = 0.0
                z_total2 += z * z
                hist_significance.SetBinContent(i, z)
            z_total = math.sqrt(max(z_total2, 0.0))
            zmax = hist_significance.GetMaximum()
            hist_significance.SetMinimum(0.1)  
            hist_significance.SetMaximum(50)
            # hist_significance.SetMaximum(zmax * 1.9)
            hist_significance.GetYaxis().SetNdivisions(505, True)
            if debug:
                print(f"Booked and created significance histogram: {hist_significance_name}")


        # Canvas and pad
        canvas = ROOT.TCanvas("c", "c", self.style["canvas"]["width"], self.style["canvas"]["height"])
        canvas.SetLogy(self.style["canvas"]["log_y"])

        if self.draw_significance_pad:
            pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.25, 1.0, 1.0)
            pad1.SetBottomMargin(0.01)                  
            pad1.SetLogy(self.style["canvas"]["log_y"])
            pad1.Draw()
            pad1.cd()
        
        stack.Draw("hist" + (" pfc plc" if self.set_color_palette is not None else ""))
        stack.GetHistogram().GetXaxis().SetLabelSize(self.style["upper_pad"]["xAxis"]["LabelSize"])
        stack.GetHistogram().GetXaxis().SetTitleSize(self.style["upper_pad"]["xAxis"]["TitleSize"])
        stack.GetHistogram().GetYaxis().SetTitleOffset(self.style["upper_pad"]["yAxis"]["TitleOffset"])
        max_y_scale = self.style["canvas"]["max_y_scale"] if self.style["canvas"]["log_y"] else 2.5
        max_y = max(list(hist_sig_dict_sorted.values())[0].GetMaximum(), stack.GetMaximum()) * max_y_scale
        stack.SetMinimum(self.style["canvas"]["min_y"])
        stack.SetMaximum(max_y)
        hist_bkg_total.Draw("hist same")
        bkg_uncertainty_band.Draw("e2 same")
        bkg_uncertainty_band_solid_fill.Draw("e2 same")
        for name, hist in hist_sig_dict_sorted.items():
            hist.Draw("hist same")
        legend.Draw("same")

        # labels
        StartX, StartY, StepY = 0.20, 0.86, -0.06
        for y, line in zip(np.arange(StartY, StartY + len(labels) * StepY, StepY), labels):
            draw_label(StartX, y, line, text_size=0.04)
        
        if self.show_auc_and_bestcut:
            StartX, StartY, StepY = 0.20, 0.68, -0.06
            for y, line in zip(np.arange(StartY, StartY + len(bestcut_labels) * StepY, StepY), bestcut_labels):
                draw_label(StartX, y, line, text_size=0.033)

        if self.draw_significance_pad:
            canvas.cd()
            pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.25)
            pad2.SetTopMargin(0.01)
            pad2.SetBottomMargin(0.4)
            pad2.SetGridy(False)
            pad2.SetLogy(True)
            pad2.Draw()
            pad2.cd()
            hist_significance.Draw("hist")
            draw_label(0.20, 0.86, f"Z_{{tot}} = {z_total:.2f} #sigma",text_size=0.12)

        canvas.Update()
        for fmt in [f for f in self.draw_histograms_formats if f != "root"]:
            out = str(output_dir / f"{feature}.{fmt}") if sufx is None else str(output_dir / f"{sufx}_{feature}.{fmt}")
            canvas.SaveAs(out)
        canvas.Close()
        collect()

        if self.debug_histogram:
            debug_df = debug_histogram(counts_sig, counts_bkg, bkg_unc, print_debug=False)
            debug_df.to_csv(f"{output_dir}/{feature}_debug.csv", index=False, float_format="%.4f")
            save_debug_report(self.df_sig, self.df_bkg, feature, output_dir, bin_edges=xRange)

        return {
            "counts_sig": counts_sig,
            "counts_bkg": counts_bkg,
            "errors_sig": sig_unc,
            "errors_bkg": bkg_unc,
            "bin_edges": xRange,
            "root_file": root_file_path,
        }

def plot_features(df_sig, df_bkg, features, output_dir, plot_scale_sig = 1000, weight_col="flat_weight"):
    N = len(features)
    n_cols = math.ceil(math.sqrt(N))
    n_rows = math.ceil(N / n_cols)
    width_per_ax = 3.0
    height_per_ax = 2.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * width_per_ax, n_rows * height_per_ax))
    axes_flat = axes.flatten()
    for idx, feature in tqdm(enumerate(features), desc = f'plotting {N} features'):
        if feature not in df_sig.columns.tolist() or feature not in df_bkg.columns.tolist():
            continue
        array_S = df_sig[feature].values.astype(np.float32)
        array_B = df_bkg[feature].values.astype(np.float32)
        weight_S = df_sig[weight_col].values.astype(np.float64)
        weight_B = df_bkg[weight_col].values.astype(np.float64)
        y_true = np.concatenate([np.zeros(len(array_B)), np.ones(len(array_S))])
        y_score = np.concatenate([array_B, array_S])
        w = np.concatenate([weight_B, weight_S])
        auc_score = roc_auc_score(y_true, y_score, sample_weight=w)
        auc_score = max(auc_score, 1 - auc_score) # direction independent
        title_color = 'red' if auc_score > 0.7 else 'black'
        fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=w)
        axes_flat[idx].hist(array_B, weights=weight_B, bins=100, alpha=0.7, color='blue', label='Background')
        axes_flat[idx].hist(array_S, weights=weight_S * plot_scale_sig, bins=100, alpha=0.7, color='red', label='Signal')
        axes_flat[idx].set_title(f"{feature} (AUC={auc_score:.4f})", color=title_color, fontweight='bold')
        axes_flat[idx].set_yscale("log")
    for ax in axes_flat[N:]:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "features.png"), dpi=300)
    plt.close()

def plot_features_correlation_matrix(dataframe, features, output_dir):
    print('plotting correlation matrix ...')
    corr_matrix = dataframe[features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1) # selects elements above the diagonal
    corr_masked = corr_matrix.mask(mask) # mask upper triangle
    
    nfeat = len(features)
    figsize_x = int(max(8, nfeat * 0.6))
    figsize_y = int(max(6, nfeat * 0.5))
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))
    cax = ax.matshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0, label="Pearson Correlation")
    
    base_font = max(10, (20 - nfeat) * 0.2) # Decreases as features increase, minimum 10
    ax.set_xticks(range(nfeat))
    ax.set_yticks(range(nfeat))
    ax.set_xticklabels(features, rotation=45, ha="right", rotation_mode="anchor", fontsize=base_font)
    ax.set_yticklabels(features, fontsize=base_font)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    # Optional: Add the correlation values as text annotations
    annot_font = max(4, base_font * 0.8)
    for i in range(nfeat):
        for j in range(nfeat):
            if not mask[i, j]:
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=annot_font)

    plt.tight_layout()
    plt.savefig(output_dir, dpi=600)
    plt.close()

def plot_significance_reduction_ratio(weights = None):
    from metrics import significance_reduction_ratio
    if weights == None:
        weights = np.logspace(-3, 3, 100)
        R = [significance_reduction_ratio(w) for w in weights]
    else:
        R = [significance_reduction_ratio(w) for w in weights]

    plt.figure(figsize=(4,3))
    plt.plot(weights, R, linewidth=2)
    plt.xscale('log')
    plt.xlabel('per-event weight')
    plt.ylabel('Reduction factor')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('R_vs_flat_weight.png', dpi=300)

def plot_feature_histograms(
    plotter, 
    N_lep, 
    N_fj, 
    fj_name="SoftDroppedFatJet", 
    use_fatjet_variants=False,              
    nbins=20,
    ):

    """Draw feature histograms organized by category. Auxiliary features are excluded."""

    # Global scalars and Event Shapes
    plotter.plot('MET_MET', nbins, units="GeV")
    plotter.plot('HT', nbins, units="GeV")
    plotter.plot('ST', nbins, units="GeV")
    plotter.plot('Meff', nbins, units="GeV")
    plotter.plot('Significance_MET', nbins)
    plotter.plot('Centrality', nbins)
    plotter.plot('Sphericity', nbins)
    plotter.plot('Circularity', nbins)
    plotter.plot('Aplanarity', nbins)

    # Leptons
    for i in range(N_lep):
        plotter.plot(f'Pt_Lepton{i}', nbins, units="GeV")
        plotter.plot(f'MtW_Lepton{i}_MET', nbins, units="GeV") 
        plotter.plot(f'DeltaR_Lepton{i}_MET', nbins)
        plotter.plot(f'DeltaPhi_Lepton{i}_MET', nbins)
        plotter.plot(f'DeltaEta_Lepton{i}_MET', nbins)

    # Lepton-MET reduced (exists only when N_lep > 1)
    if N_lep > 1:
        plotter.plot('MinDeltaR_Lepton_MET', nbins)
        plotter.plot('MaxDeltaR_Lepton_MET', nbins)
        plotter.plot('MinAbsDeltaPhi_Lepton_MET', nbins)
        plotter.plot('MaxAbsDeltaPhi_Lepton_MET', nbins)
        plotter.plot('MinAbsDeltaEta_Lepton_MET', nbins)
        plotter.plot('MaxAbsDeltaEta_Lepton_MET', nbins)
        plotter.plot('MinMtW_Lepton_MET', nbins, units="GeV")
        plotter.plot('MaxMtW_Lepton_MET', nbins, units="GeV")

    # Two-lepton systems
    if N_lep >= 2:
        plotter.plot('LT', nbins, units="GeV")
        for i in range(N_lep):
            for j in range(i + 1, N_lep):
                plotter.plot(f'M_Lepton{i}_Lepton{j}', nbins, units="GeV")
                plotter.plot(f'DeltaR_Lepton{i}_Lepton{j}', nbins)
                plotter.plot(f'DeltaPhi_Lepton{i}_Lepton{j}', nbins)
                plotter.plot(f'DeltaEta_Lepton{i}_Lepton{j}', nbins)

    # Lepton-Lepton reduced (exists only when N_lep > 2)
    if N_lep > 2:
        plotter.plot('MinM_Lepton_Lepton', nbins, units="GeV")
        plotter.plot('MaxM_Lepton_Lepton', nbins, units="GeV")
        plotter.plot('MinDeltaR_Lepton_Lepton', nbins)
        plotter.plot('MaxDeltaR_Lepton_Lepton', nbins)
        plotter.plot('MinAbsDeltaPhi_Lepton_Lepton', nbins)
        plotter.plot('MaxAbsDeltaPhi_Lepton_Lepton', nbins)
        plotter.plot('MinAbsDeltaEta_Lepton_Lepton', nbins)
        plotter.plot('MaxAbsDeltaEta_Lepton_Lepton', nbins)

    # FatJets
    fj_variants = ["FatJet", "SoftDroppedFatJet"] if use_fatjet_variants else ["SoftDroppedFatJet"]
    for i in range(N_fj):
        plotter.plot(f'Tau1_FatJet{i}', nbins)
        plotter.plot(f'Tau2_FatJet{i}', nbins)
        plotter.plot(f'Tau3_FatJet{i}', nbins)
        plotter.plot(f'Tau21_FatJet{i}', nbins) 
        plotter.plot(f'Tau32_FatJet{i}', nbins)                    
        for fj in fj_variants:
            plotter.plot(f'Pt_{fj}{i}', nbins, units="GeV")
            plotter.plot(f'M_{fj}{i}', nbins, units="GeV")
            plotter.plot(f'M_{fj}{i}_MET', nbins, units="GeV") 
            plotter.plot(f'MtW_{fj}{i}_MET', nbins, units="GeV")
            plotter.plot(f'DeltaR_{fj}{i}_MET', nbins)
            plotter.plot(f'DeltaPhi_{fj}{i}_MET', nbins)
            plotter.plot(f'DeltaEta_{fj}{i}_MET', nbins)

    # FatJet-MET reduced (exists only when N_fj > 1)
    if N_fj > 1:
        plotter.plot(f'MinDeltaR_{fj_name}_MET', nbins)
        plotter.plot(f'MaxDeltaR_{fj_name}_MET', nbins)
        plotter.plot(f'MinAbsDeltaPhi_{fj_name}_MET', nbins)
        plotter.plot(f'MaxAbsDeltaPhi_{fj_name}_MET', nbins)
        plotter.plot(f'MinAbsDeltaEta_{fj_name}_MET', nbins)
        plotter.plot(f'MaxAbsDeltaEta_{fj_name}_MET', nbins)
        plotter.plot(f'MinMtW_{fj_name}_MET', nbins, units="GeV")
        plotter.plot(f'MaxMtW_{fj_name}_MET', nbins, units="GeV")

    # Lepton + FatJet systems
    for i in range(N_lep):      
        for j in range(N_fj):        
            for fj in fj_variants:
                plotter.plot(f'Pt_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'Phi_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'Eta_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'M_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'M_Lepton{i}_{fj}{j}_MET', nbins, units="GeV")
                plotter.plot(f'MtW_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'MT2_Lepton{i}_{fj}{j}', nbins, units="GeV")
                plotter.plot(f'DeltaR_Lepton{i}_{fj}{j}', nbins)
                plotter.plot(f'DeltaPhi_Lepton{i}_{fj}{j}', nbins)
                plotter.plot(f'DeltaEta_Lepton{i}_{fj}{j}', nbins)

    # Lepton-FatJet reduced (exists only when N_lep * N_fj > 1)
    if N_lep * N_fj > 1:
        plotter.plot(f'MaxDeltaR_Lepton_{fj_name}', nbins)
        plotter.plot(f'MinDeltaR_Lepton_{fj_name}', nbins)
        plotter.plot(f'MaxAbsDeltaPhi_Lepton_{fj_name}', nbins)
        plotter.plot(f'MinAbsDeltaPhi_Lepton_{fj_name}', nbins)
        plotter.plot(f'MaxAbsDeltaEta_Lepton_{fj_name}', nbins)
        plotter.plot(f'MinAbsDeltaEta_Lepton_{fj_name}', nbins)
        plotter.plot(f'M_MaxAbsDeltaPhi_Lepton_{fj_name}', nbins, units="GeV")
        plotter.plot(f'M_MinAbsDeltaPhi_Lepton_{fj_name}', nbins, units="GeV")
        plotter.plot(f'M_MaxDeltaR_Lepton_{fj_name}', nbins, units="GeV")
        plotter.plot(f'M_MinDeltaR_Lepton_{fj_name}', nbins, units="GeV")

    # Two-FatJet systems (pair-level, meaningful only for exactly 2 fatjets)
    if N_fj == 2:
        suffx = "SoftDroppedFatJet0_SoftDroppedFatJet1"
        plotter.plot(f'M_{suffx}', nbins, units="GeV")
        plotter.plot(f'MT2_{suffx}', nbins, units="GeV")
        plotter.plot(f'DeltaR_{suffx}', nbins)
        plotter.plot(f'DeltaEta_{suffx}', nbins)
        plotter.plot(f'DeltaPhi_{suffx}', nbins)

    # FatJet-FatJet reduced (exists only when N_fj > 2)
    if N_fj > 2:
        plotter.plot(f'MinM_{fj_name}_{fj_name}', nbins, units="GeV")
        plotter.plot(f'MaxM_{fj_name}_{fj_name}', nbins, units="GeV")
        plotter.plot(f'MinDeltaR_{fj_name}_{fj_name}', nbins)
        plotter.plot(f'MaxDeltaR_{fj_name}_{fj_name}', nbins)
        plotter.plot(f'MinAbsDeltaPhi_{fj_name}_{fj_name}', nbins)
        plotter.plot(f'MaxAbsDeltaPhi_{fj_name}_{fj_name}', nbins)
        plotter.plot(f'MinAbsDeltaEta_{fj_name}_{fj_name}', nbins)
        plotter.plot(f'MaxAbsDeltaEta_{fj_name}_{fj_name}', nbins)

def draw_feature_histograms_mutlipleChannels():

    channels = {
        "1L/lepJ":       "1L-1J",
        "1L/lepJJ":      "1L-2J",
        "2OSL/leplepJ":  "2OSL-1J",
        "2SSL/leplep":   "2SSL-0J",
        "2SSL/leplepJ":  "2SSL-1J",
        "3L/lepleplep":  "3L",
    }

    def get_total_df_sig(m = 1000, filename = "signal_preselected.parquet", _dir = "PreparedData"):
        parts = []
        for ch, chName in channels.items():
            p = pd.read_parquet(output_data_dir / ch / _dir / f"M{m}" / filename, engine='pyarrow')
            p["channel"] = chName
            parts.append(p)
        return pd.concat(parts, ignore_index=True)

    def get_total_df_bkg(filename = "background_preselected.parquet", _dir = "PreparedData"):
        parts = []
        for ch, chName in channels.items():
            p = pd.read_parquet(output_data_dir / ch / _dir / filename, engine='pyarrow')
            p["channel"] = chName
            parts.append(p)
        return pd.concat(parts, ignore_index=True)
    
    def update(plt, col):
            valid_channels = sorted(df_sig.loc[df_sig[col].notna(), "channel"].unique())
            plt.labels += [", ".join(valid_channels[i:i+3]) + ("," if i+3 < len(valid_channels) else "") for i in range(0, len(valid_channels), 3)]

    df_sig = pd.concat([get_total_df_sig(m=1000),  get_total_df_sig(m=1200),  get_total_df_sig(m=1400)], ignore_index=True)
    df_bkg = get_total_df_bkg()

    plt = plot_feature(df_sig, df_bkg)
    plt.channel = f"All Channels"
    plt.draw_significance_pad = False
    plt.style["canvas"]["log_y"] = True
    plt.style["canvas"]["width"] = 900
    plt.style["canvas"]["height"] = 800
    plt.style["legends"]["TextSize"] = 0.035
    plt.regroup_bkg_proc_names = True
    plt.regroup_sig_proc_names = True
    plt.sig_plot_scale = 10.0
    plt.include_overflow = True
    plt.show_auc_and_bestcut = False
    plt.show_event_count_on_legends = False
    plt.draw_histograms_formats = ["png", "pdf"]
    plt.output_dir = output_data_dir / "histograms"

    plt.labels += ["Preselection Cuts Applied"]

    base_labels = list(plt.labels)

    plt.labels = list(base_labels)
    update(plt, 'Pt_Lepton0')
    plt.plot('Pt_Lepton0', nBins =30, xMax=1800, units="GeV", xTitle="Leading Lepton p_{T}")

    plt.labels = list(base_labels)
    update(plt, 'Pt_SoftDroppedFatJet0')
    plt.plot(f'Pt_SoftDroppedFatJet0', nBins =30, xMax=1800, units="GeV", xTitle="Leading SoftDropped Ak10-Jet p_{T}")

    plt.labels = list(base_labels)
    update(plt, 'MET_MET')
    plt.plot(f'MET_MET', nBins =30, xMax=1000, units="GeV", xTitle="Missing Transverse Momentum")

    plt.labels = list(base_labels)
    update(plt, 'HT')
    plt.plot(f'HT', nBins =30, xMax=4500, units="GeV")

    plt.labels = list(base_labels)
    update(plt, 'ST')
    plt.plot(f'ST', nBins =30, xMax=6000, units="GeV")

    plt.labels = list(base_labels)
    update(plt, 'Meff')
    plt.plot(f'Meff', nBins =30, xMax=6000, units="GeV")

    plt.style["canvas"]["max_y_scale"] = 1e7

    plt.labels = list(base_labels)
    update(plt, 'M_Lepton0_SoftDroppedFatJet0')
    plt.plot(f'M_Lepton0_SoftDroppedFatJet0', nBins =30, xMax=2800, units="GeV", xTitle="#it{m_{inv}}(l_{1}, J^{SD}_{1})")

    plt.labels = list(base_labels)
    update(plt, 'MtW_Lepton0_MET')
    plt.plot(f'MtW_Lepton0_MET', nBins =30, xMax=2500, units="GeV", xTitle="#it{m_{T}}(l_{1}, #vec{p}_{T}^{ miss})")

    plt.labels = list(base_labels)
    update(plt, 'MtW_SoftDroppedFatJet0_MET')
    plt.plot(f'MtW_SoftDroppedFatJet0_MET', nBins =30, xMax=2800, units="GeV", xTitle="#it{m_{T}}(J^{SD}_{1}, #vec{p}_{T}^{ miss})")

    plt.style["canvas"]["max_y_scale"] = 1e10

    plt.labels = list(base_labels)
    update(plt, 'Tau21_FatJet0')
    plt.plot(f'Tau21_FatJet0', nBins =30)

    plt.labels = list(base_labels)
    update(plt, 'Tau32_FatJet0')
    plt.plot(f'Tau32_FatJet0', nBins =30)

    # take absolute value of all DeltaPhi features before drawing them 
    for df in (df_sig, df_bkg):
        dphi = df.filter(like='DeltaPhi_').columns
        df[dphi] = df[dphi].abs()

    plt.labels = list(base_labels)
    update(plt, 'DeltaPhi_Lepton0_SoftDroppedFatJet0')
    plt.plot(f'DeltaPhi_Lepton0_SoftDroppedFatJet0', nBins =30, units="rad", xTitle="|#Delta#it{#phi}(l_{1}, J^{SD}_{1})|")

    plt.labels = list(base_labels)
    update(plt, 'DeltaPhi_Lepton0_MET')
    plt.plot(f'DeltaPhi_Lepton0_MET', nBins =30, units="rad", xTitle="|#Delta#it{#phi}(l_{1}, #vec{p}_{T}^{ miss})|")

    plt.labels = list(base_labels)
    update(plt, 'DeltaPhi_SoftDroppedFatJet0_MET')
    plt.plot(f'DeltaPhi_SoftDroppedFatJet0_MET', nBins =30, units="rad", xTitle="|#Delta#it{#phi}(J^{SD}_{1}, #vec{p}_{T}^{ miss})|")

def apply_drawing_options_for_histogram_debugging(plotter, get_debug_report=False):
    plotter.debug_histogram = get_debug_report
    plotter.draw_significance_pad = True
    plotter.style["canvas"]["min_y"] = 1e-3
    plotter.style["canvas"]["max_y_scale"] = 1e12
    plotter.style["canvas"]["log_y"] = True
    plotter.style["canvas"]["width"] = 1200
    plotter.style["canvas"]["height"] = 800
    plotter.style["legends"]["Y1"] = 0.3
    plotter.style["legends"]["TextSize"] = 0.35
    plotter.regroup_bkg_proc_names = False
    plotter.regroup_sig_proc_names = False
    plotter.show_auc_and_bestcut = True
    plotter.show_event_count_on_legends = True
    plotter.draw_histograms_formats = ["png"]
    return plotter

def apply_drawing_options_for_histogram_publication(plotter):
    plotter.debug_histogram = True
    plotter.regroup_bkg_proc_names = True
    plotter.regroup_sig_proc_names = True
    plotter.show_auc_and_bestcut = False
    plotter.include_overflow = True
    plotter.show_event_count_on_legends = False
    plotter.draw_histograms_formats = ["png", "pdf", "root"]
    plotter.draw_significance_pad = False
    plotter.set_color_palette = ROOT.kViridis
    plotter.style["canvas"]["min_y"] = 9e-1
    plotter.style["canvas"]["log_y"] = True
    plotter.style["canvas"]["width"] = 850
    plotter.style["canvas"]["height"] = 720
    plotter.style["canvas"]["max_y_scale"] = 1e4
    plotter.style["canvas"]["max_y_scale"] = 1e4
    return plotter

if __name__ == '__main__':
    plot_significance_reduction_ratio()