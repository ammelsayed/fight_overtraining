#!/usr/bin/env python3
import ROOT
import argparse
import pandas as pd
import tempfile
from pathlib import Path
from collections import defaultdict
ROOT.EnableImplicitMT()
ROOT.gROOT.SetBatch(True)

# # Keep only ROOT errors; suppress INFO, PROGRESS, DEBUG, and WARNING messages.
# msg = ROOT.RooMsgService.instance()
# msg.setGlobalKillBelow(ROOT.RooFit.ERROR)

# # Suppress Minuit minimizer printing.
# ROOT.Math.MinimizerOptions.SetDefaultPrintLevel(-1)

class HistFactoryHelper():

    def __init__(self, channel_name, root_files):
        self.channel_name = channel_name
        self.root_files= root_files
        self.bkg_sys_unc = 0.0
        self.lumi_unc = 0.0
        print(f"You are working with a flat background systematics error of {self.bkg_sys_unc*100:.2f}%")
        print(f"You are working with a luminosity systematics error of {self.lumi_unc*100:.2f}%")
        
        self.tmp_parent = Path(__file__).resolve().parent / "HistFactoryTempFiles"
        self.tmp_parent.mkdir(parents=True, exist_ok=True)
        self.tmpdir = tempfile.mkdtemp(prefix="tmp_", dir=self.tmp_parent)

    def make_channel(self, channel_name, root_file, bkg_hist_name="hist_bkg", sig_hist_name="hist_sig_1"):
        ch = ROOT.RooStats.HistFactory.Channel(channel_name)
        ch.SetData("hist_bkg", root_file)
        # Set signal
        sig = ROOT.RooStats.HistFactory.Sample("signal", sig_hist_name, root_file)
        sig.SetNormalizeByTheory(True)
        sig.AddNormFactor("mu", 1.0, 0.0, 15.0)
        ch.AddSample(sig)
        # Set background
        bkg = ROOT.RooStats.HistFactory.Sample("background", bkg_hist_name, root_file)
        bkg.SetNormalizeByTheory(True)
        bkg.ActivateStatError()
        bkg.AddOverallSys("bkg_norm", 1.0-self.bkg_sys_unc, 1.0+self.bkg_sys_unc)
        ch.AddSample(bkg)
        return ch
    
    def model(self):
        meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")
    
        meas.SetPOI("mu")
        meas.SetLumi(1.0)
        meas.SetLumiRelErr(self.lumi_unc)
        # Add channels
        for index, root_file in enumerate(self.root_files):
            meas.AddChannel(self.make_channel(f"ch_{self.channel_name}_{index}", root_file))
        meas.CollectHistograms()

        meas.SetOutputFilePrefix(str(Path(self.tmpdir) / "hf"))
        ws = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)
        sb = ws.obj("ModelConfig")
        b = ROOT.RooStats.ModelConfig("ModelConfig_bonly", ws)
        b.SetPdf(sb.GetPdf())
        b.SetObservables(sb.GetObservables())
        b.SetParametersOfInterest(sb.GetParametersOfInterest())
        b.SetNuisanceParameters(sb.GetNuisanceParameters())
        b.SetGlobalObservables(sb.GetGlobalObservables())
        mu = ws.var("mu")
        mu.setVal(1.0)
        sb.SetSnapshot(ROOT.RooArgSet(mu))
        mu.setVal(0.0)
        b.SetSnapshot(ROOT.RooArgSet(mu))
        return ws, sb, b

    def get_results(self):
        ws, sb, b = self.model()
        data = ws.data("obsData")
        # Expected discovery significance: s+b Asimov data tested against b-only.
        ac0 = ROOT.RooStats.AsymptoticCalculator(data, sb, b)
        ac0.SetOneSidedDiscovery(True)
        globobs = ROOT.RooArgSet(sb.GetGlobalObservables())
        asimov = ROOT.RooStats.AsymptoticCalculator.MakeAsimovData(data, sb, sb.GetSnapshot(), globobs)
        ac = ROOT.RooStats.AsymptoticCalculator(asimov, sb, b)
        ac.SetOneSidedDiscovery(True)
        z = ac.GetHypoTest().Significance()

        # Median expected CLs limit 
        # For discovery: alt=s+b, null=b. For an upper limit, test s+b as null
        # against b as alternate (the ordering used by ROOT's inversion tutorial).
        aclim = ROOT.RooStats.AsymptoticCalculator(data, b, sb)
        aclim.SetOneSided(True)
        inv = ROOT.RooStats.HypoTestInverter(aclim)
        inv.SetConfidenceLevel(0.95)
        inv.UseCLs(True)
        inv.SetFixedScan(101, 0.0, 15.0)
        r = inv.GetInterval()
        return [z] + [r.GetExpectedUpperLimit(i) for i in [0,-1,1,-2,2]]


def create_input_dict(input_dir = "./AnalysisOutput_Test", root_file_name = "publ_bdt_score_histograms.root"):
    input_dir  =  Path(input_dir).resolve()

    options = {
        "1L" : {
            "lepJ" : {
                "Nbins" : 7
            },
            "lepJJ" : {
                "Nbins" : 5
            }
        },
        "2OSL" : {
            "leplepJ" : {
                "Nbins" : 5
            }
        },
        "2SSL" : {
            "leplep" : {
                "Nbins" : 10
            },
            "leplepJ" : {
                "Nbins" : 7
            }
        },
        "3L" : {
            "lepleplep" : {
                "Nbins" : 10
            },
            "lepleplepJ" : {
                "Nbins" : 5
            }
        },
        "4L" : {
            "leplepleplep" : {
                "Nbins" : 3
            }
        }
    }

    signal_regions = {
        "1L"   : ["lepJ", "lepJJ"],
        "2OSL" : ["leplepJ"],
        "2SSL" : ["leplepJ", "leplep"],
        "3L"   : ["lepleplep", "lepleplepJ"],
        "4L"   : ["leplepleplep"]
    }
    M_min, M_max, M_step = 1000, 2000, 50
    M_list = range(M_min, M_max + M_step, M_step)

    in_dict = {}
    for m in M_list:
        files_by_channel = defaultdict(list)
        combined = []
        for ch, rgs in signal_regions.items():
            for rg in rgs:
                rootFile = input_dir / ch / rg / "bdt_strategy_A/featLevel_6" / f"M{m}" / "VariedNbins" / f"{options[ch][rg]["Nbins"]}_bins" / root_file_name
                files_by_channel[ch].append(str(rootFile))
                combined.append(str(rootFile))
                if len(rgs) > 1:
                    files_by_channel[f"{ch}_{rg}"].append(str(rootFile))
        files_by_channel["Combined"] = combined
        in_dict[m] = dict(files_by_channel)
    
    return in_dict

def main():
    parser = argparse.ArgumentParser(description="Run HistFactory expected-significance and CLs limits calculations.")
    parser.add_argument("-i", "--input-dir", default="./AnalysisOutput_Test", help="Directory containing the data")
    parser.add_argument("-o", "--output-dir", default="./HistFactoryResults", help="Directory where the per-channelresults are written.")
    parser.add_argument("--root-file-name", default="publ_bdt_score_histograms.root", help="Directory where the per-channelresults are written.")
    parser.add_argument("--bkg-sys-unc", type=float, default=0.0, help="Flat relative background systematic uncertainty.")
    parser.add_argument("--lumi-unc", type=float, default=0.0, help="Relative luminosity uncertainty.")
    args = parser.parse_args()

    in_dict = create_input_dict(args.input_dir, root_file_name=args.root_file_name)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ch_name in next(iter(in_dict.values())).keys():
        print(f"\nProcessing channel: {ch_name}")
        rows = []
        for m, data in in_dict.items():
            ch_files = data[ch_name]
            hf = HistFactoryHelper(ch_name, ch_files)
            hf.bkg_sys_unc = args.bkg_sys_unc
            hf.lumi_unc = args.lumi_unc
            res = hf.get_results()
            rows.append({
                "mass": m,
                "z_exp": res[0],
                "mu_exp": res[1],
                "mu_exp_m1s": res[2], 
                "mu_exp_p1s": res[3],  
                "mu_exp_m2s": res[4],  
                "mu_exp_p2s": res[5],  
            })
            print(f"Mass {m}: z_exp={res[0]:.4f}, mu_exp_m1s={res[2]:.4f}, mu_exp_p1s={res[3]:.4f}, mu_exp_m2s={res[4]:.4f}, mu_exp_p2s={res[5]:.4f}")
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / f"{ch_name}_sys{args.bkg_sys_unc*100:g}pct_lumi{args.lumi_unc*100:g}pct_results.csv", index=False)

if __name__ == '__main__':
    main()
