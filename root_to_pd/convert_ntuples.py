import pathlib
import sys

import pandas as pd
import psutil
import uproot
import ROOT

MB = 1024 * 1024
GB = MB * 1024

# setups
data_dir = "./data"
ntup_dir = [pathlib.Path(f"/eos/home-z/zhuyi/VBS/ZZ2l2v_minitrees_Oct2021/PFLOW/"), 
            pathlib.Path(f"/eos/user/z/ziwang/ZZ2l2v_minitrees_Oct2021/PFLOW/"), 
            pathlib.Path(f"/eos/user/k/kordas/ZZ2l2v_minitrees_Oct2021/PFLOW/"), 
            pathlib.Path(f"/eos/user/d/desampso/ZZ2l2v_minitrees_Oct2021/PFLOW/")]
#ntup_dir_bkg = pathlib.Path(f"/eos/home-z/zhuyi/VBS/ZZ2l2v_minitrees_Oct2021/PFLOW/")
#ntup_dir_sig = pathlib.Path(f"/eos/user/z/ziwang/ZZ2l2v_minitrees_Oct2021/PFLOW/")
df_dir = pathlib.Path(f"{data_dir}/data_frames")
df_dir.mkdir(parents=True, exist_ok=True)

feature_list = [
    "dMetZPhi",
    "met_signif",
    "MetOHT",
    "met_tst",
    "leading_pT_lepton",
    "subleading_pT_lepton",
    "M2Lep",
    "dLepR",
    "Z_pT",
    "Z_rapidity",
    "mT_ZZ",
    "n_jets",
    "n_bjets",
    "ZpTomT",
    "sumpT_scalar",
    "lepplus_eta",
    "lepplus_phi",
    "lepminus_eta",
    "lepminus_phi",
    "weight",
]

categorys = {
    "ww"	    :[361600, 361606, 345324, 345718],
    "ttbar"	    :[410472],
    "single-top":[410644, 410645, 410648, 410649, 410658, 410659],
    "ttV"   	:[410081, 410155, 410156, 410157],
    "VVV"	    :[364242, 364243, 364244, 364245, 364246, 364247, 364248, 364249, 363508, 363509],
    "Ztautau"	:[364128, 364129, 364130, 364131, 364132, 364133, 364134, 364135, 364136, 364137, 364138, 364139, 364140, 364141],
    "WZ"	    :[364253, 364284, 363358],
    "Zjet"	    :[364100, 364101, 364102, 364103, 364104, 364105, 364106, 364107, 364108, 364109, 364110, 364111, 364112, 364113, 364114, 364115, 364116, 364117, 364118, 364119, 364120, 364121, 364122, 364123, 364124, 364125, 364126, 364127],
    "ZZ2q2l"	:[363356],
    "ZZ4l"	    :[364250, 364283, 345706],
    "ZZ2l2v"	:[345666, 345723, 364285]
}



#bkg_names = ["364253.Sherpa_222_NNPDF30NNLO_lllv.deriv.DAOD_STDM3.e5916_s3126_r10201_p4252", 
#             "364253.Sherpa_222_NNPDF30NNLO_lllv.deriv.DAOD_STDM3.e5916_s3126_r10724_p4252",
#             "364253.Sherpa_222_NNPDF30NNLO_lllv.deriv.DAOD_STDM3.e5916_s3126_r9364_p4252"]
#read bkgs from files
with open("/afs/cern.ch/work/c/chuanshu/public/hepynet/hepynet_example/root_to_pd/bkg.list") as file:
    bkg_names = [line.strip() for line in file]
#print(bkg_names)

sig_names = ["345666.Sherpa_222_NNPDF30NNLO_llvvZZ.deriv.DAOD_STDM3.e6240_s3126_r10201_p4252", 
             "345666.Sherpa_222_NNPDF30NNLO_llvvZZ.deriv.DAOD_STDM3.e6240_s3126_r10724_p4252", 
             "345666.Sherpa_222_NNPDF30NNLO_llvvZZ.deriv.DAOD_STDM3.e6240_s3126_r9364_p4252", 
             "345723.Sherpa_222_NNPDF30NNLO_ggllvvZZ.deriv.DAOD_STDM3.e6213_s3126_r10201_p4252", 
             "345723.Sherpa_222_NNPDF30NNLO_ggllvvZZ.deriv.DAOD_STDM3.e6213_s3126_r10724_p4252", 
             "345723.Sherpa_222_NNPDF30NNLO_ggllvvZZ.deriv.DAOD_STDM3.e6213_s3126_r9364_p4252", 
             "364285.Sherpa_222_NNPDF30NNLO_llvvjj_EW6.deriv.DAOD_STDM3.e6055_s3126_r10201_p4252", 
             "364285.Sherpa_222_NNPDF30NNLO_llvvjj_EW6.deriv.DAOD_STDM3.e6055_s3126_r10724_p4252", 
             "364285.Sherpa_222_NNPDF30NNLO_llvvjj_EW6.deriv.DAOD_STDM3.e6055_s3126_r9364_p4252"]

# convert to pandas DataFrame
def process_sample(sample_name, sample_path, is_sig, is_mc, channel, factor, category, camp=None):
    print(f"Processing: {sample_name}")
    sample_dfs = list()
    for chunk_pd in uproot.iterate(
        f"{sample_path}:tree_PFLOW",
        feature_list,
        #cut=f"(leading_pT_lepton > 30) & (subleading_pT_lepton > 20) & (met_tst > 70) & (M2Lep > 76) & (M2Lep < 106) & (n_bjets == 0) & (dLepR < 1.8) & (dMetZPhi > 2.3) & (MetOHT > 0.5) &(met_signif > 10)",
        cut=f"(leading_pT_lepton > 30) & (subleading_pT_lepton > 20) & (met_tst > 70) & (M2Lep > 76) & (M2Lep < 106) & (n_bjets < 1) & (dLepR < 2.2) & (dMetZPhi > 1.3) & (MetOHT > 0) &(met_signif > 7)",
        library="pd",
        step_size="200 MB",
    ):
        mem_available = psutil.virtual_memory().available / GB
        mem_total = psutil.virtual_memory().total / GB
        print(
            f"RAM usage {mem_available:.02f} / {mem_total:.02f} GB",
            end="\r",
            flush=True,
        )
        # convert float64 to float32
        #f64_cols = chunk_pd.select_dtypes(include="float64").columns
        #chunk_pd[f64_cols] = chunk_pd[f64_cols].astype("float32")
        # add necessary tags
        chunk_pd = chunk_pd.assign(sample_name=sample_name)  # required
        chunk_pd = chunk_pd.assign(is_sig=is_sig)  # required
        chunk_pd = chunk_pd.assign(is_mc=is_mc)  # required
        # add other tags (later you can add some cuts before training based on these tags)
        chunk_pd = chunk_pd.assign(factor=factor)
        chunk_pd = chunk_pd.assign(category=category)
        chunk_pd = chunk_pd.assign(camp=camp)
        # update df list
        sample_dfs.append(chunk_pd)
    sys.stdout.write("\033[K")
    return sample_dfs


def factor(sample_name):
    sample_name_str = str(sample_name)
    inFile = ROOT.TFile.Open(sample_name_str, "READ")
    hInfo_N = inFile.Get("hInfo").GetEntries()
    file = uproot.open(sample_name)
    hInfo = file["hInfo"].values(flow=False)
    if sample_name_str.find("r9364") != -1:
        lumi = (3.21956+32.9653)
    elif sample_name_str.find("r10201") != -1:
        lumi = 44.3074
    elif sample_name_str.find("r10724") != -1:
        lumi = 58.4501
    else:
        print("No date info found!") 
        lumi = 0
    xsec = 2 * hInfo[0]/hInfo_N
    if hInfo[2] > 0 : xsec = 1.5 * xsec
    return lumi * xsec / hInfo[1]

def category(sample_name):
    sample_name_str = str(sample_name)
    for key in categorys.keys():
        for num in categorys[key]:
            if sample_name_str.find(str(num)) != -1: 
                return key
    print(sample_name_str, "doesn't find category!")
    return "NotFound"

## mm channel
df_list = list()
save_dir = df_dir 
save_dir.mkdir(parents=True, exist_ok=True)
### bkg
for bkg_name in bkg_names:
    for dir in ntup_dir:
        root_path = dir / f"{bkg_name}.root"
        if root_path.is_file(): break
    if root_path.is_file():
        df_list += process_sample(bkg_name, root_path, False, True, "n_jets", factor=factor(root_path), category=category(bkg_name), camp="run2")
    else:
        print(f"{bkg_name} does not exist!")
### sig
for sig_name in sig_names:
    for dir in ntup_dir:
        root_path = dir / f"{sig_name}.root"
        if root_path.is_file(): break
    if root_path.is_file():
        df_list += process_sample(sig_name, root_path, True, True, "n_jets", factor=factor(root_path), category=category(sig_name), camp="run2")
    else:
        print(f"{sig_name} does not exist!")
### dum
df = pd.concat(df_list, ignore_index=True)
df["weight_lumi"]=df["weight"] * df["factor"]
df["LepRatio"]=df["leading_pT_lepton"] / df["subleading_pT_lepton"]
df["RhoZ"]=df["Z_pT"]/(df["leading_pT_lepton"] + df["subleading_pT_lepton"])
df["dLepEta"]=df["lepplus_eta"] - df["lepminus_eta"]
df["dLepEtaAbs"]=df["dLepEta"].abs()
df["dLepPhi"]=df["lepplus_phi"] - df["lepminus_phi"]
df["dLepPhiAbs"]=df["dLepPhi"].abs()
df=df.rename(columns={"weight": "weight_raw", "weight_lumi": "weight"})
### Calculate the yields
print("Total Events: ", df.shape[0])
print("Yields: \n")
print("Category: weight_raw, weight, events\n")
for key in categorys.keys():
    df_temp = df[df["category"] == key]
    print("{0}: {1}, {2}, {3}".format(key, df_temp["weight_raw"].sum(), df_temp["weight"].sum(), df_temp.shape[0]))
save_path = save_dir / "input.feather.relax"
print(f"## Saving to {save_path}")
df.to_feather(save_path)  # feather is the default format for now
