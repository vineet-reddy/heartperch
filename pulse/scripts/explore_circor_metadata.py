"""
This script loads all .hea and .txt files from CirCor DigiScope Heart Sound Dataset.
It then extracts every metric contained in each .hea file (e.g., sampling frequency, duration, signal count, gain, baseline, units, etc.) 
and compares them across all others to identify which metrics vary and which remain constant.

It also parses each patient’s .txt file to grab sampling rate and file map, then checks values match corresponding .hea files.

A .hea file (header file for physiological recordings) contains the metadata for each heart sound .wav file.

For information on all metrics stored in .hea files, see:
https://physionet.org/physiotools/wag/header-5.htm
"""
import os, glob, re
import statistics as stats
import wfdb

# point this to training_data directory within circor heart sound dataset
DATA_DIR = "/Users/vineetreddy/Documents/GitHub/heartperch_data/data/training_data"

# TXT parsing (patient-level index)
def load_txt_index():
    idx = {}
    for p in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        head = lines[0].split()
        pid = head[0]
        fs = float(head[2]) if len(head) >= 3 else None
        heas = set()
        for ln in lines[1:]:
            if ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) >= 2:
                heas.add(parts[1])  # e.g., "2530_AV.hea"
        idx[pid] = {"fs": fs, "heas": heas}
    return idx

# HEA parsing (using official WFDB)
def parse_hea(hea_path):
    base = os.path.splitext(hea_path)[0]
    h = wfdb.rdheader(base)
    fs = h.fs
    n_sig = h.n_sig
    sig_len = h.sig_len
    duration = (sig_len / fs) if (fs and sig_len) else None
    
    # NOTE: The .hea files don't actually specify units. When units are omitted, the WFDB library 
    # automatically defaults to "mV" and returns that in h.units. This is misleading for audio data because:
    # 1) Physical heart sounds are pressure waves measured in Pascals (Pa)
    # 2) Digital audio samples are dimensionless integers (arbitrary units) 
    # 3) Without calibration data, we cannot convert samples to physical units
    # Thus, the "mV" shown in output is NOT the actual units, it's just WFDB's default placeholder.
    units = ";".join(h.units) if getattr(h, "units", None) else None
    
    sig_name = ";".join(h.sig_name) if h.sig_name else None
    m = re.search(r"(\d+)_([A-Z]{2})$", os.path.basename(base))
    pid = m.group(1) if m else None
    return {
        "file": os.path.basename(hea_path),
        "patient_id": pid,
        "fs": fs,
        "n_sig": n_sig,
        "units": units,
        "sig_len": sig_len,
        "duration_sec": duration,
        "sig_name": sig_name,
    }

def summarize_metric(name, values, meaning):
    uniq = set(values)
    if len(uniq) == 1:
        val = next(iter(uniq))
        formatted = f"{val:.6g}" if isinstance(val, (int, float)) else str(val)
        print(f"{name}: {formatted} <— {meaning}")
        return
    
    # Try numeric stats; fall back to listing unique values if non-numeric
    try:
        nums = [float(v) for v in values if v is not None]
        if nums:
            print(f"{name}: multiple values <— {meaning}")
            print(f"  min={min(nums):.6g}, max={max(nums):.6g}, mean={stats.fmean(nums):.6g}, median={stats.median(nums):.6g}")
        else:
            raise ValueError
    except (ValueError, TypeError):
        uniq_str = sorted(str(v) for v in uniq)
        print(f"{name}: {len(uniq_str)} different values <— {meaning}")
        print(f"  values: {', '.join(uniq_str[:50])}")

def main():
    txt_idx = load_txt_index()

    metrics = {
        "fs": [],
        "n_sig": [],
        "units": [],
        "sig_len": [],
        "duration_sec": [],
    }

    mismatches = 0

    for hea in glob.glob(os.path.join(DATA_DIR, "*.hea")):
        r = parse_hea(hea)
        # collect metrics
        for k in metrics:
            metrics[k].append(r[k])
        # txt cross-checks (if we have that patient’s txt)
        t = txt_idx.get(r["patient_id"])
        if t:
            # sampling rate check
            if (t["fs"] is not None) and (r["fs"] is not None) and abs(t["fs"] - r["fs"]) > 1e-6:
                mismatches += 1
            # file-map check (txt should list this .hea)
            if r["file"] not in t["heas"]:
                mismatches += 1

    # concise summary
    print("\n=== HEA summary (key metrics) ===")
    summarize_metric("fs (Hz)", metrics["fs"], "sampling frequency")
    summarize_metric("n_sig", metrics["n_sig"], "number of channels")
    summarize_metric("units", metrics["units"], "signal units (WFDB default, not actual)")
    summarize_metric("sig_len (samples)", metrics["sig_len"], "length in samples")
    summarize_metric("duration_sec", metrics["duration_sec"], "length in seconds")

    print("\nTXT mismatches happened?:", "Yes" if mismatches else "No")

if __name__ == "__main__":
    main()
