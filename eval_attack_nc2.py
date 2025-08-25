#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for Neural Cleanse (your eval_attack_nc.py).
- Loops over multiple poisoned checkpoints and runs the NC script once per checkpoint.
- After each run, it parses the generated manifest.json and collects a summary table.

Usage:
  python eval_attack_nc2.py

Requirements:
- Place this file next to your NC script (default name: eval_attack_nc.py). If the name/path differs,
  change the SCRIPT variable below.
- Your NC script already has nc_epoch=5 set (as in your last version).
"""

import os
import sys
import json
import time
import subprocess
from collections import OrderedDict
from pathlib import Path

# ============== CONFIG ==============
# Path to your NC script (the one you pasted above)
SCRIPT = Path(__file__).parent / "eval_attack_nc.py"
PY = sys.executable  # current python interpreter

# Checkpoints to evaluate (edit paths as needed)
CHECKPOINTS = OrderedDict([
    # ("blend",   "/mnt/sdb/models/train_attack_blend_resnet_celeba_0.1_blend_no_smooth_epoch48.pt"),
    # ("sig",     "/mnt/sdb/models/train_attack_sig_resnet_celeba_0.1_sig_no_smooth_epoch50.pt"),
    # ("square",  "/mnt/sdb/models/train_attack_square_resnet_celeba_0.1_square_no_smooth_epoch30.pt"),
    # ("ftrojan", "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"),
    # ("HCBsmile","/mnt/sdb/models/train_attack_HCBsmile_resnet_celeba_0.1_HCBsmile_no_smooth_epoch3.pt"),
    ("HCB","/mnt/sdb/models/train_attack_HCB_resnet_celeba_0.1_HCB_no_smooth_epoch14.pt"),
])

# Optional: poisoned data table (not used by NC, kept here for future ASR checks)
POISON_TABLE = (
    # ("HCBsmile", "./saved_dataset",             "poisoned_test_batch_*.pt"),
    # ("blend",    "./saved_dataset",             "resnet_blend_poisoned_test_batch_*.pt"),
    # ("sig",      "./saved_dataset",             "resnet_sig_poisoned_test_batch_*.pt"),
    # ("square",   "/mnt/sdb/dataset_checkpoint", "resnet_square_poisoned_test_batch_*.pt"),
    # ("ftrojan",  "/mnt/sdb/dataset_checkpoint", "resnet_ftrojan_poisoned_test_batch_*.pt"),
    ("HCB",  "/mnt/sdb/dataset_checkpoint", "resnet_HCB_poisoned_test_batch_*.pt"),
)
# python eval_attack_nc.py --ckpt "/mnt/sdb/models/train_attack_ftrojan_resnet_celeba_0.1_ftrojan_no_smooth_epoch44.pt"
# Root dir where NC saves results (must match your NC script)
NC_ROOT = Path("nc/results")

# ============== Helpers ==============

def run_one(attack_name: str, ckpt_path: str):
    ckpt_path = str(ckpt_path)
    ckpt_name = Path(ckpt_path).name.replace(".pt", "")

    out_dir = NC_ROOT / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw console log per run
    log_file = out_dir / f"nc_{attack_name}.log"
    cmd = [str(PY), str(SCRIPT), "--ckpt", ckpt_path]

    print(f"\n=== [{attack_name}] Running: {' '.join(cmd)}")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:  # tee to console & log
            sys.stdout.write(line)
            lf.write(line)
        ret = proc.wait()
    if ret != 0:
        print(f"[WARN] Script returned non-zero exit code: {ret}")

    # Parse manifest.json for a concise summary
    manifest = out_dir / "manifest.json"
    summary = {
        "attack": attack_name,
        "ckpt_name": ckpt_name,
        "status": "ok" if manifest.exists() else "missing_manifest",
        "top_class": None,
        "top_ai": None,
        "median_l1": None,
    }
    if manifest.exists():
        try:
            with open(manifest, "r") as f:
                man = json.load(f)
            top = man.get("mad_top_suspicious", {})
            summary["top_class"] = top.get("class")
            summary["top_ai"] = top.get("ai")
            # compute median L1 from per_class if present
            pcs = man.get("per_class", [])
            l1s = [p.get("mask_l1") for p in pcs if "mask_l1" in p]
            if l1s:
                l1s_sorted = sorted(l1s)
                m = l1s_sorted[len(l1s_sorted)//2] if len(l1s_sorted) % 2 == 1 else 0.5*(l1s_sorted[len(l1s_sorted)//2 - 1] + l1s_sorted[len(l1s_sorted)//2])
                summary["median_l1"] = m
        except Exception as e:
            summary["status"] = f"parse_error: {e}"
    else:
        print(f"[WARN] manifest not found: {manifest}")

    return summary


def main():
    t0 = time.time()
    all_rows = []

    print(f"Using NC script: {SCRIPT}")
    if not Path(SCRIPT).exists():
        print("[ERROR] NC script not found. Please set SCRIPT to your eval_attack_nc.py path.")
        sys.exit(1)

    for name, ckpt in CHECKPOINTS.items():
        row = run_one(name, ckpt)
        all_rows.append(row)

    # Write batch summary CSV
    NC_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = NC_ROOT / "batch_summary.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["attack", "ckpt_name", "status", "top_class", "top_ai", "median_l1"])
        for r in all_rows:
            w.writerow([r.get("attack"), r.get("ckpt_name"), r.get("status"), r.get("top_class"), r.get("top_ai"), r.get("median_l1")])

    print(f"\nSaved batch summary to: {csv_path}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()




