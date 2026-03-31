#!/usr/bin/env python3
"""
aezip_analysis.py — Analyse the quality of AEzip compression/decompression.

Compares an original trajectory against its reconstructed counterpart and
(optionally) the autoencoder's input vs output feature vectors.

Metrics
-------
  - Frame-wise RMSD       : cross-RMSD per frame (original vs reconstructed)
  - RMSD histogram        : distribution of frame-wise cross-RMSD
  - Per-atom RMSF         : fluctuation profiles (original vs reconstructed)
  - Per-residue mean RMSD : spatial accuracy per residue
  - AE feature error      : per-feature MAE in normalised feature space
                            (requires --ae-input and --ae-output)

Usage
-----
  python aezip_analysis.py \\
      -f original.xtc  -s original.pdb \\
      -r recon.xtc     -rp recon.pdb \\
      [--ae-input ae_in.npy  --ae-output ae_out.npy] \\
      [--selection "name CA"] \\
      [--stride N] \\
      [--output-dir ./analysis]
"""

import os
import sys
import json
import argparse
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aezip.analysis.diffio import calculate_rmsd_trajectories, calculate_rmsf_trajectories


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Analyse AEzip compression quality (RMSD, RMSF, AE feature error).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("-f",  "--orig-traj",    required=True,  help="Original trajectory (.xtc)")
parser.add_argument("-s",  "--orig-top",     required=True,  help="Original topology (.pdb)")
parser.add_argument("-r",  "--recon-traj",   required=True,  help="Reconstructed trajectory (.xtc)")
parser.add_argument("-rp", "--recon-pdb",    required=True,  help="Reconstructed topology (.pdb)")
parser.add_argument("--ae-input",  default=None, metavar="FILE",
                    help="AE input feature array (.npy, from compress.py --ae-input)")
parser.add_argument("--ae-output", default=None, metavar="FILE",
                    help="AE output feature array (.npy, from compress.py --ae-output)")
parser.add_argument("--selection", default="name CA",
                    help="MDTraj atom selection used for RMSD/RMSF calculations")
parser.add_argument("--stride", type=int, default=1, metavar="N",
                    help="Load every N-th frame from both trajectories")
parser.add_argument("--output-dir", default="./analysis",
                    help="Directory where plots and summary JSON are written")
parser.add_argument("--prefix", default="aezip",
                    help="Filename prefix for all output files")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
_pfx = os.path.join(args.output_dir, args.prefix)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _select_atoms(traj_orig, traj_recon, selection):
    """Return (n_frames, n_atoms, 3) arrays (Å) for a common atom selection."""
    idx_o = traj_orig.topology.select(selection)
    idx_r = traj_recon.topology.select(selection)
    if len(idx_o) == 0 or len(idx_r) == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms in one trajectory.")
    if len(idx_o) != len(idx_r):
        raise ValueError(
            f"Selection '{selection}' matched {len(idx_o)} atoms in original "
            f"but {len(idx_r)} in reconstructed — use a more specific selection."
        )
    xyz_o = traj_orig.atom_slice(idx_o).xyz * 10   # nm → Å
    xyz_r = traj_recon.atom_slice(idx_r).xyz * 10
    res_ids = np.array([a.residue.index
                        for a in traj_orig.atom_slice(idx_o).topology.atoms])
    return xyz_o, xyz_r, res_ids


def _residue_rmsd(xyz_o, xyz_r, res_ids):
    """Mean frame-wise RMSD per residue (Å)."""
    unique_res = np.unique(res_ids)
    rmsd_res   = np.zeros(len(unique_res))
    for i, rid in enumerate(unique_res):
        mask = res_ids == rid
        diff = xyz_o[:, mask, :] - xyz_r[:, mask, :]
        rmsd_res[i] = np.sqrt(np.nanmean(np.sum(diff ** 2, axis=2)))
    return unique_res, rmsd_res


# ---------------------------------------------------------------------------
# Load trajectories
# ---------------------------------------------------------------------------
print(f"\nLoading original   : {args.orig_traj}")
traj_orig  = md.load(args.orig_traj,  top=args.orig_top,  stride=args.stride)

print(f"Loading reconstructed: {args.recon_traj}")
traj_recon = md.load(args.recon_traj, top=args.recon_pdb, stride=args.stride)

n_frames = min(traj_orig.n_frames, traj_recon.n_frames)
traj_orig  = traj_orig[:n_frames]
traj_recon = traj_recon[:n_frames]
print(f"Frames compared: {n_frames}  |  selection: '{args.selection}'")

xyz_o, xyz_r, res_ids = _select_atoms(traj_orig, traj_recon, args.selection)
frames = np.arange(n_frames)

summary = {}

# ---------------------------------------------------------------------------
# 1. Frame-wise RMSD
# ---------------------------------------------------------------------------
print("\n[1/5] Frame-wise RMSD ...")
rmsd_frames = calculate_rmsd_trajectories(xyz_o, xyz_r)   # (n_frames,)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(frames, rmsd_frames, lw=0.8, color="steelblue", label="Cross-RMSD")
ax.axhline(np.mean(rmsd_frames), color="red", lw=1.2, ls="--",
           label=f"Mean {np.mean(rmsd_frames):.2f} Å")
ax.set_xlabel("Frame")
ax.set_ylabel("RMSD (Å)")
ax.set_title("Frame-wise RMSD: original vs reconstructed")
ax.legend()
_savefig(fig, f"{_pfx}_rmsd_frames.png")

summary["rmsd_mean_A"]   = round(float(np.mean(rmsd_frames)),  3)
summary["rmsd_std_A"]    = round(float(np.std(rmsd_frames)),   3)
summary["rmsd_max_A"]    = round(float(np.max(rmsd_frames)),   3)
summary["rmsd_median_A"] = round(float(np.median(rmsd_frames)),3)

# ---------------------------------------------------------------------------
# 2. RMSD histogram
# ---------------------------------------------------------------------------
print("[2/5] RMSD histogram ...")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(rmsd_frames, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
ax.axvline(np.mean(rmsd_frames), color="red", lw=1.5, ls="--",
           label=f"Mean {np.mean(rmsd_frames):.2f} Å")
ax.set_xlabel("RMSD (Å)")
ax.set_ylabel("Count")
ax.set_title("RMSD distribution: original vs reconstructed")
ax.legend()
_savefig(fig, f"{_pfx}_rmsd_hist.png")

# ---------------------------------------------------------------------------
# 3. Per-atom RMSF
# ---------------------------------------------------------------------------
print("[3/5] Per-atom RMSF ...")
rmsf_o = calculate_rmsf_trajectories(xyz_o, xyz_o)   # internal fluctuation
rmsf_r = calculate_rmsf_trajectories(xyz_r, xyz_r)
atom_indices = np.arange(len(rmsf_o))

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(atom_indices, rmsf_o, lw=0.8, color="steelblue", label="Original")
axes[0].plot(atom_indices, rmsf_r, lw=0.8, color="tomato",    label="Reconstructed", alpha=0.8)
axes[0].set_ylabel("RMSF (Å)")
axes[0].set_title(f"Per-atom RMSF  ({args.selection})")
axes[0].legend()

dio_rmsf = np.abs(rmsf_o - rmsf_r)
axes[1].bar(atom_indices, dio_rmsf, width=1.0, color="orange", label="|ΔRMSF|")
axes[1].set_xlabel("Atom index")
axes[1].set_ylabel("|ΔRMSF| (Å)")
axes[1].set_title("RMSF difference  (original − reconstructed)")
axes[1].legend()
_savefig(fig, f"{_pfx}_rmsf.png")

summary["rmsf_orig_mean_A"]  = round(float(np.mean(rmsf_o)),   3)
summary["rmsf_recon_mean_A"] = round(float(np.mean(rmsf_r)),   3)
summary["rmsf_delta_mean_A"] = round(float(np.mean(dio_rmsf)), 3)

# ---------------------------------------------------------------------------
# 4. Per-residue mean RMSD
# ---------------------------------------------------------------------------
print("[4/5] Per-residue RMSD ...")
res_ids_unique, rmsd_res = _residue_rmsd(xyz_o, xyz_r, res_ids)

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(res_ids_unique, rmsd_res, width=1.0, color="mediumseagreen")
ax.axhline(np.mean(rmsd_res), color="red", lw=1.2, ls="--",
           label=f"Mean {np.mean(rmsd_res):.2f} Å")
ax.set_xlabel("Residue index")
ax.set_ylabel("Mean RMSD (Å)")
ax.set_title(f"Per-residue mean RMSD  ({args.selection})")
ax.legend()
_savefig(fig, f"{_pfx}_residue_rmsd.png")

worst5 = res_ids_unique[np.argsort(rmsd_res)[-5:][::-1]].tolist()
summary["residue_rmsd_mean_A"] = round(float(np.mean(rmsd_res)), 3)
summary["residue_rmsd_worst5"] = worst5

# ---------------------------------------------------------------------------
# 5. AE feature-space reconstruction error  (optional)
# ---------------------------------------------------------------------------
if args.ae_input is not None and args.ae_output is not None:
    print("[5/5] AE feature error ...")
    x_in  = np.load(args.ae_input)
    x_out = np.load(args.ae_output)

    if x_in.shape != x_out.shape:
        print(f"  WARNING: shape mismatch {x_in.shape} vs {x_out.shape} — skipping AE plot")
    else:
        err          = np.abs(x_in - x_out)          # (n_frames, n_features)
        mae_per_feat = err.mean(axis=0)               # (n_features,)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(mae_per_feat, lw=0.7, color="mediumpurple")
        axes[0].set_xlabel("Feature index")
        axes[0].set_ylabel("MAE (normalised units)")
        axes[0].set_title(
            f"Per-feature AE reconstruction error\n"
            f"mean={mae_per_feat.mean():.4f}  std={mae_per_feat.std():.4f}"
        )

        axes[1].hist(err.ravel(), bins=80, color="mediumpurple",
                     edgecolor="white", linewidth=0.3)
        axes[1].set_xlabel("|x_in − x̂|")
        axes[1].set_ylabel("Count")
        axes[1].set_title("AE reconstruction error distribution")
        _savefig(fig, f"{_pfx}_ae_feature_error.png")

        summary["ae_feature_mae_mean"] = round(float(mae_per_feat.mean()), 5)
        summary["ae_feature_mae_std"]  = round(float(mae_per_feat.std()),  5)
        summary["ae_n_features"]       = int(x_in.shape[1])
        summary["ae_n_frames"]         = int(x_in.shape[0])
else:
    print("[5/5] AE feature error — skipped (no --ae-input / --ae-output)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary["n_frames"]  = n_frames
summary["selection"] = args.selection
summary["orig_traj"] = args.orig_traj
summary["recon_traj"]= args.recon_traj

summary_path = f"{_pfx}_summary.json"
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2)

w = 32
print("\n" + "─" * (w + 16))
print(f"  {'Metric':<{w}} {'Value':>12}")
print("─" * (w + 16))
print(f"  {'Frames compared':<{w}} {n_frames:>12}")
print(f"  {'Atom selection':<{w}} {args.selection:>12}")
print(f"  {'RMSD mean (Å)':<{w}} {summary['rmsd_mean_A']:>12.3f}")
print(f"  {'RMSD std  (Å)':<{w}} {summary['rmsd_std_A']:>12.3f}")
print(f"  {'RMSD median (Å)':<{w}} {summary['rmsd_median_A']:>12.3f}")
print(f"  {'RMSD max (Å)':<{w}} {summary['rmsd_max_A']:>12.3f}")
print(f"  {'RMSF orig mean (Å)':<{w}} {summary['rmsf_orig_mean_A']:>12.3f}")
print(f"  {'RMSF recon mean (Å)':<{w}} {summary['rmsf_recon_mean_A']:>12.3f}")
print(f"  {'|ΔRMSF| mean (Å)':<{w}} {summary['rmsf_delta_mean_A']:>12.3f}")
print(f"  {'Residue RMSD mean (Å)':<{w}} {summary['residue_rmsd_mean_A']:>12.3f}")
print(f"  {'Worst 5 residues':<{w}} {str(summary['residue_rmsd_worst5']):>12}")
if "ae_feature_mae_mean" in summary:
    print(f"  {'AE feature MAE mean':<{w}} {summary['ae_feature_mae_mean']:>12.5f}")
    print(f"  {'AE feature MAE std':<{w}} {summary['ae_feature_mae_std']:>12.5f}")
print("─" * (w + 16))
print(f"\n  Summary saved to: {summary_path}\n")
