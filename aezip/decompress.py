import os
import sys
import json
import argparse
import tempfile
import numpy as np
import mdtraj as md
import torch

# Allow running as `python decompress.py` directly from inside aezip/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aezip.utils.logger import RunLogger

from aezip.prep.featurize import (
    build_topology_dict, build_reslib_dict,
    convert_full_to_sliced_indices, build_dihedral_atom_indices,
    build_dih_traj, get_histidine_protonation_states, get_protonation_states,
)
from aezip.utils.backmapping import trajectory_reconstruction
from aezip.utils.modelling import *
from aezip.utils.residue_lib_manager import ResidueLib
from biobb_pytorch.mdae.decode_model import EvaluateDecoder

_HERE = os.path.dirname(__file__)

res_library  = ResidueLib(os.path.join(_HERE, 'dat', 'all_residues.in'))
datlib_dict  = json.load(open(os.path.join(_HERE, 'dat', 'data_lib.json')))['data_library']['residue_data']
dihedral_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_dih.json')))
reslib_dict  = build_reslib_dict(res_library)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Decompress a trajectory using an autoencoder.")
parser.add_argument("-i",  "--input-file",  default="./compressed_traj.pth",
                    help="Compressed trajectory (default: ./compressed_traj.pth)")
parser.add_argument("-o",  "--output-file", default="./recon_traj.xtc",
                    help="Output trajectory (default: ./recon_traj.xtc)")
parser.add_argument("-op", "--output-pdb",  default="./recon_traj.pdb",
                    help="Output reference PDB (default: ./recon_traj.pdb)")
parser.add_argument("--log-file", default=None, metavar="FILE",
                    help="JSON log file (default: <output>.log.json)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load compressed data  (topology and compression_type are stored inside)
# ---------------------------------------------------------------------------
log_path = args.log_file or (os.path.splitext(args.output_file)[0] + ".log.json")
logger   = RunLogger(log_path, run_type="decompress", args=vars(args))

saved_data       = torch.load(args.input_file, weights_only=False)
biobb_model      = saved_data["model"]
z                = saved_data["compressed_traj"]
hp               = saved_data["hyperparams"]
stats            = saved_data["stats"]
use_ae           = saved_data.get("use_ae", biobb_model is not None)
compression_type = saved_data["compression_type"]

topology      = saved_data["topology"]   # single-frame md.Trajectory saved by compress.py
topology_dict = build_topology_dict(topology)
hist_dict     = get_histidine_protonation_states(topology)

print(f"Compressed traj shape: {z.shape}  |  use_ae: {use_ae}")

# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
with logger.step("Decode"):
    if use_ae:
        ed = EvaluateDecoder(
            input_model=biobb_model,
            input_latent=z,
            properties={"Dataset": {"batch_size": hp["batch_size"]}},
        )
        data = ed.run_decoding()["xhat"]
        if hasattr(data, "numpy"):
            data = data.numpy()
    else:
        data = z
    print(f"  Decoded shape: {data.shape}")

# ---------------------------------------------------------------------------
# Reconstruct all-atom trajectory
# ---------------------------------------------------------------------------
with logger.step(f"Reconstruct ({compression_type})"):
    if compression_type == "cartesian":
        compressed_top = saved_data["compressed_topology"]
        partial_traj   = md.Trajectory(
            data.reshape(len(data), -1, 3),
            topology=compressed_top,
        )
        reference_frame    = saved_data["topology"]
        protonation_states = get_protonation_states(reference_frame)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            reference_pdb = tmp.name
        reference_frame.save_pdb(reference_pdb)
        try:
            recon_traj = trajectory_reconstruction(
                traj=partial_traj,
                output_pdb_file=args.output_pdb,
                output_xtc_file=args.output_file,
                reference_pdb=reference_pdb,
                protonation_states=protonation_states,
            )
        finally:
            os.remove(reference_pdb)

    elif compression_type == "dihedral":
        compressed_top  = saved_data["compressed_topology"]
        sliced_topology = md.Trajectory(
            np.zeros((1, compressed_top.n_atoms, 3)),
            topology=compressed_top,
        )
        sliced_indices = convert_full_to_sliced_indices(topology, sliced_topology)
        cart_idx = stats.get("cartesian_indices")
        dih_idx  = stats.get("dihedral_indices")
        if cart_idx is not None and dih_idx is not None:
            backbone_xyz = data[:, cart_idx]
            dih_traj_raw = data[:, dih_idx]
        else:
            n_backbone   = sliced_topology.n_atoms
            backbone_xyz = data[:, : n_backbone * 3]
            dih_traj_raw = data[:, n_backbone * 3 :]
        sliced_traj      = md.Trajectory(backbone_xyz.reshape(len(data), -1, 3),
                                         topology=sliced_topology.topology)
        dihedral_mapping = build_dihedral_atom_indices(topology, dihedral_definitions)
        dihedral_traj    = build_dih_traj(dih_traj_raw, dihedral_mapping)
        recon_traj       = reconstruct_trajectory(
            topology, sliced_traj, sliced_indices,
            dihedral_traj, dihedral_definitions,
            topology_dict, reslib_dict, datlib_dict, hist_dict,
        )
        recon_traj = md.Trajectory(recon_traj / 10, topology=topology.topology)

    elif compression_type == "cg2all":
        from aezip.model.cg2all import load_model as _cg2all_load, backmap_to_aa
        cg_model_type                 = saved_data.get("cg_model_type", "CalphaBasedModel")
        cg_batch_size                 = saved_data.get("cg_batch_size", 1)
        cg_device                     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gg_model, gg_config, cg_class = _cg2all_load(cg_model_type, device=cg_device)
        reference_frame               = saved_data["topology"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
            reference_pdb = tmp.name
        reference_frame.save_pdb(reference_pdb)
        try:
            recon_traj = backmap_to_aa(
                data, reference_pdb,
                gg_model, gg_config, cg_class,
                batch_size=cg_batch_size, device=cg_device,
            )
        finally:
            os.remove(reference_pdb)

    else:
        raise ValueError(f"Unknown compression_type: {compression_type!r}")

with logger.step("Save"):
    recon_traj.save_xtc(args.output_file)
    recon_traj[0].save_pdb(args.output_pdb)

logger.log_file(args.output_file, label="reconstructed_xtc")
logger.log_file(args.output_pdb,  label="reconstructed_pdb")
logger.save()