import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import tempfile
import numpy as np
import mdtraj as md
from aezip.config.config import cartesian_definitions, dihedral_definitions
from aezip.prep.featurize import (
    build_topology_dict, build_reslib_dict, get_dihedral_indices_and_names,
    calculate_dih_traj, convert_full_to_sliced_indices,
    build_dihedral_atom_indices, build_dih_traj,
    get_histidine_protonation_states, get_protonation_states,
)
from aezip.utils.modelling import reconstruct_trajectory
from aezip.utils.residue_lib_manager import ResidueLib
from aezip.utils.backmapping import trajectory_reconstruction

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Reconstruct all-atom trajectories without an autoencoder "
                "(cartesian and dihedral paths)."
)
parser.add_argument("-f", "--traj-file", required=True,
                    help="Input trajectory (.xtc / .dcd / ...)")
parser.add_argument("-s", "--top-file",  required=True,
                    help="Topology file (.pdb)")
parser.add_argument("--stride", type=int, default=1,
                    help="Frame stride when loading trajectory (default: 1)")
parser.add_argument("-ct", "--compression-type",
                    choices=["cartesian", "dihedral"], default="cartesian",
                    help="Reconstruction method: 'cartesian' (cg2all backmapping) "
                         "or 'dihedral' (geometry rebuild). Default: cartesian")
parser.add_argument("-o", "--output-xtc",
                    default="./recon_traj.xtc",
                    help="Output trajectory (default: ./recon_traj.xtc)")
parser.add_argument("-op", "--output-pdb",
                    default="./recon_traj.pdb",
                    help="Output reference PDB (default: ./recon_traj.pdb)")
args = parser.parse_args()

# ── load data ────────────────────────────────────────────────────────────────
res_library = ResidueLib(os.path.join(PROJECT_ROOT, "aezip/dat/all_residues.in"))
datlib_dict = json.load(open(os.path.join(PROJECT_ROOT, "aezip/dat/data_lib.json")))["data_library"]["residue_data"]
reslib_dict = build_reslib_dict(res_library)

traj          = md.load(args.traj_file, top=args.top_file, stride=args.stride)
topology      = md.load(args.top_file)
topology_dict = build_topology_dict(topology)
hist_dict     = get_histidine_protonation_states(topology)

print(f"Compression type: {args.compression_type}")

if args.compression_type == "cartesian":
    print("=" * 60)
    cart_sel = " or ".join(
        f"(resname {res} and ({' or '.join(f'name {a}' for a in atoms)}))"
        for res, atoms in cartesian_definitions.items()
    )
    partial_traj = traj.atom_slice(traj.topology.select(cart_sel))
    print(f"Partial traj shape  : {partial_traj.xyz.shape}")

    protonation_states = get_protonation_states(traj)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
        reference_pdb = tmp.name
    traj[0].save_pdb(reference_pdb)

    try:
        recon = trajectory_reconstruction(
            traj=partial_traj,
            output_pdb_file=args.output_pdb,
            output_xtc_file=args.output_xtc,
            reference_pdb=reference_pdb,
            protonation_states=protonation_states,
        )
    finally:
        os.remove(reference_pdb)

    print(f"Reconstructed shape : {recon.xyz.shape}")

elif args.compression_type == "dihedral":
    print("=" * 60)
    backbone_sel     = "name N or name CA or name C or resname PRO"
    backbone_indices = traj.topology.select(backbone_sel)
    sliced_traj      = traj.atom_slice(backbone_indices)
    backbone_top     = sliced_traj.topology

    coord_feat = sliced_traj.xyz.reshape(len(sliced_traj), -1)

    dihedral_indices_and_names = get_dihedral_indices_and_names(traj, dihedral_definitions)
    dih_feat = calculate_dih_traj(traj, dihedral_indices_and_names)

    raw_feat = np.concatenate((coord_feat, dih_feat), axis=1)
    print(f"Feature array shape : {raw_feat.shape}")
    print(f"  backbone coords   : {coord_feat.shape[1]}")
    print(f"  custom dihedrals  : {dih_feat.shape[1]}")

    n_backbone   = sliced_traj.n_atoms
    backbone_xyz = raw_feat[:, : n_backbone * 3]
    dih_raw      = raw_feat[:, n_backbone * 3 :]

    sliced_top_traj = md.Trajectory(
        np.zeros((1, backbone_top.n_atoms, 3)),
        topology=backbone_top,
    )
    sliced_indices    = convert_full_to_sliced_indices(topology, sliced_top_traj)
    sliced_traj_recon = md.Trajectory(
        backbone_xyz.reshape(len(raw_feat), -1, 3),
        topology=backbone_top,
    )

    dihedral_mapping = build_dihedral_atom_indices(topology, dihedral_definitions)
    dihedral_traj    = build_dih_traj(dih_raw, dihedral_mapping)

    recon_xyz = reconstruct_trajectory(
        topology, sliced_traj_recon, sliced_indices,
        dihedral_traj, dihedral_definitions,
        topology_dict, reslib_dict, datlib_dict, hist_dict,
    )
    recon = md.Trajectory(recon_xyz / 10, topology=topology.topology)
    print(f"Reconstructed shape : {recon.xyz.shape}")
    recon.save_xtc(args.output_xtc)
    recon[0].save_pdb(args.output_pdb)
