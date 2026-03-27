import os
import sys
import json
import pickle
import argparse
import tempfile
import numpy as np
import mdtraj as md
import torch

# Allow running as `python decompress.py` directly from inside aezip/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aezip.prep.featurize import (
    build_topology_dict, build_reslib_dict, get_dihedral_indices_and_names,
    convert_full_to_sliced_indices, build_dihedral_atom_indices,
    build_dih_traj, get_histidine_protonation_states, get_protonation_states,
)
from aezip.backmapping import backmap_trajectory
from aezip.trajectory_backmapping import trajectory_reconstruction
from aezip.model.model import *
from aezip.utils.modelling import *          # provides reconstruct_trajectory
from aezip.utils.residue_lib_manager import ResidueLib

_HERE = os.path.dirname(__file__)

res_library = ResidueLib(os.path.join(_HERE, 'dat', 'all_residues.in'))
datlib_dict = json.load(open(os.path.join(_HERE, 'dat', 'data_lib.json')))['data_library']['residue_data']
dihedral_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_dih.json')))
cartesian_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_cart.json')))

# Make reslib dictionary
reslib_dict = build_reslib_dict(res_library)

parser = argparse.ArgumentParser(description="Decompress a trajectory using an autoencoder.")
parser.add_argument("-s", "--top-file", required=True, help="Path to the topology file (.pdb)")
parser.add_argument("-i", "--input-file", default="./compressed_traj.pth", help="Path to the compressed trajectory (default: ./compressed_traj.pth)")
parser.add_argument("-o", "--output-file", default="./recon_traj.xtc", help="Path to save the reconstructed trajectory (default: ./recon_traj.xtc)")
parser.add_argument("-op", "--output-pdb", default="./recon_traj.pdb", help="Path to save the reference PDB frame (default: ./recon_traj.pdb)")
parser.add_argument("-c", "--config-file", default="../config/config.json", help="Path to the config file (default: ./config.json)")
args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)
    compression_type = config["compression_type"]

topology = md.load(args.top_file)
topology_dict = build_topology_dict(topology)

dihedral_indices_and_names = get_dihedral_indices_and_names(topology, dihedral_definitions)
dihedral_atom_indices = build_dihedral_atom_indices(topology, dihedral_definitions)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
saved_data = torch.load(args.input_file, weights_only=False)

if compression_type == "dihedral":
    sliced_topology = topology.atom_slice(topology.topology.select("name N or name CA or name C or resname PRO"))
elif compression_type == "cartesian":
    compressed_top = saved_data["compressed_topology"]
    sliced_topology = md.Trajectory(np.zeros((1, compressed_top.n_atoms, 3)), topology=compressed_top)
else:
    raise ValueError(f"Invalid compression type: {compression_type}")

sliced_indices = convert_full_to_sliced_indices(topology, sliced_topology)
residues_list = list(topology.topology.residues)
hist_dict = get_histidine_protonation_states(topology)

# Extract components
decoder_state_dict = saved_data["decoder_state_dict"]
latent = saved_data["compressed_traj"]
scaler = pickle.loads(saved_data["scaler"])
hyperparams = saved_data["hyperparams"]

print('latent', latent.shape)
decoder = Decoder(latent_dim=hyperparams["latent_dim"], 
                  nlayers=hyperparams["layers"], 
                  output_dim=hyperparams["input_dim"],
                  delta=None, 
                  dropout=hyperparams["dropout"], 
                  negative_slope=hyperparams["negative_slope"], 
                  batch_norm=hyperparams["batch_norm"]).to(device)

latent = torch.utils.data.DataLoader(dataset = torch.FloatTensor(latent), 
                                     batch_size=hyperparams["batch_size"], 
                                     drop_last=False, 
                                     shuffle = False)
data = decompress(decoder, latent, device)
print(data.shape)

data = scaler.inverse_transform(data)

if compression_type == "dihedral":
    backbone_xyz = data[:, :sliced_topology.xyz.shape[1]*3]
    dih_traj = data[:, (sliced_topology.xyz.shape[1]*3):]
    sliced_traj = md.Trajectory(backbone_xyz.reshape(len(data), -1, 3), topology=sliced_topology.topology)

    dihedral_mapping = build_dihedral_atom_indices(topology, dihedral_definitions)
    dihedral_traj = build_dih_traj(dih_traj, dihedral_mapping)
    recon_traj = reconstruct_trajectory(topology, sliced_traj, sliced_indices, dihedral_traj, dihedral_definitions,
                               topology_dict, reslib_dict, datlib_dict, hist_dict)
    print(recon_traj.shape)
    recon_traj = md.Trajectory(recon_traj / 10, topology=topology.topology)

elif compression_type == "cartesian":
    partial_traj = md.Trajectory(data.reshape(len(data), -1, 3), topology=sliced_topology.topology)
    reference_frame = saved_data["topology"]
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

recon_traj.save_xtc(args.output_file)
