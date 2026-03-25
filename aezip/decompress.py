import os
import json
import pickle
import numpy as np
import mdtraj as md
import torch
from sklearn.preprocessing import MinMaxScaler

from .prep.featurize import (
    build_topology_dict, build_reslib_dict, get_dihedral_indices_and_names,
    convert_full_to_sliced_indices, calculate_dih_traj, build_dihedral_atom_indices,
    build_dih_traj, get_histidine_protonation_states,
)
from .model.model import *
from .utils.modelling import *          # provides reconstruct_trajectory
from .utils.residue_lib_manager import ResidueLib

_HERE = os.path.dirname(__file__)

res_library = ResidueLib(os.path.join(_HERE, 'dat', 'all_residues.in'))
datlib_dict = json.load(open(os.path.join(_HERE, 'dat', 'data_lib.json')))['data_library']['residue_data']

topology = md.load("../Data/WT_apo_ChainsA_first.pdb")
topology_dict = build_topology_dict(topology)

dihedral_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_dih.json')))
dihedral_indices_and_names = get_dihedral_indices_and_names(topology, dihedral_definitions)
dihedral_atom_indices = build_dihedral_atom_indices(topology, dihedral_definitions)

sliced_topology = topology.atom_slice(topology.topology.select("name N or name CA or name C or resname PRO"))
sliced_indices = convert_full_to_sliced_indices(topology, sliced_topology)

# Make reslib dictionary
reslib_dict = build_reslib_dict(res_library)
residues_list = list(topology.topology.residues)
hist_dict = get_histidine_protonation_states(topology)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_folder = "./"
saved_data = torch.load(os.path.join(save_folder, "compressed_traj.pth"))

# Extract components
decoder_state_dict = saved_data["decoder_state_dict"]
latent = saved_data["compressed_traj"]
scaler = pickle.loads(saved_data["scaler"])
hyperparams = saved_data["hyperparams"]

print('latent', latent.shape)
decoder = Decoder(latent_dim=512, 
                  nlayers=hyperparams["layers"], 
                  output_dim=2108,  #hyperparams["input_dim"], 
                  delta=None, 
                  dropout=hyperparams["dropout"], 
                  negative_slope=hyperparams["negative_slope"], 
                  batch_norm=hyperparams["batch_norm"]).to(device)

latent = torch.utils.data.DataLoader(dataset = torch.FloatTensor(latent), 
                                     batch_size=hyperparams["batch_size"], 
                                     drop_last=False, 
                                     shuffle = False)
data = decompress(decoder, latent, device)
print(data)
print(data.shape)
np.save("decompressed_angles.npy", data)
#scaler = MinMaxScaler()

data = scaler.inverse_transform(data)

backbone_xyz = data[:, :sliced_topology.xyz.shape[1]*3]
dih_traj = data[:, (sliced_topology.xyz.shape[1]*3):]
print(data[:, 5511:], data[:, 5511:].shape)
sliced_traj = md.Trajectory(backbone_xyz.reshape(len(data), -1, 3), topology=sliced_topology.topology)
#dihedral_features_idx = np.arange(sliced_traj.xyz.shape[1]*3,
#                                  hyperparams["input_dim"])

# Reconstruct Trajectory
#dihedral_traj = compute_dihedral_trajectory(
#    traj=sliced_traj,
#    dihedral_indices_and_names=dihedral_indices_and_names,
#    dihedral_atom_indices=dihedral_atom_indices,
#    data=data,
#    dihedral_features_idx=dihedral_features_idx,
#    use_precomputed=True
#)

dihedral_mapping = build_dihedral_atom_indices(topology, dihedral_definitions)
dihedral_traj = build_dih_traj(dih_traj, dihedral_mapping)
print(dihedral_traj[0]) 
recon_traj = reconstruct_trajectory(topology, sliced_traj, sliced_indices, dihedral_traj, dihedral_definitions, 
                           topology_dict, reslib_dict, datlib_dict, hist_dict)
print(recon_traj.shape)
recon_traj = md.Trajectory(recon_traj / 10, topology=topology.topology)
recon_traj.save_xtc(os.path.join(save_folder, "recon_traj.xtc"))
