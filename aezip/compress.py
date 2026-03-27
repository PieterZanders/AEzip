import os
import sys
import json
import pickle
import argparse
import numpy as np
import mdtraj as md
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

# Allow running as `python compress.py` directly from inside aezip/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aezip.prep.featurize import (
    build_reslib_dict,
    get_dihedral_indices_and_names,
    calculate_dih_traj,
)
from aezip.model.model import *
from aezip.utils.residue_lib_manager import ResidueLib

_HERE = os.path.dirname(__file__)

res_library = ResidueLib(os.path.join(_HERE, 'dat', 'all_residues.in'))
dihedral_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_dih.json')))
cartesian_definitions = json.load(open(os.path.join(_HERE, 'config', 'aa_cart.json')))

# Make reslib dictionary
reslib_dict = build_reslib_dict(res_library)

# Build topology information dictionary
parser = argparse.ArgumentParser(description="Compress a trajectory using an autoencoder.")
parser.add_argument("-f", "--traj-file", help="Path to the input trajectory (.xtc)")
parser.add_argument("-s", "--top-file", help="Path to the topology file (.pdb)")
parser.add_argument("-o", "--output-file", default="./compressed_traj.pth", help="Path to save the compressed trajectory (default: ./compressed_traj.pth)")
parser.add_argument("-c", "--config-file", default="./config.json", help="Path to the config file (default: ./config.json)")
args = parser.parse_args()

# load config file
with open(args.config_file, "r") as f:
    config = json.load(f)
    hyperparams = config["hyperparams"]
    compression_type = config["compression_type"]

traj = md.load(args.traj_file, top=args.top_file, stride=10)
topology = traj.topology

if compression_type == "dihedral":
    sliced_traj = traj.atom_slice(traj.topology.select("name N or name CA or name C or resname PRO"))
    coord_feat = sliced_traj.xyz.reshape(len(sliced_traj.xyz), -1)
    dihedral_indices_and_names = get_dihedral_indices_and_names(traj, dihedral_definitions)
    dih_traj = calculate_dih_traj(traj, dihedral_indices_and_names)
    raw_feat = np.concatenate((coord_feat, dih_traj), axis=1)

elif compression_type == "cartesian":
    cart_sel = " or ".join(
        f"(resname {res} and ({' or '.join(f'name {a}' for a in atoms)}))"
        for res, atoms in cartesian_definitions.items()
    )
    sliced_traj = traj.atom_slice(traj.topology.select(cart_sel))
    raw_feat = sliced_traj.xyz.reshape(len(sliced_traj.xyz), -1)
else:
    raise ValueError(f"Invalid compression type: {compression_type}")

scaler = MinMaxScaler(feature_range=(0, 1))
feat = scaler.fit_transform(raw_feat)

# Autoencoder 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparams["input_dim"] = feat.shape[1]

train_dataloader = DataLoader(dataset = torch.FloatTensor(feat), 
                                   batch_size = hyperparams["batch_size"], 
                                   drop_last = False, 
                                   shuffle = False)

model = AutoEncoder(input_dim=hyperparams["input_dim"], 
                    nlayers=hyperparams["layers"], 
                    latent_dim=hyperparams["latent_dim"], 
                    dropout=hyperparams["dropout"], 
                    negative_slope=hyperparams["negative_slope"], 
                    batch_norm=hyperparams["batch_norm"]).to(device)

optimizer =  torch.optim.Adam(model.parameters(), 
                              lr=hyperparams["learning_rate"], 
                              weight_decay=hyperparams["weight_decay"], 
                              foreach=False)
loss_function = torch.nn.MSELoss()

print(model)

train_loss, latent = compress(model, 
                              train_dataloader, 
                              optimizer, 
                              loss_function, 
                              epochs=hyperparams["num_epochs"], 
                              step_size=hyperparams["step_size"], 
                              gamma=hyperparams["gamma"], 
                              device=device)

torch.save({
    "decoder_state_dict": model.decoder.state_dict(),
    "compressed_traj": latent,
    "scaler": pickle.dumps(scaler),
    "hyperparams": hyperparams,
    "topology": traj[0],
    "compressed_topology": sliced_traj.topology,
}, args.output_file)
