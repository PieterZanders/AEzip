import os
import json
import pickle
import argparse
from utils import build_reslib_dict, get_dihedral_indices_and_names, convert_full_to_sliced_indices, calculate_dih_traj
from model import *
from residue_lib_manager import ResidueLib
from sklearn.preprocessing import MinMaxScaler

res_library = ResidueLib('all_residues.in')
dihedral_definitions = json.load(open('aa_dih.json'))

# Make reslib dictionary
reslib_dict = build_reslib_dict(res_library)

# Build topology information dictionary
traj = md.load('../Data/WT_apo_ChainsA.xtc', top='../Data/WT_apo_ChainsA_first.pdb')
topology = traj.topology

sliced_traj = traj.atom_slice(traj.topology.select("resid 1"))  #"name N or name CA or name C or resname PRO"))

dihedral_indices_and_names = get_dihedral_indices_and_names(traj, dihedral_definitions)
print("dihedral_indices_and_names")

sliced_indices = convert_full_to_sliced_indices(traj, sliced_traj)
print("sliced_indices")

coord_feat = sliced_traj.xyz.reshape(len(sliced_traj.xyz), -1)
dih_traj = calculate_dih_traj(traj, dihedral_indices_and_names)
#dih_traj = np.concatenate((np.sin(dih_traj, dtype=float), np.cos(dih_traj, dtype=float)), axis=1)
print("coord_feat: ", coord_feat.shape)

partition_idx = sliced_traj.xyz.shape[1]*3
print(partition_idx)

## Autoencoder Input
scaler = MinMaxScaler(feature_range=(-1, 1))
coord_feat = scaler.fit_transform(coord_feat)
print(np.min(coord_feat), np.max(coord_feat))
#dih_traj = scaler.fit_transform(dih_traj)
#scaler = MinMaxScaler(feature_range=(0, 1))
#feat = scaler.fit_transform(dih_traj) # np.concatenate((coord_feat, dih_traj), axis=1)
print(dih_traj.shape)
feat = dih_traj #(dih_traj+ 180) / 360
np.save("WT_apo_ChainsA_SC_angles.npy", dih_traj)
feat = np.stack([np.sin(np.deg2rad(feat)), np.cos(np.deg2rad(feat))], axis=-1).reshape(len(feat), -1)
print(feat)
print(feat.shape, np.min(feat), np.max(feat))

# Autoencoder 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hyperparams = {
    "input_dim": feat.shape[1],    
    "layers": 5, 
    "latent_dim": 512,
    "learning_rate": 0.0001,
    "weight_decay": 1e-9,
    "batch_size": 16,
    "num_epochs": 100,
    "dropout": 0.0,
    "negative_slope": 0.0,
    "batch_norm": False,
    "gamma": 0.5,
    "step_size": 25,
    "optimizer": "adam",
    "loss_function": "MSE",
}


save_folder = "./"

train_dataloader = torch.utils.data.DataLoader(dataset = torch.FloatTensor(feat), 
                                               batch_size=hyperparams["batch_size"], 
                                               drop_last=False, 
                                               shuffle = False)

model = SAutoEncoder(input_dim=hyperparams["input_dim"], 
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
loss_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#loss_function = PeriodicLoss()  # CombinedLoss(partition_idx, cartesian_weight=1.0, angular_weight=1.0)
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
    "hyperparams": hyperparams
}, os.path.join(save_folder, "compressed_traj.pth"))
