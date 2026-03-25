#!/usr/bin/env python
# Colab-ready MD trajectory compression and reconstruction using cg2all and a PyTorch autoencoder

import os
import json
import numpy as np
import torch
from torch import nn, optim
import mdtraj as md
import dgl
import cg2all.lib.libcg
import cg2all.lib.libmodel
from cg2all.lib.libconfig import MODEL_HOME
from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch

import warnings

warnings.filterwarnings("ignore")

def main():
    print("Initializing cg2all + PyTorch AE pipeline...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Hardcoded settings ===
    pdb_file = '../Data/WT_apo_ChainsA_first.pdb'
    traj_file = '../Data/WT_apo_ChainsA.xtc'
    output_xtc = 'reconstructed.xtc'
    cg_model_type = 'CalphaBasedModel'
    checkpoint = None
    batch_size = 1
    latent_dim = 64
    epochs = 50
    lr = 1e-3
    time_json = None
    # ===========================

    # COMPRESS
    print(f"Settings: PDB={pdb_file}, Traj={traj_file}, CG Model={cg_model_type}")

    # Load cg2all checkpoint
    print("Loading cg2all checkpoint...")
    if checkpoint is None:
        ckpt_path = MODEL_HOME / f"{cg_model_type}.ckpt"
        if not ckpt_path.exists():
            print("Downloading checkpoint...")
            cg2all.lib.libmodel.download_ckpt_file(cg_model_type, ckpt_path)
    else:
        ckpt_path = checkpoint
    print(f"Checkpoint path: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt['hyper_parameters']
    print(f"Loaded hyperparameters: {config}")

    # Configure and instantiate cg2all model
    print("Configuring cg2all model...")
    cg_class = getattr(cg2all.lib.libcg, config['cg_model'])
    gg_config = cg2all.lib.libmodel.set_model_config(config, cg_class, flattened=False)
    gg_model = cg2all.lib.libmodel.Model(gg_config, cg_class, compute_loss=False)
    state = {k.split('.',1)[1]: v for k, v in ckpt['state_dict'].items()}
    gg_model.load_state_dict(state)
    gg_model.to(device).eval()
    gg_model.set_constant_tensors(device)
    print("cg2all model ready.")

    # Prepare CG data loader
    print("Preparing CG data loader...")
    cb_val = getattr(gg_config.globals, 'chain_break_cutoff', 10.0)
    pdata = PredictionData(
        pdb_file,
        cg_class,
        dcd_fn=traj_file,
        radius=gg_config.globals.radius,
        chain_break_cutoff=0.1 * cb_val,
        batch_size=batch_size
    )
    loader = dgl.dataloading.GraphDataLoader(pdata, batch_size=1, shuffle=False)

    # Extract CG bead coordinates
    print("Extracting coarse-grained coordinates...")
    print(gg_model)
    coords_list = []
    for i, batch in enumerate(loader, 1):
        batch = batch.to(device)
        with torch.no_grad():
            out = gg_model.forward(batch)[0]
        R = out['R'].cpu().numpy()
        mask = batch.ndata['output_atom_mask'].cpu().numpy()
        coords_list.append(R[mask > 0].flatten())
        if i % 10 == 0:
            print(f"  Processed {i} frames...")
    X = np.vstack(coords_list)
    print(f"Extracted CG data: {X.shape[0]} frames, {X.shape[1]//3} beads.")

    from mdtraj.core.element import Element
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue('CG', chain)
    for i in range(coords_list.shape[1]):
       atom_name = f'B{i:03d}'             # e.g. B000, B001, …
       top.add_atom(atom_name, Element.getBySymbol('C'), residue)
    traj_cg = md.Trajectory(coords_list, topology=top)
    traj_cg.save_xtc("cg_traj.xtc")
    traj_cg[0].save_pdb("cg_traj.pdb")
    print("CG .xtc generated")

    # Normalize (NumPy)
    print("Normalizing data...")
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)

    # Define PyTorch autoencoder
    print("Building autoencoder...")
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=64):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, input_dim)
            )
        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    input_dim = X_tensor.shape[1]
    ae = Autoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()
    print(f"Autoencoder with input dim {input_dim}, latent dim {latent_dim} ready.")

    # Train autoencoder
    print(f"Training autoencoder for {epochs} epochs...")
    n_train = int(0.8 * len(X_tensor))
    train_data, val_data = X_tensor[:n_train], X_tensor[n_train:]
    for epoch in range(1, epochs + 1):
        ae.train()
        optimizer.zero_grad()
        recon, _ = ae(train_data)
        loss = criterion(recon, train_data)
        loss.backward()
        optimizer.step()

        ae.eval()
        with torch.no_grad():
            recon_val, _ = ae(val_data)
            val_loss = criterion(recon_val, val_data)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # DECOMPRESS

    # Reconstruct entire set
    print("Reconstructing full trajectory from latent space...")
    ae.eval()
    with torch.no_grad():
        recon_all, latent = ae(X_tensor)
    recon_all = recon_all.cpu().numpy() * X_std + X_mean
    frames_cg = recon_all.reshape(-1, -1, 3)

    # Backmap and save
    print("Backmapping to all-atom coordinates and saving trajectory...")
    all_xyz = []
    for coords in frames_cg:
        traj_cg, _ = create_trajectory_from_batch(batch, torch.tensor(coords, device=device))
        traj_aa = cg2all.lib.libcg.backmap(traj_cg)
        all_xyz.append(traj_aa)
    all_xyz = np.stack(all_xyz)

    topo = md.load(traj_file, top=pdb_file).topology
    recon_traj = md.Trajectory(all_xyz, topo)
    recon_traj.save_xtc(output_xtc)
    print(f"Saved reconstructed trajectory: {output_xtc}")

    # Save artifacts
    print("Saving decoder weights, latent vectors, and scaler...")
    os.makedirs('models', exist_ok=True)
    torch.save(ae.dec.state_dict(), 'models/decoder.pth')
    np.save('models/latent.npy', latent.cpu().numpy())
    with open('models/scaler.json', 'w') as f:
        json.dump({'mean': X_mean.tolist(), 'std': X_std.tolist()}, f)
    print("Artifacts saved in ./models/")

    if time_json:
        print("Writing timing info...")
        with open(time_json, 'w') as f:
            json.dump({'frames': len(X)}, f, indent=2)
        print(f"Timing written to {time_json}")

if __name__ == '__main__':
    main()
