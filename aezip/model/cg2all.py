"""
cg2all interface for AEzip.

Three public functions:
    load_model(cg_model_type, checkpoint, device)
        → gg_model, gg_config, cg_class

    extract_cg_coords(pdb_file, traj_file, gg_model, gg_config, cg_class, ...)
        → X  (n_frames, n_beads * 3)  float32 numpy array

    backmap_to_aa(frames_cg, pdb_file, gg_model, gg_config, cg_class, ...)
        → all_xyz  (n_frames, n_atoms, 3)  numpy array
"""

import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")


def load_model(
    cg_model_type: str = "CalphaBasedModel",
    checkpoint=None,
    device: torch.device = None,
):
    """Load a cg2all model and return it ready for inference.

    Parameters
    ----------
    cg_model_type : str
        Name of the cg2all CG model (e.g. 'CalphaBasedModel').
    checkpoint : path-like, optional
        Path to a local .ckpt file. If None the model is looked up in
        MODEL_HOME and downloaded if not present.
    device : torch.device, optional
        Target device. Defaults to CUDA when available, else CPU.

    Returns
    -------
    gg_model, gg_config, cg_class
    """
    import dgl  # noqa: F401 – triggers backend selection before other cg2all imports
    import cg2all.lib.libcg
    import cg2all.lib.libmodel
    from cg2all.lib.libconfig import MODEL_HOME

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint is None:
        ckpt_path = MODEL_HOME / f"{cg_model_type}.ckpt"
        if not ckpt_path.exists():
            print(f"Downloading cg2all checkpoint: {cg_model_type}...")
            cg2all.lib.libmodel.download_ckpt_file(cg_model_type, ckpt_path)
    else:
        ckpt_path = checkpoint

    print(f"Loading cg2all checkpoint from {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt["hyper_parameters"]

    cg_class  = getattr(cg2all.lib.libcg, config["cg_model"])
    gg_config = cg2all.lib.libmodel.set_model_config(config, cg_class, flattened=False)
    gg_model  = cg2all.lib.libmodel.Model(gg_config, cg_class, compute_loss=False)

    state = {k.split(".", 1)[1]: v for k, v in ckpt["state_dict"].items()}
    gg_model.load_state_dict(state)
    gg_model.to(device).eval()
    gg_model.set_constant_tensors(device)

    print("cg2all model ready.")
    return gg_model, gg_config, cg_class


def extract_cg_coords(
    pdb_file: str,
    traj_file: str,
    gg_model,
    gg_config,
    cg_class,
    batch_size: int = 1,
    device: torch.device = None,
) -> np.ndarray:
    """Run the cg2all forward pass and return flattened CG bead coordinates.

    Parameters
    ----------
    pdb_file : str
        All-atom reference PDB (topology).
    traj_file : str
        All-atom trajectory (.xtc / .dcd).
    gg_model, gg_config, cg_class
        As returned by :func:`load_model`.
    batch_size : int
        DGL DataLoader batch size (default 1).
    device : torch.device, optional

    Returns
    -------
    X : np.ndarray, shape (n_frames, n_beads * 3), dtype float32
    """
    import dgl
    from cg2all.lib.libdata import PredictionData

    if device is None:
        device = next(gg_model.parameters()).device

    cb_val = getattr(gg_config.globals, "chain_break_cutoff", 10.0)
    pdata  = PredictionData(
        pdb_file, cg_class,
        dcd_fn=traj_file,
        radius=gg_config.globals.radius,
        chain_break_cutoff=0.1 * cb_val,
        batch_size=batch_size,
    )
    loader = dgl.dataloading.GraphDataLoader(pdata, batch_size=1, shuffle=False)

    coords_list = []
    for i, batch in enumerate(loader, 1):
        batch = batch.to(device)
        with torch.no_grad():
            out = gg_model.forward(batch)[0]
        R    = out["R"].cpu().numpy()
        mask = batch.ndata["output_atom_mask"].cpu().numpy()
        coords_list.append(R[mask > 0].flatten())
        if i % 100 == 0:
            print(f"  Extracted {i} frames...")

    X = np.vstack(coords_list).astype(np.float32)
    print(f"CG coordinates: {X.shape[0]} frames, {X.shape[1] // 3} beads")
    return X


def backmap_to_aa(
    frames_cg: np.ndarray,
    pdb_file: str,
    gg_model,
    gg_config,
    cg_class,
    batch_size: int = 1,
    device: torch.device = None,
) -> np.ndarray:
    """Backmap CG bead coordinates to all-atom using cg2all.

    A single-frame topology batch is derived from *pdb_file* and reused for
    every frame (valid because all frames share the same topology).

    Parameters
    ----------
    frames_cg : np.ndarray, shape (n_frames, n_beads * 3)
        CG coordinates in the same units as cg2all output.
    pdb_file : str
        All-atom reference PDB used to define the topology batch.
    gg_model, gg_config, cg_class
        As returned by :func:`load_model`.
    batch_size : int
    device : torch.device, optional

    Returns
    -------
    all_xyz : np.ndarray, shape (n_frames, n_atoms, 3)
    """
    import dgl
    import cg2all.lib.libcg
    from cg2all.lib.libdata import PredictionData, create_trajectory_from_batch

    if device is None:
        device = next(gg_model.parameters()).device

    # Build topology batch from PDB only (no trajectory)
    cb_val = getattr(gg_config.globals, "chain_break_cutoff", 10.0)
    pdata  = PredictionData(
        pdb_file, cg_class,
        radius=gg_config.globals.radius,
        chain_break_cutoff=0.1 * cb_val,
        batch_size=batch_size,
    )
    topo_loader = dgl.dataloading.GraphDataLoader(pdata, batch_size=1, shuffle=False)
    topo_batch  = next(iter(topo_loader)).to(device)

    n_beads = frames_cg.shape[1] // 3
    all_xyz = []
    for i, coords_flat in enumerate(frames_cg):
        coords = torch.tensor(
            coords_flat.reshape(n_beads, 3),
            dtype=torch.float32,
            device=device,
        )
        traj_cg, _ = create_trajectory_from_batch(topo_batch, coords)
        traj_aa    = cg2all.lib.libcg.backmap(traj_cg)
        all_xyz.append(traj_aa)
        if (i + 1) % 100 == 0:
            print(f"  Backmapped {i + 1} frames...")

    return np.stack(all_xyz)
