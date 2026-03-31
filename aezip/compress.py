import os
import sys
import copy
import json
import argparse
import numpy as np
import mdtraj as md
import torch

# Allow running as `python compress.py` directly from inside aezip/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aezip.config.config import (
    cartesian_definitions, dihedral_definitions,
    featurizer_config, model_config, trainer_config,
)
from aezip.prep.featurize import get_dihedral_indices_and_names, calculate_dih_traj
from biobb_pytorch.mdae.mdfeaturizer import MDFeaturizer
from biobb_pytorch.mdae.build_model import BuildModel
from biobb_pytorch.mdae.train_model import TrainModel
from biobb_pytorch.mdae.evaluate_model import EvaluateModel


def _build_cart_selection(cart_defs: dict) -> str:
    return " or ".join(
        f"(resname {res} and ({' or '.join(f'name {a}' for a in atoms)}))"
        for res, atoms in cart_defs.items()
    )


def _hidden_layers(n_features: int, n_cvs: int, n_layers: int):
    """Compute intermediate hidden layer sizes for encoder and decoder.

    biobb_pytorch builds:
        encoder = FeedForward([n_features] + encoder_layers + [n_cvs])
        decoder = FeedForward([n_cvs]      + decoder_layers + [n_features])
    so encoder_layers/decoder_layers must NOT include the input/output endpoints.
    """
    step = max(1, (n_features - n_cvs) // n_layers)
    encoder_hidden = [max(n_cvs + 1, n_features - i * step) for i in range(1, n_layers)]
    decoder_hidden = list(reversed(encoder_hidden))
    return encoder_hidden, decoder_hidden


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compress a trajectory using an autoencoder.")
parser.add_argument("-f", "--traj-file",   required=True, help="Input trajectory (.xtc)")
parser.add_argument("-s", "--top-file",    required=True, help="Topology file (.pdb)")
parser.add_argument("-o", "--output-file", default="./compressed_traj.pth",
                    help="Output compressed file (default: ./compressed_traj.pth)")
parser.add_argument("-c", "--config-file", default="./config/config.json",
                    help="Main config JSON (default: ./config/config.json)")
parser.add_argument("--stride", type=int, default=1, metavar="N",
                    help="Use every N-th frame (default: 1, all frames)")
parser.add_argument("--no-ae", action="store_true",
                    help="Skip autoencoder: save raw features directly "
                         "(featurize only, no training or encoding)")
args = parser.parse_args()

with open(args.config_file) as f:
    run_config = json.load(f)
hp               = run_config["hyperparams"]
compression_type = run_config["compression_type"]

# ---------------------------------------------------------------------------
# Featurize
# ---------------------------------------------------------------------------
if compression_type not in featurizer_config:
    raise ValueError(
        f"Unknown compression_type {compression_type!r}. "
        f"Valid options: {list(featurizer_config)}"
    )

if compression_type == "cartesian":
    # biobb_pytorch handles cartesian featurization end-to-end
    feat_props = copy.deepcopy(featurizer_config["cartesian"])
    feat_props["cartesian"]["selection"] = _build_cart_selection(cartesian_definitions)

    if args.stride > 1:
        feat_props["stride"] = args.stride

    feat = MDFeaturizer(
        input_topology_path=args.top_file,
        input_trajectory_path=args.traj_file,
        output_stats_pt_path=None,
        output_dataset_pt_path=None,
        properties=feat_props,
    )

elif compression_type == "dihedral":
    # biobb_pytorch does NOT support the custom per-residue dihedrals in
    # aa_dih.json, so we featurize manually and build a compatible stats dict.
    traj = md.load(args.traj_file, top=args.top_file, stride=args.stride)

    backbone_sel = featurizer_config["dihedral"]["cartesian"]["selection"]
    sliced_traj  = traj.atom_slice(traj.topology.select(backbone_sel))
    coord_feat   = sliced_traj.xyz.reshape(len(sliced_traj), -1)

    dih_names_and_idx = get_dihedral_indices_and_names(traj, dihedral_definitions)
    dih_feat          = calculate_dih_traj(traj, dih_names_and_idx)

    raw_feat = np.concatenate((coord_feat, dih_feat), axis=1).astype(np.float32)

    # Create a minimal biobb-compatible stats dict for BuildModel
    feat_tensor = torch.FloatTensor(raw_feat)
    feat_stats  = {
        "min":   feat_tensor.min(dim=0).values,
        "max":   feat_tensor.max(dim=0).values,
        "mean":  feat_tensor.mean(dim=0),
        "std":   feat_tensor.std(dim=0),
        "shape": list(raw_feat.shape),
        "cartesian_indices": list(range(coord_feat.shape[1])),
        "dihedral_indices":  list(range(coord_feat.shape[1], raw_feat.shape[1])),
        "distance_indices":  [],
        "angle_indices":     [],
    }

    from mlcolvar.data import DictDataset
    feat_dataset = DictDataset({"data": feat_tensor})

    # Wrap in a lightweight namespace so the rest of the script uses .stats/.dataset
    class _Feat:
        stats   = feat_stats
        dataset = feat_dataset
    feat = _Feat()

elif compression_type == "cg2all":
    from aezip.model.cg2all import load_model as _cg2all_load, extract_cg_coords
    from mlcolvar.data import DictDataset

    cg_cfg    = featurizer_config["cg2all"]
    cg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gg_model, gg_config, cg_class = _cg2all_load(
        cg_model_type=cg_cfg["cg_model_type"],
        device=cg_device,
    )
    raw_feat = extract_cg_coords(
        pdb_file=args.top_file,
        traj_file=args.traj_file,
        gg_model=gg_model,
        gg_config=gg_config,
        cg_class=cg_class,
        batch_size=cg_cfg.get("batch_size", 1),
        stride=args.stride,
        device=cg_device,
    )   # (n_frames, n_beads * 3)

    feat_tensor = torch.FloatTensor(raw_feat)
    feat_stats  = {
        "min":   feat_tensor.min(dim=0).values,
        "max":   feat_tensor.max(dim=0).values,
        "mean":  feat_tensor.mean(dim=0),
        "std":   feat_tensor.std(dim=0),
        "shape": list(raw_feat.shape),
        "cartesian_indices": list(range(raw_feat.shape[1])),
        "dihedral_indices":  [],
        "distance_indices":  [],
        "angle_indices":     [],
    }
    feat_dataset = DictDataset({"data": feat_tensor})

    class _Feat:
        stats   = feat_stats
        dataset = feat_dataset
    feat = _Feat()

n_features = feat.stats["shape"][1]
print(f"Features: {n_features}")

# ---------------------------------------------------------------------------
# Autoencoder (build → train → encode)  —  skipped when --no-ae is set
# ---------------------------------------------------------------------------
if args.no_ae:
    # Store raw features as the compressed representation
    if compression_type == "cartesian":
        raw = feat.dataset["data"]
    else:  # dihedral: already a numpy array in raw_feat
        raw = raw_feat
    ae_model = None
    z        = raw
    print("--no-ae: skipping autoencoder, storing raw features")
    print(f"  Frames: {len(z)}  |  Features: {z.shape[1]}")
else:
    n_cvs = hp["latent_dim"]
    encoder_hidden, decoder_hidden = _hidden_layers(n_features, n_cvs, hp["layers"])
    print(f"Latent dim: {n_cvs}  |  Encoder hidden layers: {encoder_hidden}")

    # Build model
    ae_model = BuildModel(
        input_stats=feat.stats,
        output_model_pth_path=None,
        properties={
            "model_type":     model_config["model_type"],
            "n_cvs":          n_cvs,
            "encoder_layers": encoder_hidden,
            "decoder_layers": decoder_hidden,
            "options": {
                "encoder":       model_config["encoder"],
                "decoder":       model_config["decoder"],
                "optimizer":     {"lr": hp["learning_rate"], "weight_decay": hp["weight_decay"]},
                "loss_function": {"loss_type": hp["loss_function"]},
            },
        },
    )
    print(ae_model.model)

    # Train
    train_props = copy.deepcopy(trainer_config)
    train_props["Dataset"]["batch_size"] = hp["batch_size"]
    train_props["Trainer"]["max_epochs"] = hp["num_epochs"]

    tm = TrainModel(
        input_model=ae_model.model,
        input_dataset=feat.dataset,
        properties=train_props,
    )
    tm.run_training()

    # Encode — collect latent vectors for the full dataset
    em = EvaluateModel(
        input_model=ae_model.model,
        input_dataset=feat.dataset,
        properties={"Dataset": {"batch_size": hp["batch_size"]}},
    )
    z = em.run_evaluation()["z"]   # numpy array, shape (n_frames, n_cvs)
    print(f"  Frames: {len(z)}  |  Latent dim: {z.shape[1]}")

# ---------------------------------------------------------------------------
# Save reference topology for backmapping
# ---------------------------------------------------------------------------
traj_ref = md.load(args.top_file)

if compression_type == "cartesian":
    compressed_topology = traj_ref.atom_slice(
        traj_ref.topology.select(_build_cart_selection(cartesian_definitions))
    ).topology
elif compression_type == "dihedral":
    compressed_topology = traj_ref.atom_slice(
        traj_ref.topology.select(
            featurizer_config["dihedral"]["cartesian"]["selection"]
        )
    ).topology
else:  # cg2all — backmapping recreates full topology; no sliced topology needed
    compressed_topology = None

cg_cfg = featurizer_config.get("cg2all", {})
torch.save({
    "model":               ae_model.model if ae_model is not None else None,
    "compressed_traj":     z,
    "stats":               feat.stats,
    "hyperparams":         hp,
    "topology":            traj_ref[0],
    "compressed_topology": compressed_topology,
    "compression_type":    compression_type,
    "use_ae":              not args.no_ae,
    "cg_model_type":       cg_cfg.get("cg_model_type"),
    "cg_batch_size":       cg_cfg.get("batch_size", 1),
}, args.output_file)

print(f"\nCompressed trajectory saved to: {args.output_file}")
