import os
import sys
import copy
import json
import argparse
import numpy as np
import mdtraj as md
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aezip.utils.logger import RunLogger

# Allow running as `python compress.py` directly from inside aezip/

from aezip.config import (
    cartesian_definitions, dihedral_definitions,
    featurizer_config, model_config, trainer_config, evaluator_config,
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

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compress a trajectory using an autoencoder.")
parser.add_argument("-f", "--traj-file",   required=True, help="Input trajectory (.xtc)")
parser.add_argument("-s", "--top-file",    required=True, help="Topology file (.pdb)")
parser.add_argument("-o", "--output-file", default="./compressed_traj.pth",
                    help="Output compressed file (default: ./compressed_traj.pth)")
parser.add_argument("--stride", type=int, default=1, metavar="N",
                    help="Use every N-th frame (default: 1, all frames)")
parser.add_argument("--no-ae", action="store_true",
                    help="Skip autoencoder: save raw features directly "
                         "(featurize only, no training or encoding)")
parser.add_argument("--ae-input",  default=None, metavar="FILE",
                    help="Save AE input features (normalised) to this .npy file")
parser.add_argument("--ae-output", default=None, metavar="FILE",
                    help="Save AE reconstructed output to this .npy file")
parser.add_argument("-t", "--compression-type", default="cartesian",
                    choices=["cartesian", "dihedral", "cg2all"],
                    help="Compression type (default: cartesian)")
parser.add_argument("--log-file",  default=None, metavar="FILE",
                    help="JSON log file (default: <output>.log.json)")
args = parser.parse_args()

log_path = args.log_file or (os.path.splitext(args.output_file)[0] + ".log.json")
logger   = RunLogger(log_path, run_type="compress", args=vars(args))
logger.log_input_file(args.traj_file,   label="trajectory")
logger.log_input_file(args.top_file,    label="topology")

compression_type = args.compression_type

# ---------------------------------------------------------------------------
# Featurize
# ---------------------------------------------------------------------------
if compression_type not in featurizer_config:
    raise ValueError(
        f"Unknown compression_type {compression_type!r}. "
        f"Valid options: {list(featurizer_config)}"
    )

with logger.step(f"Featurize ({compression_type})"):
    if compression_type == "cartesian":
        feat_props = copy.deepcopy(featurizer_config["cartesian"])
        feat_props["cartesian"]["selection"] = _build_cart_selection(cartesian_definitions)

        # MDFeaturizer has no stride parameter — pre-slice the trajectory.
        if args.stride > 1:
            import tempfile
            _tmp_traj = md.load(args.traj_file, top=args.top_file, stride=args.stride)
            _tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xtc")
            _tmp_traj.save_xtc(_tmp_file.name)
            _tmp_file.close()
            _cart_traj_path = _tmp_file.name
        else:
            _cart_traj_path = args.traj_file

        try:
            feat = MDFeaturizer(
                input_topology_path=args.top_file,
                input_trajectory_path=_cart_traj_path,
                output_stats_pt_path=None,
                output_dataset_pt_path=None,
                properties=feat_props,
            )
        finally:
            if args.stride > 1:
                os.remove(_cart_traj_path)

    elif compression_type == "dihedral":
        traj = md.load(args.traj_file, top=args.top_file, stride=args.stride)
        backbone_sel = featurizer_config["dihedral"]["cartesian"]["selection"]
        sliced_traj  = traj.atom_slice(traj.topology.select(backbone_sel))
        coord_feat   = sliced_traj.xyz.reshape(len(sliced_traj), -1)
        dih_names_and_idx = get_dihedral_indices_and_names(traj, dihedral_definitions)
        dih_feat          = calculate_dih_traj(traj, dih_names_and_idx)
        raw_feat = np.concatenate((coord_feat, dih_feat), axis=1).astype(np.float32)
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
        class _FeatDih:
            stats   = feat_stats
            dataset = feat_dataset
        feat = _FeatDih()

    elif compression_type == "cg2all":
        from aezip.model.cg2all import load_model as _cg2all_load, extract_cg_coords
        from mlcolvar.data import DictDataset
        cg_cfg    = featurizer_config["cg2all"]
        cg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gg_model, gg_config, cg_class = _cg2all_load(
            cg_model_type=cg_cfg["cg_model_type"], device=cg_device,
        )
        raw_feat = extract_cg_coords(
            pdb_file=args.top_file, traj_file=args.traj_file,
            gg_model=gg_model, gg_config=gg_config, cg_class=cg_class,
            batch_size=cg_cfg.get("batch_size", 1), stride=args.stride, device=cg_device,
        )
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
        class _FeatCG:
            stats   = feat_stats
            dataset = feat_dataset
        feat = _FeatCG()

n_features = feat.stats["shape"][1]
print(f"Features: {n_features}")

# ---------------------------------------------------------------------------
# Autoencoder (build → train → encode)  —  skipped when --no-ae is set
# ---------------------------------------------------------------------------
if args.no_ae:
    with logger.step("Store raw features (no AE)"):
        raw      = feat.dataset["data"] if compression_type == "cartesian" else raw_feat
        ae_model = None
        z        = raw
        print(f"  Frames: {len(z)}  |  Features: {z.shape[1]}")
else:
    with logger.step("Build model"):
        ae_model = BuildModel(
            input_stats=feat.stats,
            output_model_pth_path=None,
            properties=model_config,
        )
        print(ae_model.model)

    with logger.step("Train"):
        tm = TrainModel(
            input_model=ae_model.model,
            input_dataset=feat.dataset,
            properties=trainer_config,
        )
        tm.run_training()

    with logger.step("Encode"):
        em     = EvaluateModel(
            input_model=ae_model.model,
            input_dataset=feat.dataset,
            properties=evaluator_config,
        )
        em_out = em.run_evaluation()
        z      = em_out["z"]
        xhat   = em_out["xhat"]
        x_in   = feat.dataset["data"]
        print(f"  Frames: {len(z)}  |  Latent dim: {z.shape[1]}")

    if args.ae_input is not None:
        np.save(args.ae_input, x_in)
        logger.log_file(args.ae_input, label="ae_input")
    if args.ae_output is not None:
        np.save(args.ae_output, xhat)
        logger.log_file(args.ae_output, label="ae_output")

# ---------------------------------------------------------------------------
# Save reference topology for backmapping
# ---------------------------------------------------------------------------
with logger.step("Save"):
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
    else:
        compressed_topology = None

    cg_cfg = featurizer_config.get("cg2all", {})
    torch.save({
        "model":               ae_model.model if ae_model is not None else None,
        "compressed_traj":     z,
        "stats":               feat.stats,
        "hyperparams":         model_config,
        "topology":            traj_ref[0],
        "compressed_topology": compressed_topology,
        "compression_type":    compression_type,
        "use_ae":              not args.no_ae,
        "cg_model_type":       cg_cfg.get("cg_model_type"),
        "cg_batch_size":       cg_cfg.get("batch_size", 1),
    }, args.output_file)

logger.log_file(args.output_file, label="compressed")
logger.save()
