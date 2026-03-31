"""
Exposes all JSON config files in this directory as Python objects.

Usage:
    from aezip.config.config import (
        cartesian_definitions, dihedral_definitions,
        config, featurizer_config, model_config, trainer_config,
    )
"""
import json
import os

_HERE = os.path.dirname(__file__)


def _load(filename: str) -> dict:
    with open(os.path.join(_HERE, filename)) as f:
        return json.load(f)


cartesian_definitions = _load("aa_cart.json")
dihedral_definitions  = _load("aa_dih.json")
config                = _load("config.json")
featurizer_config     = _load("featurizer.json")
model_config          = _load("model.json")
trainer_config        = _load("trainer.json")
