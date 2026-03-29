"""
Convenience module that exposes the JSON config files as Python objects.

Usage:
    from aezip.config.config import cartesian_definitions, dihedral_definitions, config
"""
import json
import os

_HERE = os.path.dirname(__file__)

with open(os.path.join(_HERE, "aa_cart.json")) as _f:
    cartesian_definitions = json.load(_f)

with open(os.path.join(_HERE, "aa_dih.json")) as _f:
    dihedral_definitions = json.load(_f)

with open(os.path.join(_HERE, "config.json")) as _f:
    config = json.load(_f)
