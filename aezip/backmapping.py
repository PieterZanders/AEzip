import os
import warnings
import numpy as np
import mdtraj as md

from .utils.modelling import (
    build_coords_from_internal,
    build_coords_backbone_oxygen,
    CBINTERNALS,
    OINTERNALS,
)

_HERE = os.path.dirname(__file__)


def backmap_trajectory(data, topology, sliced_topology, reslib_dict, cartesian_definitions=None):
    """
    Reconstruct full-atom trajectory from partial cartesian coordinates.

    Parameters
    ----------
    data : np.ndarray, shape (n_frames, n_features)
        Inverse-transformed decoded coordinates in nm.
    topology : md.Trajectory
        Full-atom topology reference (from md.load).
    sliced_topology : md.Trajectory
        Partial topology containing only the aa_cart.json atoms.
    reslib_dict : dict
        Residue geometry library (from build_reslib_dict).
    cartesian_definitions : dict, optional
        Unused; kept for backwards-compatible call sites.

    Returns
    -------
    md.Trajectory
        Full-atom trajectory in nm.
    """
    full_topology = topology.topology

    # Map each atom in sliced_topology back to its index in the full topology.
    # Matching by (chain index, residue index, atom name) is robust regardless of
    # how many atoms the full topology has — avoids re-running cart_sel which may
    # select a different count if the full topology differs from the compressed one.
    full_atom_lookup = {
        (atom.residue.chain.index, atom.residue.index, atom.name): atom.index
        for atom in full_topology.atoms
    }
    sliced_indices = np.array([
        full_atom_lookup[(a.residue.chain.index, a.residue.index, a.name)]
        for a in sliced_topology.topology.atoms
    ])
    available_indices = set(sliced_indices.tolist())

    # Build full-topology atom index lookup: {str(res): {atom_name: global_index}}
    topology_dict = {
        str(res): {atom.name: atom.index for atom in res.atoms}
        for res in full_topology.residues
    }

    residues_list = list(full_topology.residues)
    n_frames = len(data)
    n_atoms = full_topology.n_atoms

    # Working buffer in Å — reslib bond lengths are in Å; mdtraj uses nm.
    # data after inverse_transform is in nm → multiply by 10.
    trajectory = np.zeros((n_frames, n_atoms, 3))
    trajectory[:, sliced_indices, :] = data.reshape(n_frames, -1, 3) * 10.0  # nm -> Å

    for frame in range(n_frames):
        for i, res in enumerate(residues_list):
            next_res = residues_list[i + 1] if i + 1 < len(residues_list) else None
            resname = res.name

            # Dict of already-placed atoms for this residue: name -> coords (Å)
            res_atoms = {
                atom.name: trajectory[frame, atom.index, :]
                for atom in res.atoms
                if atom.index in available_indices
            }

            # Missing heavy atoms (skip hydrogens and terminal OXT)
            missing = [
                atom for atom in res.atoms
                if atom.name not in res_atoms
                and atom.element.symbol != 'H'
                and atom.name != 'OXT'
            ]

            if not missing:
                continue

            # Retry loop: each pass places atoms whose 3 link atoms are already available.
            # Atoms with unresolved dependencies are deferred to the next pass.
            for _ in range(len(missing) + 1):
                if not missing:
                    break

                still_missing = []

                for atom in missing:
                    coords = None
                    atom_name = atom.name

                    # CB: standard backbone-derived placement (CA, N, C as reference)
                    if atom_name == 'CB':
                        if all(k in res_atoms for k in ('CA', 'N', 'C')):
                            coords = build_coords_from_internal(
                                res_atoms['CA'], res_atoms['N'], res_atoms['C'], CBINTERNALS
                            )

                    # O: peptide-plane geometry requires the next residue's N
                    elif atom_name == 'O':
                        if all(k in res_atoms for k in ('C', 'CA')):
                            if next_res is not None:
                                n_next_idx = topology_dict.get(str(next_res), {}).get('N')
                                if n_next_idx is not None:
                                    n_next_c = trajectory[frame, n_next_idx, :]
                                    o_entry = reslib_dict.get(resname, {}).get('O')
                                    geom = o_entry[0] if o_entry else OINTERNALS
                                    coords = build_coords_backbone_oxygen(
                                        res_atoms['C'], res_atoms['CA'], n_next_c, geom
                                    )
                            else:
                                # C-terminal residue: fall back to library internal coords
                                o_entry = reslib_dict.get(resname, {}).get('O')
                                if o_entry is not None:
                                    geom, links = o_entry
                                    l1, l2, l3 = links
                                    if all(k in res_atoms for k in (l1, l2, l3)):
                                        coords = build_coords_from_internal(
                                            res_atoms[l1], res_atoms[l2], res_atoms[l3], geom
                                        )

                    # General: geometry and reference atoms from the residue library
                    else:
                        entry = reslib_dict.get(resname, {}).get(atom_name)
                        if entry is not None:
                            geom, links = entry
                            l1, l2, l3 = links
                            if all(k in res_atoms for k in (l1, l2, l3)):
                                coords = build_coords_from_internal(
                                    res_atoms[l1], res_atoms[l2], res_atoms[l3], geom
                                )

                    if coords is not None:
                        trajectory[frame, atom.index, :] = coords
                        res_atoms[atom_name] = coords
                    else:
                        still_missing.append(atom)

                # No progress this pass → unresolvable, warn and stop retrying
                if len(still_missing) == len(missing):
                    for atom in still_missing:
                        warnings.warn(
                            f"Frame {frame}: could not reconstruct {resname}{res.index} "
                            f"atom '{atom.name}' — link atoms unavailable in residue library"
                        )
                    break

                missing = still_missing

    # Convert Å -> nm and return as mdtraj Trajectory
    return md.Trajectory(trajectory / 10.0, topology=full_topology)
