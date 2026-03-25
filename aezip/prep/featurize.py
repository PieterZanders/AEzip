import mdtraj as md
import numpy as np
import multiprocessing as mp

def build_topology_dict(traj):
    """
    Constructs a dictionary of residues and their atoms with indices from the trajectory topology.

    Parameters:
        traj : md.Trajectory
            The molecular dynamics trajectory from which to extract topology information.

    Returns:
        topology_dict : dict
            Dictionary where each key is a residue, and each value is a dictionary of atoms and their indices.
    """
    topology_dict = {}
    for res in traj.topology.residues:
        atom_dict = {str(atom.name): atom.index for atom in res.atoms}
        topology_dict[str(res)] = atom_dict
    return topology_dict

def build_reslib_dict(res_library):
    """
    Constructs a dictionary from the residue library, mapping each residue to its atoms, geometry, and links.

    Parameters:
        res_library : ResidueLibrary
            A custom residue library object containing residue information.

    Returns:
        reslib_dict : dict
            Dictionary where each key is a residue name, and each value is a dictionary of atoms with geometry and links.
    """
    reslib_dict = {}
    for res in res_library.residues:
        atom_dict = {}
        for i, atom in enumerate(res_library.residues[res].ats):
            if atom and str(atom) != 'DUMM':
                reslib_ats = res_library.residues[res].ats
                link1 = str(reslib_ats[reslib_ats[i].link[0]])
                link2 = str(reslib_ats[reslib_ats[i].link[1]])
                link3 = str(reslib_ats[reslib_ats[i].link[2]])
                links = [link1, link2, link3]
                atom_dict[str(atom)] = [reslib_ats[i].geom, links]
        reslib_dict[res] = atom_dict
    return reslib_dict

def build_dihedral_atom_indices(traj, dihedral_definitions):
    """
    Constructs a dictionary mapping each residue to its dihedral atom indices based on dihedral definitions.

    Parameters:
        traj : md.Trajectory
            The trajectory topology from which to extract residue atom indices.
        dihedral_definitions : dict
            A dictionary defining the dihedral atoms for each residue type.

    Returns:
        dihedral_atom_indices : dict
            Dictionary where each residue maps to its dihedral definitions with atom names and indices.
    """
    dihedral_atom_indices = {}
    for residue in traj.topology.residues:
        dihedrals_for_residue = {}
        for atom_name, dihedral_atoms in dihedral_definitions[residue.name].items():
            atom_indices = {atom.name: atom.index for atom in residue.atoms if atom.name in dihedral_atoms}
            dihedrals_for_residue[str(atom_name)] = atom_indices
            dihedral_atom_indices[str(residue)] = dihedrals_for_residue
    return dihedral_atom_indices

def compute_dihedral_trajectory(traj, dihedral_indices_and_names, dihedral_atom_indices, data=None, dihedral_features_idx=None, use_precomputed=False):
     
    dihedral_traj = []
    for frame in range(len(traj)):
        aidx = 0
        dihedral_dict = {}
        for dihedrals1, dihedrals2 in zip(dihedral_indices_and_names.items(), dihedral_atom_indices.items()):
            residue_key = dihedrals1[0]
            dh_atom_dict = {}
            
            if use_precomputed and data is not None and dihedral_features_idx is not None:
                # Use precomputed values from `data[:, dihedral_features_idx]`
                for i, dh in enumerate(list(dihedrals2[1].keys())):
                    # Get the correct index for this dihedral angle from `dihedral_features_idx`
              
                    dihedral_idx = dihedral_features_idx[aidx]
                    dh_atom_dict[dh] = data[frame, dihedral_idx]
                    aidx += 1
            else:
                # Calculate dihedral angles using mdtraj
                dihedral_indices = [[atom_idx for atom_idx in dihedral.values()] for dihedral in dihedrals1[1]]
                dihedral_angles = md.compute_dihedrals(traj, dihedral_indices)
                for i, dh in enumerate(list(dihedrals2[1].keys())):
                    dh_atom_dict[dh] = np.degrees(dihedral_angles[0][i])
            
            dihedral_dict[str(residue_key)] = dh_atom_dict
        dihedral_traj.append(dihedral_dict)
    
    return dihedral_traj

def construct_dihedral_array(traj, dihedral_traj):   
    dih_traj = []
    for frame in range(len(traj.xyz)):
        frame_dihedrals = np.concatenate([list(value.values()) for value in dihedral_traj[frame].values()])
        dih_traj.append(frame_dihedrals)
    return np.array(dih_traj)

def get_histidine_protonation_states(traj):
    """
    Identifies the protonation states of histidine residues in a given MDTraj topology.

    Parameters:
        traj : md.Trajectory
            The molecular dynamics trajectory containing the topology.

    Returns:
        histidine_states : dict
            A dictionary with residue indices as keys and their protonation states (HIE, HID, HIP) as values.
    """
    histidine_states = {}
    
    for res in traj.topology.residues:
        if res.name == "HIS":
            # Check for the presence of specific atoms
            has_ND1_H = any(atom.name == 'HD1' for atom in res.atoms)
            has_NE2_H = any(atom.name == 'HE2' for atom in res.atoms)

            # Determine protonation state based on presence of protons
            if has_ND1_H and has_NE2_H:
                histidine_states[res.index] = "HIP"
            elif has_ND1_H:
                histidine_states[res.index] = "HID"
            elif has_NE2_H:
                histidine_states[res.index] = "HIE"
            else:
                histidine_states[res.index] = "Unknown"  # In case neither proton is found

    return histidine_states

def convert_full_to_sliced_indices(original_traj, sliced_traj):
    sliced_traj_atoms_indices = []
    for atom in sliced_traj.topology.atoms:
        match_idx = next(i for i, original_atom in enumerate(original_traj.topology.atoms)
                        if original_atom.serial == atom.serial)  # or use atom.name and atom.residue
        sliced_traj_atoms_indices.append(match_idx)
    sliced_traj_atoms_indices = np.array(sliced_traj_atoms_indices)
    return sliced_traj_atoms_indices

def get_dihedral_indices_and_names(traj, dihedral_definitions):
    dihedral_atom_indices = {}
    for residue in traj.topology.residues:
        if residue.name in dihedral_definitions and dihedral_definitions[residue.name]:
            dihedrals_for_residue = []
            for atom_name, dihedral_atoms in dihedral_definitions[residue.name].items():
                try:
                    atom_indices = {atom.name: atom.index for atom in residue.atoms if atom.name in dihedral_atoms + [atom_name]}
                    if len(atom_indices) == 4:
                        dihedrals_for_residue.append(atom_indices)
                except KeyError:
                    continue
            if dihedrals_for_residue:
                dihedral_atom_indices[f"{residue}"] = dihedrals_for_residue
    return dihedral_atom_indices

def compute_dihedrals_for_protein(traj, dihedral_indices_and_names):

    dihedral_data = {}

    for residue_key, dihedrals in dihedral_indices_and_names.items():
        dihedral_indices = [[atom_idx for atom_idx in dihedral.values()] for dihedral in dihedrals]

        dihedral_angles = md.compute_dihedrals(traj, dihedral_indices)

        dihedral_data[residue_key] = []

        dihedral_data[residue_key].append({
                    'dihedral_angle': np.degrees(dihedral_angles), 
                    'indices': dihedral_indices
                })

    return dihedral_data


def compute_dihedral_per_frame(frame, traj, dihedral_indices_and_names, dihedral_atom_indices,
                               data=None, dihedral_features_idx=None, use_precomputed=False):
    aidx = 0
    dihedral_dict = {}
    for dihedrals1, dihedrals2 in zip(dihedral_indices_and_names.items(), dihedral_atom_indices.items()):
        residue_key = dihedrals1[0]
        dh_atom_dict = {}

        if use_precomputed and data is not None and dihedral_features_idx is not None:
            for i, dh in enumerate(list(dihedrals2[1].keys())):
                dihedral_idx = dihedral_features_idx[aidx]
                dh_atom_dict[dh] = data[frame, dihedral_idx]
                aidx += 1
        else:
            dihedral_indices = [[atom_idx for atom_idx in dihedral.values()] for dihedral in dihedrals1[1]]
            dihedral_angles = md.compute_dihedrals(traj, dihedral_indices, indices=[frame])
            for i, dh in enumerate(list(dihedrals2[1].keys())):
                dh_atom_dict[dh] = np.degrees(dihedral_angles[0][i])

        dihedral_dict[str(residue_key)] = dh_atom_dict

    return dihedral_dict

def compute_dihedral_trajectory_parallel(traj, dihedral_indices_and_names, dihedral_atom_indices,
                                         data=None, dihedral_features_idx=None, use_precomputed=False, num_processes=None):
    # Define a partial function to include all constant parameters
    from functools import partial

    compute_frame_func = partial(
        compute_dihedral_per_frame,
        traj=traj,
        dihedral_indices_and_names=dihedral_indices_and_names,
        dihedral_atom_indices=dihedral_atom_indices,
        data=data,
        dihedral_features_idx=dihedral_features_idx,
        use_precomputed=use_precomputed
    )

    # Use multiprocessing Pool for parallel execution
    with mp.Pool(processes=num_processes) as pool:
        dihedral_traj = pool.map(compute_frame_func, range(len(traj)))

    return dihedral_traj

def calculate_dih_traj(traj, dihedral_indices_and_names):
    dih_idx = []
    for _, value in dihedral_indices_and_names.items():
        for sub_dict in value:
            dih_idx.append(list(sub_dict.values()))
    return np.degrees(md.compute_dihedrals(traj, np.array(dih_idx)))

def build_dih_traj(dih_traj, mapping):
    flat_mapping = []
    column_index = 0
    for residue, atoms in mapping.items():
        for atom in atoms:
            flat_mapping.append((residue, atom, column_index))
            column_index += 1

    # Build the output structure
    atom_angle_mapping = []
    for frame in dih_traj:
        frame_dict = {}
        for residue, atom, col_index in flat_mapping:
            if residue not in frame_dict:
                frame_dict[residue] = {}
            frame_dict[residue][atom] = frame[col_index]
        atom_angle_mapping.append(frame_dict)

    return atom_angle_mapping
