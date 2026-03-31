import os
import io
import sys
import glob
import numpy as np
from modeller import Environ
from modeller.scripts import complete_pdb
import mdtraj as md
import tempfile
import concurrent.futures
from tqdm import tqdm
from biobb_structure_utils.utils.str_check_add_hydrogens import str_check_add_hydrogens   
from aezip.utils.TopologyManager import TopologyManager

def remove_log_files():
    """
    Removes log files with the pattern 'log*.*' in the current directory.
    """
    log_files = glob.glob('log*.*')
    for log_file in log_files:
        try:
            os.remove(log_file)
        except OSError as e:
            print(f"Error deleting file {log_file}: {e}")

def generate_histidine_list(hist_prot, resids):
    histidine_list = []
    for i, value in enumerate(hist_prot):
        resid = resids[i]
        if value == 0:
            histidine_list.append(f"His{resid}Hid")
        elif value == 1:
            histidine_list.append(f"His{resid}Hie")
        elif value == 2:
            histidine_list.append(f"His{resid}HIP")
        elif value == 3:
            histidine_list.append(f"His{resid}HIS1")
    return ','.join(histidine_list)

def get_mdtraj_atoms(pdb):
    atoms = []
    topology = pdb.topology
    for atom in topology.atoms:
        atoms.append((atom.name, atom.residue.name, atom.residue.index))
    return atoms

def find_missing_atoms(atoms1, atoms2):
    missing_atoms = []
    atoms2_set = set(atoms2)
    for i, atom in enumerate(atoms1):
        if atom not in atoms2_set:
            missing_atoms.append(i)
    return missing_atoms

def add_missing_hydrogens(pdb_target, pdb_reference, protonation_states):
    """
    Processes PDB files to ensure consistency between target and reference.

    Args:
        pdb_target (str): Path to the target PDB file.
        pdb_reference (str): Path to the reference PDB file.
        protonation_states (list): List of protonation states for histidine residues.

    Returns:
        None
    """

    pdb1 = md.load(pdb_target)
    pdb2 = md.load(pdb_reference)
    TopUtils1 = TopologyManager(pdb1)
    TopUtils2 = TopologyManager(pdb2)
    #create_pdb(renumber_atom_idx(extract_atoms(pdb2)), None, output_pdb=pdb_reference)

    # Extract atoms
    atoms_pdb1 = get_mdtraj_atoms(pdb1)
    atoms_pdb2 = get_mdtraj_atoms(pdb2)

    # Find Wrong Histidine protonation
    his_h_idx = find_missing_atoms(atoms_pdb2, atoms_pdb1)
    his_h_resid = [atoms_pdb2[i] for i in his_h_idx]
    # print('Wrong His Protonation', his_h_resid)

    # Fix histidine protonation
    if len(his_h_resid) > 0:
        _his_state_to_int = {"HID": 0, "HIE": 1, "HIP": 2, "HIS": 3}

        atom_info = TopUtils1.extract_atoms()
        hist_resname = [i[2] for i in TopUtils1.filter_atoms(atom_info, residue_names=['HIS'], atom_names=['CA'])]
        # hist_resname contains residue indices; map each to its protonation state int
        hist_prot = np.int8([
            _his_state_to_int.get(protonation_states.get(idx, "HIS"), 3)
            for idx in hist_resname
        ])

        histidine_list = generate_histidine_list(hist_prot, hist_resname)
        # print(histidine_list)

        properties={'charges': False, 'ph': 7.4, 'mode': 'list', 'list': histidine_list, 'no_fix_side': True, 'keep_canonical_resnames': True, 'keep_h': True} 
        str_check_add_hydrogens(input_structure_path=pdb_target, output_structure_path=pdb_target, properties=properties) 

    # Match same Hydrogen atoms (and fix if needed)
    atoms_pdb1 = get_mdtraj_atoms(md.load(pdb_target))
    missing_atoms_indices = find_missing_atoms(atoms_pdb1, atoms_pdb2)
    # print([atoms_pdb2[i] for i in missing_atoms_indices])

    TopUtils1 = TopologyManager(md.load(pdb_target))
    filtered_atoms = [i for i in TopUtils1.extract_atoms() if i[5] not in missing_atoms_indices]
    TopUtils2.create_pdb(filtered_atoms, None, output_pdb=pdb_target)
    print("Lenght of PDB atoms:", len(filtered_atoms))

def _modeller_modlib() -> str:
    """Return the path to Modeller's modlib directory.

    Modeller installs its data files under {sys.prefix}/lib/modeller-{version}/modlib/,
    not inside the Python package directory.
    """
    import glob as _glob
    import sys as _sys
    candidates = _glob.glob(os.path.join(_sys.prefix, 'lib', 'modeller-*', 'modlib'))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find Modeller modlib under {_sys.prefix}/lib/modeller-*/modlib. "
            "Check that Modeller is installed in the active environment."
        )
    return sorted(candidates)[-1]  # take highest version if multiple

def add_missing_atoms(pdb_input, modeller_topology=None, modeller_parameters=None):
    modlib = _modeller_modlib()
    if modeller_topology is None:
        modeller_topology = os.path.join(modlib, 'top_heav.lib')
    if modeller_parameters is None:
        modeller_parameters = os.path.join(modlib, 'par.lib')
    env = Environ()
    env.libs.topology.read(modeller_topology)
    env.libs.parameters.read(modeller_parameters)
    m = complete_pdb(env, pdb_input)
    pdb_io = io.StringIO()
    m.write(file=pdb_io)
    return m, pdb_io.getvalue()

def _process_frame(args):
    """
    Worker function that reconstructs a single frame in a subprocess.
    Each subprocess has isolated memory so stdout/stderr redirection and
    Modeller's Environ are independent across workers.

    Returns (frame_index, xyz, topology_pickle) so the main process can
    reassemble frames in order without keeping md.Trajectory objects in
    worker memory.
    """
    import pickle
    frame_idx, frame_xyz, frame_top_pickle, reference_pdb, protonation_states = args

    frame_top = pickle.loads(frame_top_pickle)
    frame = md.Trajectory(frame_xyz, topology=frame_top)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
        tmp_filename = tmp.name

    frame.save_pdb(tmp_filename)

    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            # Step 1: fill in missing heavy atoms with Modeller
            m, _ = add_missing_atoms(tmp_filename)
            m.write(file=tmp_filename)

            # Step 2: add all hydrogens at physiological pH
            str_check_add_hydrogens(
                input_structure_path=tmp_filename,
                output_structure_path=tmp_filename,
                properties={
                    'charges': False,
                    'ph': 7.4,
                    'keep_canonical_resnames': True,
                    'keep_h': False,
                },
            )

            # Step 3: fix HIS protonation states to match the reference,
            #         then trim any atoms not present in the reference
            add_missing_hydrogens(
                pdb_target=tmp_filename,
                pdb_reference=reference_pdb,
                protonation_states=protonation_states,
            )
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    result = md.load(tmp_filename)
    os.remove(tmp_filename)
    remove_log_files()

    return frame_idx, result.xyz, pickle.dumps(result.topology)


def trajectory_reconstruction(traj: md.Trajectory, output_pdb_file: str, output_xtc_file: str,
                               reference_pdb, protonation_states, n_workers=None):
    import pickle
    top_pickle = pickle.dumps(traj.topology)

    worker_args = [
        (i, traj[i].xyz, top_pickle, reference_pdb, protonation_states)
        for i in range(traj.n_frames)
    ]

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_frame, a): a[0] for a in worker_args}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=traj.n_frames,
            desc="Processing frames",
            file=sys.stdout,
        ):
            frame_idx, xyz, top_pkl = future.result()
            results[frame_idx] = md.Trajectory(xyz, topology=pickle.loads(top_pkl))

    ordered = [results[i] for i in range(traj.n_frames)]
    ordered[0].save_pdb(output_pdb_file)

    final_traj = md.join(ordered)
    final_traj.save_xtc(output_xtc_file)
    return final_traj

