import os
import io
import glob
import numpy as np
from modeller import Environ
from modeller.scripts import complete_pdb
import mdtraj as md
import tempfile
import contextlib
from tqdm import tqdm
from biobb_structure_utils.utils.str_check_add_hydrogens import str_check_add_hydrogens   
from AI_ML.Utils.TopologyManager import *
from AI_ML.Utils.TrajectoryManager import *
from AI_ML.Utils.TrajectoryAnalysis import *

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
        hist_prot = np.int8((protonation_states))  # Example values
        # print(hist_prot)

        atom_info = TopUtils1.extract_atoms()
        hist_resname = [i[2] for i in TopUtils1.filter_atoms(atom_info, residue_names=['HIS'], atom_names=['CA'])]
        # print(hist_resname)

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

def add_missing_atoms(pdb_input,
                      modeller_topology='/home/bsc/bsc023645/scratch_mn4/Environments/aezip_env/lib/modeller-10.5/modlib/top_heav.lib',
                      modeller_parameters='/home/bsc/bsc023645/scratch_mn4/Environments/aezip_env/lib/modeller-10.5/modlib/par.lib'):
    env = Environ()
    env.libs.topology.read(modeller_topology)
    env.libs.parameters.read(modeller_parameters)
    m = complete_pdb(env, pdb_input)
    pdb_io = io.StringIO()
    m.write(file=pdb_io)
    return m, pdb_io.getvalue()

def trajectory_reconstruction(xtc_file, top_file, output_pdb_file, output_xtc_file, reference_pdb, protonation_states):
    traj = md.load(xtc_file, top=top_file)
    pdb_obj = []

    for i in tqdm(range(traj.n_frames), desc="Processing frames", file=sys.stdout):
        frame = traj[i]
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmpfile:
            tmp_filename = tmpfile.name + '.pdb'
        frame.save_pdb(tmp_filename)

        if i == 0:
           # Reconstruct the missing atoms
           m, _ = add_missing_atoms(tmp_filename)
           m.write(file=tmp_filename)

           # Add Hydrogens
           add_missing_hydrogens(pdb_target=tmp_filename, pdb_reference=reference_pdb, protonation_states=protonation_states)
        
        else:
           # Redirect stdout and stderr to os.devnull (silence output)
           original_stdout = sys.stdout
           original_stderr = sys.stderr
           try:
               sys.stdout = open(os.devnull, 'w')
               sys.stderr = open(os.devnull, 'w')

               # Reconstruct the missing atoms
               m, _ = add_missing_atoms(tmp_filename)
               m.write(file=tmp_filename)

               # Add Hydrogens
               add_missing_hydrogens(pdb_target=tmp_filename, pdb_reference=reference_pdb, protonation_states=protonation_states)

           finally:
               # Restore original stdout and stderr
               sys.stdout.close()
               sys.stderr.close()
               sys.stdout = original_stdout
               sys.stderr = original_stderr

        # Load the reconstructed PDB file
        p1 = md.load(tmp_filename)
        pdb_obj.append(p1)

        if i == 0:
           p1.save_pdb(output_pdb_file)

        os.remove(tmp_filename)
        remove_log_files()
    final_traj = md.join(pdb_obj) 

    final_traj.save_xtc(output_xtc_file) 
    return final_traj

