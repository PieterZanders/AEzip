import inspect
import os, io, sys
import json
import requests
import numpy as np
import mdtraj as md
from Bio.PDB import PDBIO, PDBParser, PDBList, Structure, Model, Chain, Residue, Atom

class TopologyManager:
    def __init__(self, mdtraj_obj, domains=None, secondary_structure_regions=None, functional_sites=None, interfaces=None, variants=None, custom_regions=None):
        if isinstance(mdtraj_obj, str):
            mdtraj_obj = md.load(mdtraj_obj)
        self.traj = mdtraj_obj
        self.domains = domains if domains is not None else {}
        self.secondary_structure_regions = secondary_structure_regions if secondary_structure_regions is not None else {}
        self.functional_sites = functional_sites if functional_sites is not None else {}
        self.interfaces = interfaces if interfaces is not None else {}
        self.custom_regions = custom_regions if custom_regions is not None else {}
        self.variants = variants if variants is not None else {}

    def generate_random_topology(self, n_atoms_per_residue: int, n_residues: int=1) -> md.Topology:
        topology = md.Topology()
        chain = topology.add_chain()
        for i in range(n_residues):
            residue = topology.add_residue(f'R{i+1}', chain)
            for j in range(n_atoms_per_residue):
                topology.add_atom(f'A{j+1}', md.element.carbon, residue)
        return topology

    def renumber_atom_idx(self, atom_info):
        return [(t[0], t[1], t[2], t[3], t[4], new_index, t[6], t[7], t[8], t[9], t[10], t[11])
                for new_index, t in zip(np.arange(len(atom_info)), atom_info)]

    def renumber_residue_idx(self, atom_info):
        renumbered_residues = []
        resid = 1
        prev_residue = None
        prev_chain = None

        for atom in atom_info:
            chain, residue_index = atom[0], atom[2]

            if chain != prev_chain:
                resid = 1
            elif residue_index != prev_residue:
                resid += 1

            renumbered_residues.append((
                chain, atom[1], resid, atom[3], atom[4], atom[5],
                atom[6], atom[7], atom[8], atom[9], atom[10], atom[11]
            ))
            prev_residue = residue_index
            prev_chain = chain

        return renumbered_residues

    def get_domain(self, residue):
        res_id = residue.resSeq
        for domain, info in self.domains.items():
            res_ranges = info['ResIDs']
            if isinstance(res_ranges[0], list):
                for res_range in res_ranges:
                    if res_range[0] <= res_id <= res_range[1]:
                        return domain
            else:
                if res_ranges[0] <= res_id <= res_ranges[1]:
                    return domain
        return None

    def get_secondary_structures(self, residue):
        for sec_struct, info in self.secondary_structure_regions.items():
            if info['ResIDs'][0] <= residue.resSeq <= info['ResIDs'][-1]:
                return sec_struct
        return "Ligand/Cofactor"

    def get_functional_sites(self, residue):
        res_id = residue.resSeq
        sites = []
        for site, info in self.functional_sites.items():
            res_ranges = info['ResIDs']
            if isinstance(res_ranges[0], list):
                for res_range in res_ranges:
                    if res_range[0] <= res_id <= res_range[1]:
                        sites.append(site)
            else:
                if res_ranges[0] <= res_id <= res_ranges[1]:
                    sites.append(site)
        return sites if sites else None

    def get_interfaces(self, residue):
        res_id = residue.resSeq
        interfaces_list = []
        for site, info in self.interfaces.items():
            res_ranges = info['ResIDs']
            if isinstance(res_ranges[0], list):
                for res_range in res_ranges:
                    if res_range[0] <= res_id <= res_range[1]:
                        interfaces_list.append(site)
            else:
                if res_ranges[0] <= res_id <= res_ranges[1]:
                    interfaces_list.append(site)
        return interfaces_list if interfaces_list else None

    def get_variants(self, residue):
        res_id = residue.resSeq
        variants_list = []
        for site, info in self.variants.items():
            res_ranges = info['ResIDs']
            if isinstance(res_ranges[0], list):
                for res_range in res_ranges:
                    if len(res_range) == 2 and res_range[0] <= res_id <= res_range[1]:
                       variants_list.append({site: info['Status']})
                    elif len(res_range) == 1 and res_range[0] == res_id:
                       variants_list.append({site: info['Status']})
            else:
                if len(res_ranges) == 2 and res_ranges[0] <= res_id <= res_ranges[1]:
                    variants_list.append({site: info['Status']})
                elif len(res_ranges) == 1 and res_ranges[0] == res_id:
                    variants_list.append({site: info['Status']})
        return variants_list if variants_list else None

    def get_custom_regions(self, residue):
        res_id = residue.resSeq
        custom_list = []
        for site, info in self.custom_regions.items():
            res_ranges = info['ResIDs']
            if isinstance(res_ranges[0], list):
                for res_range in res_ranges:
                    if res_range[0] <= res_id <= res_range[1]:
                        custom_list.append(site)
            else:
                if res_ranges[0] <= res_id <= res_ranges[1]:
                    custom_list.append(site)
        return custom_list if custom_list else None

    def extract_atoms(self):
        atoms_info = []
        for frame_idx in range(self.traj.n_frames):
            for residue in self.traj.topology.residues:
                chain_letter = chr(65 + residue.chain.index)
                domain = self.safe_call(self.get_domain, residue)
                secondary_structure = self.safe_call(self.get_secondary_structures, residue)
                functional_sites = self.safe_call(self.get_functional_sites, residue)
                interfaces_list = self.safe_call(self.get_interfaces, residue)
                custom_region = self.safe_call(self.get_custom_regions, residue)
                variant = self.safe_call(self.get_variants, residue)
                for atom in residue.atoms:
                    coordinates = self.traj.xyz[frame_idx, atom.index] * 10
                    atoms_info.append((
                        chain_letter, residue.name, residue.resSeq, atom.name, atom.element.symbol,
                        atom.index, domain, secondary_structure, functional_sites,
                        interfaces_list, variant, custom_region, coordinates
                    ))
        return atoms_info

    @staticmethod
    def safe_call(func, *args, default=None):
        try:
            return func(*args)
        except Exception as e:
            print(f"An error occurred: {e}")
            return default

    def filter_atoms(self, atoms_info, residue_names=None, atom_names=None, element_name=None, chain_letters=None, domains=None, secondary_structures=None, functional_sites=None, interfaces=None, variants=None, custom_region=None, atom_indices=None, json_file=None):
        filtered_atoms = atoms_info

        self.filter_selections = {'residue_names': residue_names, 'atom_names': atom_names, 'element_name': element_name,
            'chain_letters': chain_letters, 'domains': domains, 'secondary_structures': secondary_structures,
            'functional_sites': functional_sites, 'interfaces': interfaces, 'variants': variants, 'custom_region': custom_region,
            'atom_indices': atom_indices}

        # Filter by specific attributes if provided
        filter_conditions = {
            'residue_names': 1,
            'chain_letters': 0,
            'domains': 6,
            'element_name': 4,
            'atom_indices': 5,
            'custom_region': 11
        }

        # Load additional filters from a JSON file if provided
        if json_file:
            custom_atoms = []
            with open(json_file, 'r') as json_file:
                json_filters = json.load(json_file).get("filter_atoms", [])
            for filter_params in json_filters:
                custom_atoms.extend(self.filter_atoms(atoms_info, **filter_params))

            filtered_atoms = sorted(custom_atoms, key=lambda x: x[5])

        for key, index in filter_conditions.items():
            value = locals()[key]
            if value is not None:
                filtered_atoms = [atom for atom in filtered_atoms if atom[index] in value]

        # Filter by secondary structures
        if secondary_structures is not None:
            if "Loops" in secondary_structures:
                filtered_atoms = [atom for atom in filtered_atoms if atom[7].startswith("L-")]
            elif "Helix" in secondary_structures:
                filtered_atoms = [atom for atom in filtered_atoms if not atom[7].startswith("L-") and "α" in atom[7]]
            elif "Strand" in secondary_structures:
                filtered_atoms = [atom for atom in filtered_atoms if not atom[7].startswith("L-") and "β" in atom[7]]
            else:
                filtered_atoms = [atom for atom in filtered_atoms if atom[7] in secondary_structures]

        # Filter by Sidechains
        if atom_names is not None:
            if "Sidechains" in atom_names:
                filtered_atoms = [atom for atom in filtered_atoms if atom[3] not in ['CA', 'N', 'C', 'O']]
            elif "Backbone" in atom_names:
                filtered_atoms = [atom for atom in filtered_atoms if not atom[3] in ['CA', 'N', 'C', 'O']]
            elif "Heavy" in atom_names:
                filtered_atoms = [atom for atom in filtered_atoms if not atom[4] in ['C', 'N', 'O', 'S', 'P']]
            elif "Hydrogens" in atom_names:
                filtered_atoms = [atom for atom in filtered_atoms if atom[4] in ['H']]
            else:
                filtered_atoms = [atom for atom in filtered_atoms if atom[3] in atom_names]

        # Filter by functional sites and interfaces
        for key, index in {'functional_sites': 8, 'interfaces': 9}.items():
            value = locals()[key]
            if value is not None:
                filtered_atoms = [atom for atom in filtered_atoms if any(site in (atom[index] or []) for site in value)]

        return filtered_atoms

    def create_pdb(self, filtered_atoms, b_factors=None, numpy_array=None, select_frame=0, output_pdb="filtered_structure.pdb"):
        # Create a new structure
        structure = Structure.Structure("filtered_structure")
        model = Model.Model(0)
        structure.add(model)

        current_chain = None
        current_residue = None
        for atom in filtered_atoms:
            chain_letter, residue_name, resSeq, atom_name, atom_type, atom_index, domain, secondary_structure, functional_site, interfaces, variant, custom_regions, coordinates = atom
            if current_chain is None or current_chain.id != chain_letter:
                current_chain = Chain.Chain(chain_letter)
                model.add(current_chain)

            if current_residue is None or current_residue.id[1] != resSeq:
                current_residue = Residue.Residue((' ', resSeq, ' '), residue_name, current_chain.id)
                current_chain.add(current_residue)

            if numpy_array is not None:
                numpy_array = np.reshape(numpy_array, (len(numpy_array), -1, 3))
                x, y, z = numpy_array[select_frame,atom_index,:]
            else:
                x, y, z = coordinates
            b_factor = b_factors[atom_index] if b_factors else 0.00
            new_atom = Atom.Atom(atom_name, np.array([x, y, z], dtype=float), 1.0, b_factor, ' ', atom_name, atom_index, atom_type)
            current_residue.add(new_atom)

        # Write the structure to a PDB file
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_pdb)

    def get_indices(self, atoms_info):
        return np.array([atom[5] for atom in atoms_info])



