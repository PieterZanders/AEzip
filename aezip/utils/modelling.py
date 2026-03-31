import numpy as np
from numpy import cos, pi, sin
from numpy.linalg import norm
from Bio.PDB.vectors import Vector, rotaxis
import multiprocessing as mp
from tqdm import tqdm

# Geometry
CBINTERNALS = [1.5, 115.5, -123.]
OINTERNALS = [1.229, 120.500, 0.000]
SP3ANGLE = 109.470
SP3DIHS = [60.0, 180.0, 300.0]
HDIS = 1.08

def build_coords_backbone_oxygen(at1c, at2c, atNc_next, geom):
    """
    Calculates Cartesian coordinates for the oxygen backbone atom (O) in the peptide bond.
    Parameters:
    - at1c: Cartesian coordinates of the carbonyl carbon (C)
    - at2c: Cartesian coordinates of the alpha carbon (CA)
    - atNc_next: Cartesian coordinates of the nitrogen of the next residue (N_next)
    - geom: [bond length, bond angle (degrees), torsion angle (degrees)]
    Returns:
    - o_coord: Cartesian coordinates of the oxygen atom
    """
    dst = geom[0] 
    ang = geom[1] * pi / 180.0  

    vec_ca_c = at2c - at1c
    vec_ca_c /= norm(vec_ca_c)

    vec_n_c = atNc_next - at1c
    vec_n_c /= norm(vec_n_c)

    normal = np.cross(vec_ca_c, vec_n_c)
    normal /= norm(normal)

    normal = -normal

    vec_perp = np.cross(normal, vec_ca_c)
    vec_perp /= norm(vec_perp)

    bond_vec = dst * (cos(ang) * vec_ca_c + sin(ang) * vec_perp)

    return at1c + bond_vec

def build_coords_from_internal(at1c, at2c, at3c, geom):
    """
     Calculates cartesian coordinates for a new atom from internal coordinates.
    """
    dst = geom[0]
    ang = geom[1] * pi / 180.
    tor = geom[2] * pi / 180.0

    vec1 = at1c - at2c
    vec2 = at1c - at3c

    vcr12 = np.cross(vec1, vec2)
    vcr112 = np.cross(vec1, vcr12)

    vcr12 /= norm(vcr12)
    vcr112 /= norm(vcr112)

    vcr12 *= -sin(tor)
    vcr112 *= cos(tor)

    vec3 = vcr12 + vcr112
    vec3 /= norm(vec3)
    vec3 *= dst * sin(ang)

    vec1 /= norm(vec1)
    vec1 *= dst * cos(ang)

    return at1c + vec3 - vec1

def reconstruct_CB(trajectory, topology, CBINTERNALS):
    """ Add CB to residue"""
    residues_list = list(topology.topology.residues)  

    for res in residues_list:
        c_c = ca_c = n_c = None

        if res.name != 'GLY':
            for at in res.atoms:
                if at.name == 'C':
                    c_c = trajectory[0, at.index, :]  
                if at.name == 'CA':
                    ca_c = trajectory[0, at.index, :]  
                if at.name == 'N':
                    n_c = trajectory[0, at.index, :]                  
                if at.name == 'CB':
                    cb_idx = at.index
            
            cb_c = build_coords_from_internal(ca_c, n_c, c_c, CBINTERNALS) 
            trajectory[:, cb_idx, :] = cb_c

    return trajectory

def reconstruct_backbone_oxygens(trajectory, topology, OINTERNALS):

    residues_list = list(topology.topology.residues)  

    for res in residues_list:
        c_c = ca_c = n_next_c = None
        o_idx = None
        
        for at in res.atoms:
            if at.name == 'C':
                c_c = trajectory[0, at.index, :] 
            if at.name == 'CA':
                ca_c = trajectory[0, at.index, :]  
            if at.name == 'O':
                o_idx = at.index                  
        
        if res.index < len(residues_list) - 1: 
            next_res = residues_list[res.index + 1]
            for next_at in next_res.atoms:
                if next_at.name == 'N':  
                    n_next_c = trajectory[0, next_at.index, :]
                    break
        else:
            if at.name == 'N':
                n_next_c = trajectory[0, at.index, :]

        if c_c is not None and ca_c is not None and n_next_c is not None and o_idx is not None:
            o_c = build_coords_backbone_oxygen(c_c, ca_c, n_next_c, OINTERNALS) 
            trajectory[:, o_idx, :] = o_c

    return trajectory

def build_coords_3xSP3(dst, at0, at1, at2):
    """ Generates coordinates for 3 SP3 atoms
        **dst** bond distance
        **at0**  central atom
        **at1** atom to define bond angles
        **at2** atom to define dihedrals
    """
    # TODO try a pure geometrical generation to avoid at2
    crs = []
    for i in range(0, 3):
        crs.append(
            build_coords_from_internal(
                at0, at1, at2, [dst, SP3ANGLE, SP3DIHS[i]]
            )
        )
    return crs

def build_coords_2xSP3(dst, cr0, cr1, cr2):
    """
        Generates coordinates for two SP3 bonds given the other two
        **dst** Bond distance
        **at0** Central atom
        **at1** atom with existing bond
        **at2** atom with existing bond
    """
    cr0 = Vector(cr0); cr1 = Vector(cr1); cr2 = Vector(cr2)
    axe = cr0 - cr1
    mat = rotaxis(120.*pi/180., axe)
    bond = cr2 - cr0
    bond.normalize()
    bond._ar = bond._ar * dst
    cr3 = cr0 + bond.left_multiply(mat)
    cr4 = cr0 + bond.left_multiply(mat).left_multiply(mat)
    crs = []
    crs.append(cr3._ar)
    crs.append(cr4._ar)
    return crs

def build_coords_1xSP3(dst, cr0, cr1, cr2, cr3):
    """ Calculate cartesian coordinates to complete a SP3 group
    """
    avg = cr1 + cr2
    avg = avg + cr3
    avg /= 3.
    avec = cr0 - avg
    avec /= norm(avec)
    avec *= dst
    return cr0 + avec

def build_coords_SP2(dst, cr0, cr1, cr2):
    """ Calculate cartesian coordinaties to complete a SP2 group
    """
    avg = cr1 + cr2
    avg /= 2.
    avec = cr0 - avg
    avec /= norm(avec)
    avec *= dst
    return cr0 + avec

def add_hydrogens_side(trajectory, topology_dict, reslib_dict, dat, res, atomname, hisname, frame):
    """ Add hydrogens to side chains"""

    rule = dat[atomname]['mode']
    ref_atoms = dat[atomname]['ref_ats']
    h_ats = dat[atomname]['ats']
    dist = dat[atomname]['dist']
    at0c = trajectory[frame, topology_dict.get(str(res)).get(str(atomname)), :]

    if rule == 'B2':

        at0_idx = topology_dict.get(str(res)).get(str(h_ats[0]))
        at1_idx = topology_dict.get(str(res)).get(str(h_ats[1]))

        crs = build_coords_2xSP3(
            dist,
            at0c,
            trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[0])), :],
            trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[1])), :]
        )
        trajectory[frame, at0_idx, :] = crs[0]
        trajectory[frame, at1_idx, :] = crs[1]

    elif rule == "B1":
        at0_idx = topology_dict.get(str(res))[(str(h_ats[0]))]
        trajectory[frame, at0_idx, :] = build_coords_1xSP3(
            dist,
            at0c,
            trajectory[frame, topology_dict[(str(res))][(str(ref_atoms[0]))], :],
            trajectory[frame, topology_dict[(str(res))][(str(ref_atoms[1]))], :],
            trajectory[frame, topology_dict[(str(res))][(str(ref_atoms[2]))], :]
        )

    elif rule == 'S2':
        at0_idx = topology_dict.get(str(res)).get(str(h_ats[0]))
        trajectory[frame, at0_idx, :] = build_coords_SP2(
            dist,
            at0c,
            trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[0])), :],
            trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[1])), :]
        )

    elif rule == 'L':
        if str(res.name) == 'HIS':
            res_type = hisname
        else:
            res_type = res.name
        for at_id in h_ats:
            at_idx = topology_dict.get(str(res)).get(str(at_id))
            trajectory[frame, at_idx, :] = build_coords_from_internal(
                trajectory[frame, topology_dict.get(str(res)).get(str(atomname)), :],
                trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[0])), :],
                trajectory[frame, topology_dict.get(str(res)).get(str(ref_atoms[1])), :],
                np.array(reslib_dict.get(str(res_type)).get(str(at_id))[0])
            )

    return trajectory

# ---------------------------------------------------------------------------
# Per-frame worker for reconstruct_trajectory
# ---------------------------------------------------------------------------

_RECONSTRUCT_CTX = {}  # populated once per worker process by the pool initializer


def _init_reconstruct_worker(residues_list, topology_dict, dihedral_definitions,
                              reslib_dict, datlib_dict, hist_dict):
    """Initialise worker-process globals so large dicts are pickled only once."""
    _RECONSTRUCT_CTX['residues_list']       = residues_list
    _RECONSTRUCT_CTX['topology_dict']       = topology_dict
    _RECONSTRUCT_CTX['dihedral_definitions'] = dihedral_definitions
    _RECONSTRUCT_CTX['reslib_dict']         = reslib_dict
    _RECONSTRUCT_CTX['datlib_dict']         = datlib_dict
    _RECONSTRUCT_CTX['hist_dict']           = hist_dict


def _reconstruct_single_frame(args):
    """Rebuild all heavy atoms and hydrogens for a single frame.

    Parameters
    ----------
    args : (frame_xyz, dihedral_frame)
        frame_xyz     – (n_atoms, 3) float64 array, pre-filled at sliced_indices
        dihedral_frame – dict returned by build_dih_traj for this frame

    Returns
    -------
    frame_xyz : (n_atoms, 3) array with all atoms placed
    """
    frame_xyz, dihedral_frame = args

    residues_list       = _RECONSTRUCT_CTX['residues_list']
    topology_dict       = _RECONSTRUCT_CTX['topology_dict']
    dihedral_definitions = _RECONSTRUCT_CTX['dihedral_definitions']
    reslib_dict         = _RECONSTRUCT_CTX['reslib_dict']
    datlib_dict         = _RECONSTRUCT_CTX['datlib_dict']
    hist_dict           = _RECONSTRUCT_CTX['hist_dict']

    sc_internals = None   # may be referenced before assignment in the 'else' branch
    n_next_c     = None   # set when the 'O' atom is processed

    # --- Heavy atoms + backbone hydrogens ---
    for i, res in enumerate(residues_list):
        next_res = residues_list[i + 1] if i + 1 < len(residues_list) else None
        prev_res = residues_list[i - 1] if i - 1 >= 0 else None

        n_c  = frame_xyz[topology_dict.get(str(res)).get('N'), :]
        ca_c = frame_xyz[topology_dict.get(str(res)).get('CA'), :]
        c_c  = frame_xyz[topology_dict.get(str(res)).get('C'), :]

        for atom in res.atoms:
            if np.all(frame_xyz[atom.index, :] == [0., 0., 0.], axis=(0)):
                if str(atom.name) == 'CB':
                    frame_xyz[atom.index, :] = build_coords_from_internal(ca_c, n_c, c_c, CBINTERNALS)

                elif str(atom.name) == 'O' and next_res is not None:
                    o_internals = np.array(reslib_dict.get(str(res.name)).get(str(atom.name))[0])
                    n_next_c = frame_xyz[topology_dict.get(str(next_res)).get("N"), :]
                    frame_xyz[atom.index, :] = build_coords_backbone_oxygen(c_c, ca_c, n_next_c, o_internals)

                elif str(atom.name) in dihedral_definitions[res.name].keys():
                    if str(res.name) == 'PRO':
                        frame_xyz[topology_dict.get(str(res)).get('CB'), :] = build_coords_from_internal(ca_c, n_c, c_c, CBINTERNALS)
                        sc_internals = np.array(reslib_dict.get(str(res.name)).get('CG')[0])
                        sc_internals[-1] = dihedral_frame.get(str(res)).get('CG') + 180
                        at1c = frame_xyz[topology_dict.get(str(res)).get('CB'), :]
                        frame_xyz[topology_dict.get(str(res)).get('CG'), :] = build_coords_from_internal(n_c, ca_c, c_c, sc_internals)
                        sc_internals = np.array(reslib_dict.get(str(res.name))['CD'][0])
                        sc_internals[-1] = dihedral_frame.get(str(res))['CD']
                        frame_xyz[topology_dict.get(str(res)).get('CD'), :] = build_coords_from_internal(at1c, ca_c, n_c, sc_internals)
                    else:
                        sc_internals = np.array(reslib_dict.get(str(res.name)).get(str(atom.name))[0])
                        sc_internals[-1] = dihedral_frame.get(str(res)).get(str(atom.name))
                        at1n, at2n, at3n = list(dihedral_definitions[str(res.name)][str(atom.name)])
                        at1c, at2c, at3c = [frame_xyz[topology_dict[str(res)][a], :] for a in (at1n, at2n, at3n)]
                        frame_xyz[atom.index, :] = build_coords_from_internal(at1c, at2c, at3c, sc_internals)

                else:
                    if str(atom.element) != 'hydrogen':
                        resname = res.name if str(atom.name) != 'OXT' else 'C' + res.name
                        if str(res.name) in ['TRP', 'PHE', 'TYR', 'HIS']:
                            res_internals = np.array(reslib_dict.get(str(resname)).get(str(atom.name))[0])
                            atom_indices  = reslib_dict[str(resname)][str(atom.name)][1]
                            at1c, at2c, at3c = [frame_xyz[topology_dict[str(res)][idx], :] for idx in atom_indices]
                            frame_xyz[atom.index, :] = build_coords_from_internal(at1c, at2c, at3c, res_internals)
                        elif str(res.name) in ['ARG', 'GLN', 'ASN', 'GLU', 'ASP']:
                            sc_internals[-1] += 180
                            atom_indices = reslib_dict[str(resname)][str(atom.name)][1]
                            at1c, at2c, at3c = [frame_xyz[topology_dict[str(res)][idx], :] for idx in atom_indices]
                            frame_xyz[atom.index, :] = build_coords_from_internal(at1c, at2c, at3c, sc_internals)
                        elif str(res.name) in ['LEU', 'ILE', 'VAL']:
                            sc_internals[-1] += 120
                            atom_indices = reslib_dict[str(resname)][str(atom.name)][1]
                            at1c, at2c, at3c = [frame_xyz[topology_dict[str(res)][idx], :] for idx in atom_indices]
                            frame_xyz[atom.index, :] = build_coords_from_internal(at1c, at2c, at3c, sc_internals)
                        elif str(res.name) in ['THR']:
                            sc_internals[-1] -= 120
                            atom_indices = reslib_dict[str(resname)][str(atom.name)][1]
                            at1c, at2c, at3c = [frame_xyz[topology_dict[str(res)][idx], :] for idx in atom_indices]
                            frame_xyz[atom.index, :] = build_coords_from_internal(at1c, at2c, at3c, sc_internals)
                        elif str(res.name) in ['ILE']:
                            sc_internals = np.array(reslib_dict.get(str(res.name)).get('CG2')[0])
                            sc_internals[-1] = dihedral_frame.get(str(res)).get('CG1') + 180
                            at1n, at2n, at3n = list(dihedral_definitions[str(res.name)]['CG1'])
                            c_c, ca_c, n_c = [frame_xyz[topology_dict[str(res)][a], :] for a in (at1n, at2n, at3n)]
                            frame_xyz[atom.index, :] = build_coords_from_internal(c_c, ca_c, n_c, sc_internals)

        # Backbone hydrogens
        if prev_res is None:
            if res.name == 'ACE':
                crs = build_coords_3xSP3(HDIS, ca_c, n_next_c, c_c)
                frame_xyz[topology_dict.get(str(res)).get('HA1'), :] = crs[0]
                frame_xyz[topology_dict.get(str(res)).get('HA2'), :] = crs[1]
                frame_xyz[topology_dict.get(str(res)).get('HA3'), :] = crs[2]
            else:
                crs = build_coords_3xSP3(HDIS, n_c, ca_c, c_c)
                frame_xyz[topology_dict.get(str(res)).get('H'),  :] = crs[0]
                frame_xyz[topology_dict.get(str(res)).get('H2'), :] = crs[1]
                frame_xyz[topology_dict.get(str(res)).get('H3'), :] = crs[2]
        elif res.name != 'PRO':
            c_prev_c = frame_xyz[topology_dict.get(str(prev_res)).get('C'), :]
            frame_xyz[topology_dict.get(str(res)).get('H'), :] = build_coords_SP2(HDIS, n_c, ca_c, c_prev_c)

        if res.name in ['GLY', 'NGLY']:
            crs = build_coords_2xSP3(HDIS, ca_c, n_c, c_c)
            frame_xyz[topology_dict.get(str(res)).get('HA2'), :] = crs[0]
            frame_xyz[topology_dict.get(str(res)).get('HA3'), :] = crs[1]
        elif res.name == 'NME':
            c_prev_c = frame_xyz[topology_dict.get(str(prev_res)).get('C'), :]
            crs = build_coords_3xSP3(HDIS, ca_c, n_c, c_prev_c)
            frame_xyz[topology_dict.get(str(res)).get('HA1'), :] = crs[0]
            frame_xyz[topology_dict.get(str(res)).get('HA2'), :] = crs[1]
            frame_xyz[topology_dict.get(str(res)).get('HA3'), :] = crs[2]
        else:
            frame_xyz[topology_dict.get(str(res)).get('HA'), :] = build_coords_1xSP3(
                HDIS, ca_c, n_c, c_c, build_coords_from_internal(ca_c, n_c, c_c, CBINTERNALS)
            )

    # --- Sidechain hydrogens ---
    # add_hydrogens_side expects a 3-D array and a frame index;
    # we wrap the 2-D slice as shape (1, n_atoms, 3) and use frame=0.
    frame_xyz_3d = frame_xyz[np.newaxis, :]
    for i, res in enumerate(residues_list):
        for atom in res.atoms:
            if str(res.name) != 'GLY':
                if isinstance(datlib_dict[str(res.name)].get('hydrogen_atoms'), dict):
                    if res.index in list(hist_dict.keys()):
                        hdat = datlib_dict[str(res.name)]['addH_rules'][hist_dict[int(res.index)]]
                        if hdat.get(str(atom.name)):
                            frame_xyz_3d = add_hydrogens_side(frame_xyz_3d, topology_dict, reslib_dict,
                                                              hdat, res, atom.name, hist_dict[int(res.index)], 0)
                    else:
                        hdat = datlib_dict[str(res.name)]['addH_rules'][res.name]
                        if hdat.get(str(atom.name)):
                            frame_xyz_3d = add_hydrogens_side(frame_xyz_3d, topology_dict, reslib_dict,
                                                              hdat, res, atom.name, None, 0)
                else:
                    if str(atom.name) in datlib_dict[str(res.name)]['addH_rules']:
                        hdat = datlib_dict[str(res.name)]['addH_rules']
                        if hdat.get(str(atom.name)):
                            frame_xyz_3d = add_hydrogens_side(frame_xyz_3d, topology_dict, reslib_dict,
                                                              hdat, res, atom.name, None, 0)

    return frame_xyz_3d[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_trajectory(topology, sliced_traj, sliced_indices, dihedral_traj, dihedral_definitions,
                           topology_dict, reslib_dict, datlib_dict, hist_dict):

    n_frames      = sliced_traj.n_frames
    n_atoms       = topology.n_atoms
    residues_list = list(topology.topology.residues)

    # Pre-fill backbone / sliced atoms for all frames
    trajectory = np.zeros((n_frames, n_atoms, 3))
    trajectory[:, sliced_indices, :] = sliced_traj.xyz * 10

    n_workers = mp.cpu_count()
    chunksize = max(1, n_frames // (n_workers * 4))

    task_args = [(trajectory[f].copy(), dihedral_traj[f]) for f in range(n_frames)]

    with mp.Pool(
        processes=n_workers,
        initializer=_init_reconstruct_worker,
        initargs=(residues_list, topology_dict, dihedral_definitions,
                  reslib_dict, datlib_dict, hist_dict),
    ) as pool:
        results = list(tqdm(
            pool.imap(_reconstruct_single_frame, task_args, chunksize=chunksize),
            total=n_frames,
            desc="Reconstructing frames",
            unit="frame",
        ))

    for f, frame_xyz in enumerate(results):
        trajectory[f] = frame_xyz

    return trajectory
