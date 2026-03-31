import os, sys
import numpy as np
import inspect
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from numpy.linalg import det
from itertools import combinations
# import deeptime
import mdtraj as md
from typing import Any, Dict, List, Optional, Union

class TrajectoryAnalysis:
    @staticmethod
    def calculate_mean_conformation(traj):
        """
        Calculate the mean conformation of a trajectory.

        Args:
            traj (md.Trajectory): The trajectory object.

        Returns:
            np.ndarray: The mean conformation of the trajectory.
        """
        return np.mean(traj.xyz, axis=0)
    @staticmethod
    def calculate_rmsd(target, reference, frame=0, atom_indices=None, precentered=False):
        """
        Calculate the root-mean-square deviation (RMSD) between two trajectories.

        Args:
            target (md.Trajectory): The target trajectory.
            reference (md.Trajectory): The reference trajectory.
            frame (int, optional): The frame index. Defaults to 0.
            atom_indices (np.ndarray, optional): The atom indices. Defaults to None.
            precentered (bool, optional): Whether to precenter the trajectories. Defaults to False.

        Returns:
            np.ndarray: RMSD values for each frame.

        """
        return md.rmsd(target, reference, frame=frame, atom_indices=atom_indices, precentered=precentered)
    @staticmethod
    def calculate_rmsf(target: md.Trajectory, reference: md.Trajectory, frame: int=0, atom_indices: np.ndarray=None, precentered: bool=False):
        """
        Calculate the root-mean-square deviation (RMSF) between two trajectories.

        Args:
            target (md.Trajectory): The target trajectory.
            reference (md.Trajectory): The reference trajectory.
            frame (int, optional): The frame index. Defaults to 0.
            atom_indices (np.ndarray, optional): The atom indices. Defaults to None.
            precentered (bool, optional): Whether to precenter the trajectories. Defaults to False.

        Returns:
            np.ndarray: RMSF values for each atom.

        """
        return md.rmsf(target, reference, frame=frame, atom_indices=atom_indices, precentered=precentered)
        
    @staticmethod
    def fluctuation_traj(traj: md.Trajectory):
        """
        Calculate the fluctuation of a trajectory.

        Args:
            traj (md.Trajectory): The trajectory object.

        Returns:
            np.ndarray: The fluctuation respect to the mean conformation of the trajectory.
        """
        mean_traj = np.mean(traj.xyz, axis=0)
        return traj.xyz - mean_traj, mean_traj

    @staticmethod
    def calculate_rmsf_eigenvectors(eigenvectors, eigenvalues, atom_mass):
        n_atoms = eigenvectors.shape[1] // 3
        nmodes = len(eigenvalues)
        y = np.zeros((nmodes, n_atoms))
        masses = atom_mass
        for g in range(nmodes):
            for i in range(n_atoms):
                norm2 = np.sum(eigenvectors[g, i*3:(i+1)*3]**2)
                eigval_w = eigenvalues[g] * norm2
                if eigval_w < 0:
                    amplitude = 0.0
                else: 
                    amplitude = np.sqrt(eigval_w) / np.sqrt(3*masses[i])
                y[g, i] = amplitude
        if eigenvectors.shape[0] != eigenvectors.shape[1]:
            y = np.delete(y, np.arange(abs(eigenvectors.shape[0] - eigenvectors.shape[1])), axis=0)
        return y

    @staticmethod
    def PCA(self, traj: md.Trajectory, mass_weighting=True, n_components=3):
        """
        Perform PCA on molecular dynamics trajectory data.

        Parameters:
        traj (md.Trajectory): Trajectory as MDTraj Object.
        mass_weighting (bool, optional): If True, applies mass-weighting to the PCA. Default is True.
        n_components (int, optional): Number of principal components to project. Default is 3.

        Returns:
        tuple: Eigenvalues, eigenvectors, PCA projection of the trajectory, RMSF of eigenvectors, and covariance matrix.
        """

        # Get the number of atoms
        n_atoms = traj.n_atoms

        # Extract the coordinates
        coordinates = traj.xyz

        # Center the coordinates by subtracting the mean position (over the trajectory)
        mean_position = np.mean(coordinates, axis=0)
        centered_coordinates = coordinates - mean_position

        # Construct the mass matrix M (if mass-weighting is desired)
        if mass_weighting:
            masses = np.repeat(12, 3*n_atoms)
            mass_matrix = np.diag(masses)
        else:
            mass_matrix = np.eye(3*n_atoms)

        # Flatten the centered coordinates
        flattened_coordinates = centered_coordinates.reshape(len(coordinates), -1)

        # Compute the covariance matrix C with mass weighting
        cov_matrix = np.cov(flattened_coordinates, rowvar=False)
        weighted_cov_matrix = np.dot(np.dot(mass_matrix**0.5, cov_matrix), mass_matrix**0.5)

        # Diagonalize the covariance matrix to get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].T

        # Project the trajectory onto the selected principal components
        pca_projection = np.dot(eigenvectors[:n_components, :], flattened_coordinates.T)

        # Calculate RMSF of the eigenvectors
        rmsf_eigvec = self.calculate_rmsf_eigenvectors(eigenvectors, eigenvalues, masses)

        return eigenvalues, eigenvectors, pca_projection, rmsf_eigvec, cov_matrix
        
    # @staticmethod
    # def TICA(traj: md.Trajectory, time_lag: int):
    #     """
    #     Perform time-independent collective variable analysis (TICA) on a trajectory.

    #     Args:
    #         traj (md.Trajectory): The trajectory object.
    #         time_lag (int): The time lag.

    #     Returns:
    #         deeptime.decomposition.TICA: The TICA model.
    #         np.ndarray: The TICA transformed trajectory.
    #     """
    #     tica = deeptime.decomposition.TICA(lagtime=time_lag)
    #     tica_model = tica.fit(traj.xyz.reshape(len(traj.xyz), -1), lagtime=time_lag).fetch_model()
    #     tica_transformed = tica_model.transform(traj.xyz)
    #     return tica_model, tica_transformed

    @staticmethod
    def calculate_distances(traj: md.Trajectory, options: str, **kwargs):
        """
        Calculate distances or related metrics based on the specified option.

        Args:
            traj (md.Trajectory): The trajectory to perform calculations on.
            options (str): The type of calculation to perform. Supported options are:
                - "distances": Computes distances between atom pairs.
                    Parameters:
                    - atom_pairs (List[int]): List of atom pairs.
                    - periodic (bool): Whether to use periodic boundary conditions.
                    - opt (bool): Optimization flag.
                - "squareform": Converts a contacts matrix into a square form.
                    Parameters:
                    - contacts (Any): Contacts information.
                    - periodic (bool): Whether to use periodic boundary conditions.
                    - scheme (str): Scheme to use for calculation.
                - "displacements": Computes displacements between atom pairs.
                    Parameters:
                    - atom_pairs (List[int]): List of atom pairs.
                    - periodic (bool): Whether to use periodic boundary conditions.
                    - opt (bool): Optimization flag.
                - "neighbors": Computes neighbors within a cutoff distance.
                    Parameters:
                    - cutoff (float): Cutoff distance.
                    - query_indices (List[int]): Query indices.
                    - haystack_indices (Optional[List[int]]): Haystack indices.
                    - periodic (bool): Whether to use periodic boundary conditions.
                - "contacts": Computes contacts between atoms.
                    Parameters:
                    - contacts (str): Type of contacts ('all' or specific).
                    - scheme (str): Scheme to use for calculation.
                    - ignore_nonprotein (bool): Whether to ignore non-protein atoms.
                    - periodic (bool): Whether to use periodic boundary conditions.
                    - soft_min (bool): Soft minimum flag.
                    - soft_min_beta (float): Soft minimum beta value.
                - "drid": Computes the DRID (distance root-mean-square deviation).
                    Parameters:
                    - atom_indices (Optional[List[int]]): Atom indices.
                - "rdf": Computes the radial distribution function.
                    Parameters:
                    - pairs (List[int]): Pairs of atoms.
                    - r_range (Optional[List[float]]): Range of r values.
                    - bin_width (float): Bin width.
                    - n_bins (Optional[int]): Number of bins.
                    - periodic (bool): Whether to use periodic boundary conditions.
                    - opt (bool): Optimization flag.
                - "center_of_mass": Computes the center of mass of the trajectory.
                    No additional parameters.

        Returns:
            Any: Result of the specified calculation.
        """
        if options == "distances":
            atom_pairs = kwargs.get('atom_pairs', None)
            return md.compute_distances(traj, atom_pairs=atom_pairs, periodic=kwargs.get('periodic', True), opt=kwargs.get('opt', True))
        elif options == "squareform":
            contacts = kwargs.get('contacts', None)
            return md.geometry.squareform(traj, contacts, periodic=kwargs.get('periodic', True), scheme=kwargs.get('scheme', "closest-heavy"))
        elif options == "displacements":
            atom_pairs = kwargs.get('atom_pairs', None)
            return md.compute_displacements(traj, atom_pairs, periodic=kwargs.get('periodic', True), opt=kwargs.get('opt', True))
        elif options == "neighbors":
            cutoff = kwargs.get('cutoff', None)
            query_indices = kwargs.get('query_indices', None)
            haystack_indices = kwargs.get('haystack_indices', None)
            return md.compute_neighbors(traj, cutoff, query_indices, haystack_indices=haystack_indices, periodic=kwargs.get('periodic', True))
        elif options == "contacts":
            contacts = kwargs.get('contacts', 'all')
            scheme = kwargs.get('scheme', 'closest-heavy')
            ignore_nonprotein = kwargs.get('ignore_nonprotein', True)
            soft_min = kwargs.get('soft_min', False)
            soft_min_beta = kwargs.get('soft_min_beta', 20)
            return md.compute_contacts(traj, contacts=contacts, scheme=scheme, ignore_nonprotein=ignore_nonprotein, periodic=kwargs.get('periodic', True), soft_min=soft_min, soft_min_beta=soft_min_beta)
        elif options == "drid":
            atom_indices = kwargs.get('atom_indices', None)
            return md.compute_drid(traj, atom_indices=atom_indices)
        elif options == "rdf":
            pairs = kwargs.get('pairs', None)
            r_range = kwargs.get('r_range', None)
            bin_width = kwargs.get('bin_width', 0.005)
            n_bins = kwargs.get('n_bins', None)
            return md.compute_rdf(traj, pairs, r_range=r_range, bin_width=bin_width, n_bins=n_bins, periodic=kwargs.get('periodic', True), opt=kwargs.get('opt', True))
        elif options == "center_of_mass":
            return md.compute_center_of_mass(traj)
        else:
            raise ValueError("Invalid options. Supported options are: distances, squareform, displacements, neighbors, contacts, drid, rdf, center_of_mass.")

    @staticmethod
    def calculate_angles(traj: md.Trajectory, options: str = 'dihedrals', **kwargs) -> np.ndarray:
        """
        Calculate various angles and dihedrals from the trajectory.

        Args:
            traj (md.Trajectory): The trajectory object.
            options (str): Calculation type. Supported options are:
                - "angles": Computes bond angles.
                - "dihedrals": Computes dihedral angles.
                - "phi": Computes the phi dihedral angles.
                - "psi": Computes the psi dihedral angles.
                - "omega": Computes the omega dihedral angles.
                - "chi1": Computes the chi1 dihedral angles.
                - "chi2": Computes the chi2 dihedral angles.
                - "chi3": Computes the chi3 dihedral angles.
                - "chi4": Computes the chi4 dihedral angles.

        Returns:
            np.ndarray: Array of calculated angles or dihedrals.
        """
        if options == 'angles':
            angle_indices = kwargs.get('angle_indices', [[0,1,2]])
            return md.compute_angles(traj, angle_indices=angle_indices)
        elif options == 'dihedrals':
            return md.compute_dihedrals(traj)
        elif options == 'phi':
            return md.compute_phi(traj)
        elif options == 'psi':
            return md.compute_psi(traj)
        elif options == 'omega':
            return md.compute_omega(traj)
        elif options == 'chi1':
            return md.compute_chi1(traj)
        elif options == 'chi2':
            return md.compute_chi2(traj)
        elif options == 'chi3':
            return md.compute_chi3(traj)
        elif options == 'chi4':
            return md.compute_chi4(traj)
        else:
            raise ValueError("Invalid options. Supported options are: angles, dihedrals, phi, psi, omega, chi1, chi2, chi3, chi4.")

    @staticmethod
    def hydrogen_bonding(traj, options="baker_hubbard"):
        """
        Perform hydrogen bonding analysis on the trajectory.

        Args:
            traj (md.Trajectory): The trajectory object.
            options (str): Calculation type. Supported options are:
                - "baker_hubbard": Baker-Hubbard method.
                - "kabsch_sander": Kabsch-Sander method.
                - "wernet_nilsson": Werner-Nilsson method.

        Returns:
            np.ndarray: Array of calculated Hydrogen Bonding.
        """
        if options == "baker_hubbard":
            return md.baker_hubbard(traj, freq=0.1, exclude_water=True, periodic=True, sidechain_only=False, distance_cutoff=0.25, angle_cutoff=120)
        elif options == "kabsch_sander":
            return md.kabsch_sander(traj)
        elif options == "wernet_nilsson":
            return md.wernet_nilsson(traj, exclude_water=True, periodic=True, sidechain_only=False)

    @staticmethod
    def dccm(self, traj, method="pearson"):
        """
        Calculate Distance Correlation Matrices (DCCM) based on the specified method.

        Args:
            traj (md.Trajectory): The trajectory object.
            method (str): Calculation type. Supported options are:
                - "pearson": Pearson correlation coefficient.
                - "lmi": LMI correlation coefficient.
                - "mi": Mutual information correlation coefficient.
                - "cov": Covariance correlation coefficient.

        Returns:
            np.ndarray: Array of calculated DCCM.
        """
        if method == "pearson":
            return self.dccm_pearson(traj)
        elif method == "lmi":
            return self.dccm_lmi(traj)
        elif method == "mi":
            return self.dccm_mi(traj)
        elif method == "cov":
            return self.dccm_cov(traj)
        else:
            raise ValueError("Invalid method. Supported methods are: pearson, lmi, mi, cov.")

    @staticmethod
    def dccm_pearson(traj):
        if isinstance(traj, md.Trajectory):
            xyz = traj.xyz

        reference = np.mean(xyz, axis=0)
        dxyz = xyz - reference
        dxyz_flat = dxyz.reshape(dxyz.shape[0], -1)
        covmat = np.cov(dxyz_flat, rowvar=False)

        mxyz = np.mean(dxyz, axis=0).flatten()
        covmat += np.outer(mxyz, mxyz)

        ccmat = TrajectoryAnalysis.cov2dccm_pearson(covmat)
        return ccmat

    @staticmethod
    def cov2dccm_pearson(covmat):
        n = covmat.shape[0] // 3
        ccmat = np.zeros((n, n))
        d = np.sqrt(np.diag(covmat).reshape(-1, 3).sum(axis=1))
        for i in range(n):
            for j in range(i, n):
                i1, i2 = i * 3, j * 3
                sub_matrix = covmat[i1:i1+3, i2:i2+3]
                ccmat[i, j] = np.sum(np.diag(sub_matrix)) / (d[i] * d[j])
        ccmat += ccmat.T - np.diag(np.diag(ccmat))
        np.fill_diagonal(ccmat, 1)
        return ccmat

    @staticmethod
    def dccm_lmi(traj):
        if isinstance(traj, md.Trajectory):
            xyz = traj.xyz
        reference = np.mean(xyz, axis=0)
        dxyz = xyz - reference
        dxyz_reshaped = dxyz.reshape(len(xyz), -1)
        cov_matrix = np.cov(dxyz_reshaped, rowvar=False)
        n_atoms = cov_matrix.shape[0] // 3
        lmi_matrix = np.zeros((n_atoms, n_atoms))
        marginals = [det(cov_matrix[i*3:i*3+3, i*3:i*3+3]) for i in range(n_atoms)]
        for i, j in combinations(range(n_atoms), 2):
            idx_i = slice(3*i, 3*i+3)
            idx_j = slice(3*j, 3*j+3)
            sub_matrix = cov_matrix[np.r_[idx_i, idx_j]][:, np.r_[idx_i, idx_j]]
            joint_determinant = det(sub_matrix)
            lmi_value = 0.5 * (np.log(marginals[i]) + np.log(marginals[j]) - np.log(joint_determinant))
            lmi_value = np.sqrt(1 - np.exp(-2 * lmi_value / 3))
            lmi_matrix[i, j] = lmi_matrix[j, i] = lmi_value
        np.fill_diagonal(lmi_matrix, 1)
        return lmi_matrix

    @staticmethod
    def dccm_cov(traj):
        if isinstance(traj, md.Trajectory):
            trajectory = traj.xyz
        frames, atoms, _ = trajectory.shape
        flattened_trajectory = trajectory.reshape(frames, atoms * 3)
        mean_positions = np.mean(flattened_trajectory, axis=0)
        centered_trajectory = flattened_trajectory - mean_positions
        cov_matrix = np.cov(centered_trajectory, rowvar=False)
        cov_matrix_reshaped = cov_matrix.reshape(atoms, 3, atoms, 3)
        cov_matrix_atoms = np.zeros((atoms, atoms))
        for i in range(atoms):
            for j in range(atoms):
                cov_matrix_atoms[i, j] = np.sum(cov_matrix_reshaped[i, :, j, :])
        return cov_matrix_atoms

    @staticmethod
    def dccm_mi(traj):
        if isinstance(traj, md.Trajectory):
            xyz = traj.xyz
        reference = np.mean(xyz, axis=0)
        dxyz = xyz - reference
        dxyz_reshaped = dxyz.reshape(len(xyz), -1)
        n_atoms = dxyz_reshaped.shape[1] // 3
        mi_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    mi_values = []
                    for k in range(3):  # Loop over x, y, z coordinates
                        x = dxyz_reshaped[:, 3*i + k].reshape(-1, 1)
                        y = dxyz_reshaped[:, 3*j + k]
                        mi_value = mutual_info_regression(x, y, discrete_features='auto')[0]
                        mi_values.append(mi_value)
                    mi_matrix[i, j] = np.mean(mi_values)
        np.fill_diagonal(mi_matrix, 1)
        return mi_matrix

    @staticmethod
    def print_method_documentation():
        print(f"Annotations for class: {TrajectoryAnalysis.__name__}")
        for name, method in TrajectoryAnalysis.__dict__.items():
            if callable(method):
                annotations = method.__doc__
                if annotations is not None:
                   print(f"\n  {name}:")
                   print(f"    {annotations}")
