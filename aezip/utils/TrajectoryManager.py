import os, sys
import numpy as np
import inspect
import mdtraj as md
from typing import Any, Dict, List, Optional, Union

class TrajectoryManager:
    def __init__(self, traj_file: str, top_file: str):
        self.top_file = top_file
        self.traj = md.load(traj_file, top=top_file)
    def get_trajectory(self) -> md.Trajectory:
        return self.traj
    def get_topology(self) -> md.Topology:
        return self.traj.topology
    def get_xyz(self) -> np.ndarray:
        return self.traj.xyz
    def traj_sampling(self, traj: md.Trajectory, frame_interval: int, frame_idx: Optional[np.ndarray]=None) -> md.Trajectory:
        if frame_idx is not None:
            return md.Trajectory(traj.xyz[frame_idx], topology=traj.topology)
        else:
            return md.Trajectory(traj.xyz[::frame_interval], topology=traj.topology)
    def traj_atom_subset(self, traj: md.Trajectory, atom_idx: np.ndarray) -> md.Trajectory:
        return md.Trajectory(traj.xyz[:, atom_idx, :], topology=traj.topology.subset(atom_idx))
    def numpy2mdtraj(self, numpy_array: np.ndarray, top_file: Optional[Union[str, md.Topology]]=None) -> md.Trajectory:
        if top_file is None:
            return md.Trajectory(numpy_array, topology=self.get_topology())
        else:
            if isinstance(top_file, str):
                return md.Trajectory(numpy_array, topology=md.load(top_file).topology)
            if isinstance(top_file, md.Topology):
                return md.Trajectory(numpy_array, topology=top_file)

    def traj_join(self, other_traj_file: Union[List[str], str]) -> md.Trajectory:
        if isinstance(other_traj_file, str):
            other_traj_file = [other_traj_file]
        conc_traj = self.traj
        for traj_file in other_traj_file:
            other_traj = md.load(traj_file, top=self.top_file)
            conc_traj = md.join(conc_traj, other_traj)
        return conc_traj
    def traj_fitting(self, traj: md.Trajectory, reference: Optional[md.Trajectory]=None, frame: int=0, atom_indices: Optional[np.ndarray]=None, ref_atom_indices: Optional[np.ndarray]=None) -> md.Trajectory:
        if traj is not None:
            self.traj = traj
        if reference is not None:
            return self.traj.superpose(reference, frame=frame, atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)
        else:
            return self.traj.superpose(self.traj, frame=frame, atom_indices=atom_indices, ref_atom_indices=ref_atom_indices)
    def save_traj(self, traj: md.Trajectory, output_format: str, output_file: str) -> None:
        if output_format == "xtc":
            traj.save_xtc(output_file)
        elif output_format == "pdb":
            traj.save_pdb(output_file)
        elif output_format == "trr":
            traj.save_trr(output_file)
        elif output_format == "dcd":
            traj.save_dcd(output_file)
        elif output_format == "tng":
            traj.save_tng(output_file)
        elif output_format == "xyz":
            traj.save_xyz(output_file)
        else:
            print("Unsupported output format. Supported formats are: xtc, pdb, dcd.")
    def normalize_minmax_traj(self, traj: md.Trajectory, maximums: Optional[np.ndarray]=None, minimums: Optional[np.ndarray]=None) -> tuple:
        data = np.reshape(traj.xyz, (len(traj.xyz), -1))
        if maximums is None:
           maximums = np.max(data, axis=0)
        if minimums is None:
           minimums = np.min(data, axis=0)
        norm_data = (data - minimums) / (maximums - minimums)
        norm_traj = self.numpy2mdtraj(np.reshape(norm_data, (len(norm_data), -1, 3)), traj.topology)
        return norm_traj, maximums, minimums
    def denormalize_traj(self, traj: Union[md.Trajectory, np.ndarray], minimums: np.ndarray, maximums: np.ndarray) -> md.Trajectory:
        if isinstance(traj, md.Trajectory):
            data = traj.xyz.reshape(len(traj.xyz), -1)
            denorm_data = data * (maximums - minimums) + minimums
            denorm_traj = denorm_data.reshape(len(traj.xyz), -1, 3)
            return self.numpy2mdtraj(denorm_traj, traj.topology)
        elif isinstance(traj, np.ndarray):
            data = traj.reshape(len(traj), -1)
            denorm_data = data * (maximums - minimums) + minimums
            denorm_traj = self.numpy2mdtraj(denorm_data, self.get_topology())
            return denorm_traj
    def data_normalization(self, data_array):
        maximums = np.max(data_array, axis=0)
        minimums = np.min(data_array, axis=0)
        # Normalize the data_array
        norm_array = (data_array - minimums) / (maximums - minimums)
        return norm_array, maximums, minimums
    def data_denormalization(self, normalized_data, max_values, min_values):
        # Denormalize the data
        return normalized_data * (max_values - min_values) + min_values
    def print_method_annotations():
        print(f"Annotations for class: {TrajectoryManager.__name__}")
        for name, method in TrajectoryManager.__dict__.items():
            if not name.startswith('_') and callable(method):
                print(f"\n  {name}:")
                annotations = method.__annotations__
                #print(f"  Annotations: {annotations}")
                for param, param_type in annotations.items():
                    if param != 'return':
                      print(f"    · {param}: {param_type}")
                    else:
                      print(f"    {param}:  {param_type}")
