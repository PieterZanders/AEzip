#!/usr/bin/env python

'''
To display the usage instructions, run:
analysis.py -h
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

###################
## Training Plot ##
###################

def train_valid_plot(train_loss, save_plot):
    # Finding the minimum value points for train and validation losses
    min_train_loss_value = min(train_loss)
    min_train_loss_idx = np.argmin(train_loss)

    # Plot
    plt.plot(range(len(train_loss)), train_loss, label=f"Training (min.: {min_train_loss_idx})", color='blue')

    # Adding triangle markers at the minimum points
    plt.scatter(min_train_loss_idx, train_loss[min_train_loss_idx], color='blue', marker='v', s=50)

    # Setting the rest of the plot elements
    plt.legend()
    plt.ylabel('Total Loss')
    plt.xlabel('Epochs')
    plt.title('Training')
    plt.show()
    plt.savefig(save_plot)

##########
## RMSD ##
##########

def calculate_rmsd(reference, trajectory):
    diff = reference - trajectory
    squared_diff = np.sum(diff ** 2, axis=2)
    mean_squared_diff = np.mean(squared_diff, axis=1)
    rmsd = np.sqrt(mean_squared_diff)
    return rmsd

def RMSD_plot(trajectory_data, output_data, save_plot):
    # Create the reference trajectory
    reference = trajectory_data[0]

    # Compute RMSD for both datasets
    rmsd_trajectory = calculate_rmsd(reference, trajectory_data)
    rmsd_output = calculate_rmsd(reference, output_data)

    # Calculate mean RMSD values
    mean_rmsd_trajectory = np.mean(rmsd_trajectory)
    mean_rmsd_output = np.mean(rmsd_output)

    # Generate time array (assuming 0.5 ns per frame)
    time = np.arange(len(rmsd_trajectory))

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 6))

    # Plot RMSD over time for trajectory_data
    ax.plot(time, rmsd_trajectory, color='blue', linewidth=1, label=f'Original (Mean RMSD: {mean_rmsd_trajectory:.2f} Å)')

    # Plot RMSD over time for output_data
    ax.plot(time, rmsd_output, color='red', linewidth=1, label=f'Reconstruction (Mean RMSD: {mean_rmsd_output:.2f} Å)')

    # Labels, title, and legend
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (Å)')
    plt.title('RMSD Plot')
    plt.legend()

    # Show and save the plot
    plt.show()
    plt.savefig(save_plot)

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 6))

    save_plot = save_plot.replace("rmsd", "dio_rmsd")

    # Plot RMSF for DIO
    dio = abs(rmsd_trajectory - rmsd_output)
    print("Mean DIO: ", np.nanmean(dio))
    ax.plot(time, dio, color='orange', linewidth=1, label='DIO')
    #ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1)

    # Labels, title, and legend
    ax.set_xlabel('Residue or Atom Index')
    ax.set_ylabel('RMSD (�~E)')
    plt.title('RMSF Plot')
    plt.legend()
    plt.show()
    plt.savefig(save_plot)
    # save_npy = save_plot.replace("png", "npy")
    # np.save(save_npy, dio)
    return rmsd_trajectory, rmsd_output, dio


##########
## RMSF ##
##########

def calculate_rmsf(trajectory):
    # Calculate the mean position for each atom across all frames
    mean_positions = np.mean(trajectory, axis=0)

    # Calculate the squared deviations from the mean
    squared_deviations = (trajectory - mean_positions) ** 2

    # Sum the squared deviations for each coordinate, then average over frames and take the square root
    rmsf = np.sqrt(np.mean(np.sum(squared_deviations, axis=2), axis=0))
    return rmsf

def RMSF_plot(trajectory_data, output_data, save_plot):
    # Calculate RMSF for both datasets
    rmsf_trajectory = calculate_rmsf(trajectory_data)
    rmsf_output = calculate_rmsf(output_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 6))

    # Assuming the x-axis represents residue or atom indices
    indices = np.arange(len(rmsf_trajectory))

    # Plot RMSF for trajectory_data
    ax.plot(indices, rmsf_trajectory, color='blue', linewidth=1, label='Original')

    # Plot RMSF for output_data
    ax.plot(indices, rmsf_output, color='red', linewidth=1, label='Reconstruction')

    # Labels, title, and legend
    ax.set_xlabel('Residue or Atom Index')
    ax.set_ylabel('RMSF (Å)')
    plt.title('RMSF Plot')
    plt.legend()
    plt.show()
    plt.savefig(save_plot)

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 6))

    save_plot = save_plot.replace("rmsf", "dio_rmsf")    

    # Plot RMSF for DIO
    dio = abs(rmsf_trajectory - rmsf_output)
    print("Mean DIO: ", np.nanmean(dio))
    ax.plot(indices, dio, color='orange', linewidth=1, label='DIO')
    #ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=1)

    # Labels, title, and legend
    ax.set_xlabel('Residue or Atom Index')
    ax.set_ylabel('RMSF (Å)')
    plt.title('RMSF Plot')
    plt.legend()
    plt.show()
    plt.savefig(save_plot)
    # save_npy = save_plot.replace("png", "npy")
    # np.save(save_npy, dio)
    return rmsf_trajectory, rmsf_output, dio

##################
## Latent Space ##
##################

def plot_latent_space(z, save_plot):
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure(figsize=(15, 10))

    ax_main = plt.subplot(gs[1:4, :3])
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)

    sc = ax_main.scatter(z[::1, 0], z[::1, 1], c=np.arange(len(z)), alpha=1, cmap='jet', s=2)

    # Position and size of colorbar based on ax_yDist
    pos = ax_yDist.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Frames')

    # X-axis marginal distribution
    ax_xDist.hist(z[::1, 0], bins=100, color='blue', alpha=0.7)

    # Y-axis marginal distribution
    ax_yDist.hist(z[::1, 1], bins=100, color='blue', alpha=0.7, orientation='horizontal')

    ax_main.set_xlabel('z0', labelpad=20)
    ax_main.set_ylabel('z1', labelpad=20)
    plt.show()
    plt.savefig(save_plot)

###############
## RMSD/RMSF ##
###############


def calculate_rmsd_trajectories(traj1, traj2):
    """
    Calculate RMSD between two trajectories over time.

    Parameters:
    traj1: np.array of shape (num_frames, num_atoms, 3)
        Coordinates for the first trajectory.
    traj2: np.array of shape (num_frames, num_atoms, 3)
        Coordinates for the second trajectory.

    Returns:
    rmsd_per_frame: np.array of shape (num_frames,)
        RMSD for each frame.
    """
    # Ensure the two arrays have the same shape
    assert traj1.shape == traj2.shape, "Trajectories must have the same shape"

    # Calculate the RMSD for each frame
    diff = traj1 - traj2
    rmsd_per_frame = np.sqrt(np.nanmean(np.sum(diff**2, axis=2), axis=1))
    return rmsd_per_frame


def calculate_rmsf_trajectories(traj1, traj2):
    """
    Calculate RMSF (per-atom fluctuation) between two trajectories.

    Parameters:
    traj1: np.array of shape (num_frames, num_atoms, 3)
        Coordinates for the first trajectory.
    traj2: np.array of shape (num_frames, num_atoms, 3)
        Coordinates for the second trajectory.

    Returns:
    rmsf_per_atom: np.array of shape (num_atoms,)
        RMSF for each atom across all frames.
    """
    # Ensure the two arrays have the same shape
    assert traj1.shape == traj2.shape, "Trajectories must have the same shape"

    # Calculate per-atom fluctuations between the two trajectories
    diff = traj1 - traj2
    rmsf_per_atom = np.sqrt(np.nanmean(np.nanmean(diff**2, axis=0), axis=1))

    return rmsf_per_atom
