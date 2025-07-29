import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_evolution(data, fn = None, benchmark = None, return_fig = False):
    """
    Plots the evolution of sequences of 2 x S vectors over T time steps.
    
    Parameters:
    - data: A numpy array of shape (T, 2, S) representing the sequences of vectors.
    """
    T, _, S = data.shape
    time_steps = np.arange(T)
    states = np.arange(S)
    
    # Create subplots
    fig, axes = plt.subplots(1, T, figsize=(2 * T, 3), sharex=True, sharey=True)
    
    # Set a common color scheme
    colors = ['salmon', 'skyblue']
    barwidth = 0.5
    real_barwidth = 0.4
    for t in range(T):
        ax = axes[t] if T > 1 else axes
        dead_dist = data[t, 0, :]
        alive_dist = data[t, 1, :]
        
        positions = barwidth * states + barwidth / 2
        # Plotting dead distribution

        
        ax.bar(positions, dead_dist, color=colors[0], label='Stopped' if t == 0 else "", width = real_barwidth)
        # Plotting alive distribution stacked on top of dead distribution
        ax.bar(positions, alive_dist, bottom=dead_dist, color=colors[1], label='Continuing' if t == 0 else "", width = real_barwidth)
        
        if benchmark is not None:
            benchmark_handle = ax.bar(positions, benchmark, edgecolor='black', linestyle='--', linewidth=1.5, fill=False, width = real_barwidth, label='Target dist.')
        
        ax.set_ylim(0, np.max(data) * 1.05)
        ax.set_xlim(np.min(positions) - barwidth / 2, np.max(positions) + barwidth / 2)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.set_xticks(barwidth * states + barwidth / 2)
        ax.set_xticklabels(states)
        ax.set_xlabel(r'$X$', fontsize=14)
        
        if t != T - 1:
            ax.set_title(f'Time {t}', fontsize = 14)
        else:
            ax.set_title('Final State', fontsize = 14)
            
        
    # Add legend to the first subplot
    if benchmark is None:
        handles = [plt.Rectangle((0,0),1,1, color=colors[0]), plt.Rectangle((0,0),1,1, color=colors[1])]
        labels = ['Stopped', 'Continuing']
        
    else:
        handles = [plt.Rectangle((0,0),1,1, color=colors[0]), plt.Rectangle((0,0),1,1, color=colors[1]), benchmark_handle ]
        labels = ['Stopped', 'Continuing', 'Target dist.']
    
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.6), fontsize=12)
        
    
    # Add a common x-axis label
    # fig.text(0.5, -0.01, 'States', ha='center', fontsize=20)
    # fig.suptitle('Evolution of Mean Field Distribution', fontsize=16)
    fig.text(-0.01, 0.4, 'Probability Mass', va='center', rotation = 'vertical', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig       
    plt.show()

def plot_stop_prob(data, fn = None, return_fig = False):
    """
    Plots the decision prob vectors over T time steps.
    
    Parameters:
    - data: A numpy array of shape (T, S) representing the sequences of vectors.
    """
    
    T, S = data.shape
    time_steps = np.arange(T)
    states = np.arange(S)
    
    # Create subplots
    fig, axes = plt.subplots(1, T, figsize=(2 * T, 3), sharex=True, sharey=True)
    
    # Set a common color scheme
    colors = ['orange']
    barwidth = 0.5
    real_barwidth = 0.4
    for t in range(T):
        ax = axes[t] if T > 1 else axes
        prob_dist = data[t, :]
        
        positions = barwidth * states + barwidth / 2
        ax.bar(positions, prob_dist, color=colors[0], label='Decision Prob' if t == 0 else "", width = real_barwidth)
           
        ax.set_ylim(0, np.max(data) * 1.05)
        ax.set_xlim(np.min(positions) - barwidth / 2, np.max(positions) + barwidth / 2)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.set_xticks(barwidth * states + barwidth / 2)
        ax.set_xticklabels(states)
        ax.set_xlabel(r'$X$', fontsize=14)
        ax.set_title(f'Time {t}', fontsize = 14)
            
        
    # Add legend to the first subplot
    handles = [plt.Rectangle((0,0),1,1, color=colors[0])]
    labels = ['Decision Prob']
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.3, 0.6), fontsize=12)
    
    # Add a common x-axis label
    # fig.text(0.5, -0.01, 'States', ha='center', fontsize=20)
    # fig.suptitle('Evolution of Mean Field Distribution', fontsize=16)
    fig.text(-0.01, 0.4, 'Probability Mass', va='center', rotation = 'vertical', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig  
    plt.show()


def plot_loss_dpp(train_loss, train_loss_var, test_loss, test_loss_var, benchmark_value = None, fn = None, return_fig = False):
    """
    Plots the training and test losses with confidence intervals in separate subplots.
    
    Parameters:
    - train_loss: A numpy array of shape (T, n_iter) representing the training loss.
    - train_loss_var: A numpy array of shape (T, n_iter) representing the variance of the training loss.
    - test_loss: A numpy array of shape (T, n_iter) representing the test loss.
    - test_loss_var: A numpy array of shape (T, n_iter) representing the variance of the test loss.
    - benchmark_value : A float representing the minimum possible value for the test loss.
    """
    T, n_iter = train_loss.shape
    iterations = np.arange(n_iter)
    
    # Create subplots
    fig, axes = plt.subplots(T, 2, figsize=(8, 1.5 * T), sharex=True)
    # fig.suptitle('Training and Test Loss Over Iterations', fontsize=16)
    
    # Define line styles and colors
    line_styles = ['-', '--', '-.', ':']
    colors = plt.cm.viridis(np.linspace(0, 1, T))
    
    # Plot training loss
    for t in range(T):
       
        ax = axes[t, 0]
        ax.plot(iterations, train_loss[t], color=colors[t], linestyle=line_styles[t % len(line_styles)], label = 'Training Loss')
        ax.fill_between(iterations, train_loss[t] - train_loss_var[t], train_loss[t] + train_loss_var[t], color=colors[t], alpha=0.2)
        ax.set_ylabel(f'Time {t} \n\nLoss')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_xlim(-1, n_iter + 1)
        
        if t == 0:
            ax.set_title('Training Loss', fontsize = 14)
        
        # ax.legend(loc='best')
    
    # Plot test loss
    for t in range(T):
        ax = axes[t, 1]
        ax.plot(iterations, test_loss[t], color=colors[t], linestyle=line_styles[t % len(line_styles)], label='Testing Loss')
        ax.fill_between(iterations, test_loss[t] - test_loss_var[t], test_loss[t] + test_loss_var[t], color=colors[t], alpha=0.2)
        ax.set_ylabel(f'Loss')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        if t == 0:
            if benchmark_value is not None:
                ax.axhline(y=benchmark_value, color='red', linestyle='--', label='Optimal Cost', linewidth = 2.5)
            ax.set_title('Testing Loss', fontsize = 14)
            ax.legend(loc='best', fontsize = 'small')
            
        ax.set_xlim(-1, n_iter + 1)
    # Set common x-axis label
    for ax in axes[-1, :]:
        ax.set_xlabel('Iterations')
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
        
    if return_fig:
        return fig  
    plt.show()
    


def plot_loss_direct(train_loss, train_loss_var, test_loss, test_loss_var, benchmark_value = None, fn = None, return_fig = False):
    """
    Plots the training and test losses with confidence intervals in two horizontal subplots.
    
    Parameters:
    - train_loss: A numpy array of shape (n_iter,) representing the training loss.
    - train_loss_var: A numpy array of shape (n_iter,) representing the variance of the training loss.
    - test_loss: A numpy array of shape (n_iter,) representing the test loss.
    - test_loss_var: A numpy array of shape (n_iter,) representing the variance of the test loss.
    - benchmark_value: A float representing the benchmark value for the test loss.
    """
    n_iter = train_loss.shape[0]
    iterations = np.arange(n_iter)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    
    sns.set_palette("pastel")
    train_color = sns.color_palette("pastel")[0]
    test_color = sns.color_palette("pastel")[1]
    benchmark_color = sns.color_palette("pastel")[2]
    
    # Plot training loss
    axes[0].plot(iterations, train_loss, color=train_color, label='Training Loss')
    axes[0].fill_between(iterations, train_loss - train_loss_var, train_loss + train_loss_var, color=train_color, alpha=0.2)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].set_xlim(-1, n_iter + 1)
    
    # Plot test loss
    axes[1].plot(iterations, test_loss, color=test_color, label='Testing Loss')
    axes[1].fill_between(iterations, test_loss - test_loss_var, test_loss + test_loss_var, color=test_color, alpha=0.2)
    if benchmark_value is not None:
        axes[1].axhline(y=benchmark_value, color='red', linestyle='--', linewidth=2.5, label='Optimal Cost')
    axes[1].set_title('Testing Loss')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[1].legend(loc='best', fontsize='small')
    axes[1].set_xlim(-1, n_iter + 1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig  
    plt.show()


def plot_evolution_2D(array, fn = None, return_fig = False):
    """
    Plots heatmap grids for the given arrays.

    Parameters:
    - array1: A numpy array of shape (T, N, N) representing the first set of data.
    - array2: A numpy array of shape (T, N, N) representing the second set of data.
    """
    array1, array2 = array[:, 0, :, :], array[:, 1, :, :]
    
    T, N, _ = array1.shape

    # Use the 'viridis' colormap
    cmap = 'viridis'

    # Create subplots
    fig, axes = plt.subplots(2, T, figsize=(5 * T, 10), sharex=True, sharey=True)
    # fig.suptitle('Heatmaps of Array Entries', fontsize=16)

    # Normalize the color scale
    vmin = min(array1.min(), array2.min())
    vmax = max(array1.max(), array2.max())

    for t in range(T):
        # Plot array1 heatmap in the first row
        sns.heatmap(array1[t], ax=axes[0, t], vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, square=True)
        if t != T - 1:
            axes[0, t].set_title(f'Time: {t}', fontsize=30)
        else:
            axes[0, t].set_title(f'Final State', fontsize=30)
        axes[0, t].tick_params(axis='both', which='major', labelsize=30)
        axes[0, t].tick_params(axis='both', which='minor', labelsize=30)
        
        axes[1, t].tick_params(axis='both', which='major', labelsize=30)
        axes[1, t].tick_params(axis='both', which='minor', labelsize=30)
        
        # Plot array2 heatmap in the second row
        sns.heatmap(array2[t], ax=axes[1, t], vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, square=True)
    axes[0, 0].set_ylabel('Stopped Dist.', fontsize = 30)
    axes[1, 0].set_ylabel('Continuting Dist.', fontsize = 30)
    # Create a single colorbar for all heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar_ax.tick_params(labelsize=30)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig  
    plt.show()
    
def plot_stop_prob_2D(array, fn = None, return_fig = False):
    """
    Plots heatmap grids for the given array.

    Parameters:
    - array: A numpy array of shape (T, N, N) representing the set of data.
    - label: A string to describe the data in the colorbar (serving as a legend).
    """
        
    T, N, _ = array.shape

    # Use the 'inferno' colormap
    cmap = 'inferno'

    # Create subplots
    fig, axes = plt.subplots(1, T, figsize=(5 * T, 5), sharex=True, sharey=True)
    # Normalize the color scale
    vmin = array.min()
    vmax = array.max()

    for t in range(T):
        # Plot array heatmap in the row
        sns.heatmap(array[t], ax=axes[t], vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, square=True)
        axes[t].set_title(f'Time: {t}', fontsize=20)
        axes[t].tick_params(axis='both', which='major', labelsize=30)
        axes[t].tick_params(axis='both', which='minor', labelsize=30)
    
    # Create a single colorbar for all heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cbar_ax)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Decision Prob', fontsize=20, rotation=270, labelpad=15)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.xaxis.set_ticks_position('bottom')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight', pad_inches=0.1)
    if return_fig:
        return fig
    plt.show()