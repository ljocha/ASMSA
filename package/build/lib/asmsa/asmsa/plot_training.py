import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython.display import clear_output
import os

class LiveTrainingPlot(tf.keras.callbacks.Callback):
    """
    Callback to plot multiple training and validation metrics in real-time during training,
    with separate subplots for different metric groups.
    """
    
    def __init__(self, 
                 metric_groups={
                     'Autoencoder Loss': ['AE loss min', 'val_AE loss min'],
                     'Discriminator Loss': ['disc loss min', 'val_disc loss min']
                 }, 
                 figsize=(16, 6), 
                 freq=1,
                 save=False,
                 save_path='training_plot.png',
                 dpi=300):
        """
        Initialize the callback.
        
        Args:
            metric_groups: Dictionary with group names as keys and lists of metrics as values
                Default: {'Autoencoder Loss': ['AE loss min', 'val_AE loss min'],
                          'Discriminator Loss': ['disc loss min', 'val_disc loss min']}
            figsize: Figure size (width, height). Default: (16, 6)
            freq: Update frequency in epochs. Default: 1 (update after each epoch)
            save: Whether to save the final plot. Default: False
            save_path: Path where to save the final plot. Default: 'training_plot.png'
            dpi: DPI for saved image. Default: 300
        """
        super(LiveTrainingPlot, self).__init__()
        self.metric_groups = metric_groups
        self.figsize = figsize
        self.freq = freq
        self.save = save
        self.save_path = save_path
        self.dpi = dpi
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'x', '+', 'h']
        
    def on_train_begin(self, logs=None):
        # Flatten all metrics into a single list
        all_metrics = []
        for metrics in self.metric_groups.values():
            all_metrics.extend(metrics)
            
        self.metrics = {}
        for metric in all_metrics:
            self.metrics[metric] = []
        self.epochs = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Store metrics at the end of each epoch
        logs = logs or {}
        self.epochs.append(epoch + 1)  # +1 because epochs are 0-indexed
        
        for metric in self.metrics.keys():
            if metric in logs:
                self.metrics[metric].append(logs[metric])
            else:
                # Handle nested metrics with double 'val_' prefix
                # For example, val_val_AE loss min should match with val_AE loss min
                # from logs if both refer to the same metric
                metric_without_prefix = metric
                if metric.startswith('val_val_'):
                    metric_without_prefix = 'val_' + metric[8:]
                    
                if metric_without_prefix in logs:
                    self.metrics[metric].append(logs[metric_without_prefix])
                else:
                    # Skip metrics not in logs but don't add None or it'll break plotting
                    continue
        
        # Only update plot on specified frequency to avoid slowing down training
        if (epoch + 1) % self.freq == 0 or epoch == 0:
            self._update_plot()
    
    def on_train_end(self, logs=None):
        """Called at the end of training to save the final plot if save=True"""
        if self.save:
            self._save_final_plot()
    
    def _update_plot(self):
        # Clear the previous output in Jupyter notebook
        clear_output(wait=True)
        
        # Create a figure with subplots - one for each metric group
        fig, axes = plt.subplots(1, len(self.metric_groups), figsize=self.figsize)
        
        # If there's only one group, axes won't be an array, so make it one
        if len(self.metric_groups) == 1:
            axes = [axes]
        
        # Plot each group in its own subplot
        for i, (group_name, group_metrics) in enumerate(self.metric_groups.items()):
            ax = axes[i]
            
            best_epochs = {}
            plot_handles = []
            for j, metric_name in enumerate(group_metrics):
                if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                    continue  # Skip metrics with no data
                    
                metric_values = self.metrics[metric_name]
                
                # Find the index of the minimum value
                best_idx = np.argmin(metric_values)
                best_epochs[metric_name] = self.epochs[best_idx]
                
                # Plot the metric with consistent marker and line style
                color = self.colors[j % len(self.colors)]
                marker = self.markers[j % len(self.markers)]
                line, = ax.plot(self.epochs, metric_values, marker=marker, linestyle='-', 
                              label=metric_name, color=color, markersize=5)
                plot_handles.append(line)
                
                # Add vertical line at best epoch
                ax.axvline(self.epochs[best_idx], linestyle='--', color=color, alpha=0.5)
            
            # Set title and labels for this subplot
            ax.set_title(group_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create a legend with best epochs using the actual plot handles
            legend_labels = []
            for k, metric_name in enumerate(group_metrics):
                if metric_name in best_epochs:
                    best_epoch = best_epochs[metric_name]
                    best_value = self.metrics[metric_name][self.epochs.index(best_epoch)]
                    legend_labels.append(f"{metric_name} (best: {best_value:.6f} @ {best_epoch})")
            
            if plot_handles:
                ax.legend(plot_handles, legend_labels)
        
        plt.tight_layout()
        plt.show()
    
    def _save_final_plot(self):
        """Save the final training plot"""
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(self.save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Create a new figure for saving (don't interfere with live plotting)
            fig, axes = plt.subplots(1, len(self.metric_groups), figsize=self.figsize)
            
            # If there's only one group, axes won't be an array, so make it one
            if len(self.metric_groups) == 1:
                axes = [axes]
            
            # Plot each group in its own subplot
            for i, (group_name, group_metrics) in enumerate(self.metric_groups.items()):
                ax = axes[i]
                
                best_epochs = {}
                plot_handles = []
                for j, metric_name in enumerate(group_metrics):
                    if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                        continue  # Skip metrics with no data
                        
                    metric_values = self.metrics[metric_name]
                    
                    # Find the index of the minimum value
                    best_idx = np.argmin(metric_values)
                    best_epochs[metric_name] = self.epochs[best_idx]
                    
                    # Plot the metric with consistent marker and line style
                    color = self.colors[j % len(self.colors)]
                    marker = self.markers[j % len(self.markers)]
                    line, = ax.plot(self.epochs, metric_values, marker=marker, linestyle='-', 
                                  label=metric_name, color=color, markersize=5)
                    plot_handles.append(line)
                    
                    # Add vertical line at best epoch
                    ax.axvline(self.epochs[best_idx], linestyle='--', color=color, alpha=0.5)
                
                # Set title and labels for this subplot
                ax.set_title(group_name)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Create a legend with best epochs using the actual plot handles
                legend_labels = []
                for k, metric_name in enumerate(group_metrics):
                    if metric_name in best_epochs:
                        best_epoch = best_epochs[metric_name]
                        best_value = self.metrics[metric_name][self.epochs.index(best_epoch)]
                        legend_labels.append(f"{metric_name} (best: {best_value:.6f} @ {best_epoch})")
                
                if plot_handles:
                    ax.legend(plot_handles, legend_labels)
            
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)  # Close the figure to free memory
            print(f"Training plot saved to: {self.save_path}")
            
        except Exception as e:
            print(f"Error saving plot: {e}")