import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from IPython.display import clear_output

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
                 freq=1):
        """
        Initialize the callback.
        
        Args:
            metric_groups: Dictionary with group names as keys and lists of metrics as values
                Default: {'Autoencoder Loss': ['AE loss min', 'val_AE loss min'],
                          'Discriminator Loss': ['disc loss min', 'val_disc loss min']}
            figsize: Figure size (width, height). Default: (16, 6)
            freq: Update frequency in epochs. Default: 1 (update after each epoch)
        """
        super(LiveTrainingPlot, self).__init__()
        self.metric_groups = metric_groups
        self.figsize = figsize
        self.freq = freq
        self.colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
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
                # Skip metrics not in logs but don't add None or it'll break plotting
                continue
        
        # Only update plot on specified frequency to avoid slowing down training
        if (epoch + 1) % self.freq == 0 or epoch == 0:
            self._update_plot()
    
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
            for j, metric_name in enumerate(group_metrics):
                if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
                    continue  # Skip metrics with no data
                    
                metric_values = self.metrics[metric_name]
                
                # Find the index of the minimum value
                best_idx = np.argmin(metric_values)
                best_epochs[metric_name] = self.epochs[best_idx]
                
                # Plot the metric
                color = self.colors[j % len(self.colors)]
                ax.plot(self.epochs, metric_values, 'o-', label=metric_name, color=color)
                
                # Add vertical line at best epoch
                ax.axvline(self.epochs[best_idx], linestyle='--', color=color, alpha=0.5)
            
            # Set title and labels for this subplot
            ax.set_title(group_name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create a legend with best epochs
            legend_labels = []
            for metric_name, best_epoch in best_epochs.items():
                best_value = self.metrics[metric_name][self.epochs.index(best_epoch)]
                legend_labels.append(f"{metric_name} (best: {best_value:.6f} @ {best_epoch})")
            
            ax.legend(legend_labels)
        
        plt.tight_layout()
        plt.show()