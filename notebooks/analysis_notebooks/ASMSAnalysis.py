import os
import numpy as np
import matplotlib.pyplot as plt
import time
import mdtraj as md
import nglview as nv
import gromacs as gmx
import joblib
from functools import lru_cache
from datetime import datetime
from matplotlib.gridspec import GridSpec
from IPython.display import display, clear_output
from ipywidgets import interact, IntSlider, HBox, Output


class Analysis:
    """
    Class for enhanced molecular dynamics simulation analysis and visualization.
    """
    
    def __init__(self, conf, traj, hills, lows_filename):
        """
        Initialize the Analysis class.
        
        Parameters:
        -----------
        conf : str
            Path to the configuration file
        traj : str
            Path to the trajectory file
        hills : str
            Path to the HILLS file from metadynamics
        lows_filename : str
            Path to the latent space representation file
        """
        self.hills = hills
        self.lows_filename = lows_filename
        self.conf = conf
        self.traj = traj
        self._hills_data = None
        self._last_hills_length = 0
    
    def latent_space(self):
        """Load and return latent space and RMSD data."""
        tr = md.load(self.traj, top=self.conf)
        lows = np.loadtxt(self.lows_filename)
        base = md.load(self.conf)
        lows_rmsd = md.rmsd(tr, base[0])
        lows_rg = md.compute_rg(tr), 
        return lows, lows_rmsd, lows_rg

    def prepare_trajectory(self):
        """Prepare trajectory for analysis by removing PBCs and fitting."""
        # Execute gromacs commands to prepare trajectory files
        gmx.trjconv(f='md.xtc', s='npt.gro', input='Protein-H', o='protein.xtc')
        gmx.trjconv(f='protein.xtc', s='npt_nh.pdb', pbc='nojump', 
                    input='Protein-H', o='MD_pbc.xtc')
        gmx.trjconv(f='MD_pbc.xtc', s='npt_nh.pdb', center=True, 
                    fit='rot+trans', input='Protein-H Protein-H Protein-H'.split(), 
                    o='MD_fit.xtc')

        conf_nh = "npt_nh.pdb"
        traj_fit = "MD_fit.xtc"

        return conf_nh, traj_fit
    
    @staticmethod
    @lru_cache(maxsize=None)
    def load_hills(hills_file):
        """Load and cache HILLS data."""
        return np.loadtxt(hills_file)

    def read_last_step_time(self, filename='md.log'):
        """Read the last step and time from log file."""
        last_step = None
        last_time = None
        
        try:
            with open(filename, 'r') as file:
                # Read the file in reverse for efficiency
                for line in reversed(file.readlines()):
                    if "Step" in line and "Time" in line:
                        # Get the line after the header
                        continue
                    else:
                        parts = line.split()
                        if len(parts) >= 2 and parts[0].isdigit():
                            last_step, last_time = parts[0], parts[1]
                            break
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return "N/A", "0"
            
        return last_step, last_time
            
    def update_hills_data(self):
        """Update HILLS data by only loading new entries."""
        try:
            new_hills = np.loadtxt(self.hills, skiprows=self._last_hills_length)
            if new_hills.size > 0:
                if isinstance(new_hills, np.ndarray) and new_hills.ndim == 1:
                    # Reshape single row to 2D array
                    new_hills = new_hills.reshape(1, -1)
                    
                if self._hills_data is None:
                    self._hills_data = new_hills
                else:
                    self._hills_data = np.vstack((self._hills_data, new_hills))
                    
                self._last_hills_length = self._hills_data.shape[0]
                
            return self._hills_data
        except Exception as e:
            print(f"Error updating HILLS data: {e}")
            return self._hills_data
    
    def on_the_flight(self, interval=60, cmap='rainbow'):
        """
        Real-time visualization of enhanced molecular dynamics simulation.
        
        Parameters:
        -----------
        interval : float
            Time in seconds to update the visualization
        cmap : str
            Colormap for visualization
        """
        lows, lows_rmsd, lows_rg = self.latent_space()
        
        plt.ion()  # Enable interactive mode
        
        l1, l2 = lows[:,0], lows[:,1]
    
        try:
            while True:
                # Read simulation status
                last_step, last_time = self.read_last_step_time()
                times = float(last_time) / 1000 if last_time != "N/A" else 0
                
                # Check if simulation is running
                if os.path.exists('md.log'):
                    last_modified = os.path.getmtime('md.log')
                    current_time = datetime.now().timestamp()
                    seconds_ago = current_time - last_modified
                    
                    if seconds_ago < 100:
                        print(f"Running time: {times:.2f} ns")
                    else:
                        print("Simulation not currently running...")
                
                # Update hills data
                hills_data = self.update_hills_data()
                if hills_data is None or len(hills_data) == 0:
                    print("No data available yet")
                    time.sleep(interval)
                    continue
    
                # Extract data columns
                t = hills_data[:, 0] / 1000  # Time in ns
                cv1, cv2 = hills_data[:, 1], hills_data[:, 2]
                cv3, hh = hills_data[:, 3], hills_data[:, 5]
    
                # Create figure with new grid layout: 2 rows x 2 colonne
                fig = plt.figure(figsize=(10, 8))
                gs = GridSpec(2, 2, height_ratios=[1, 1])
    
                # Primo plot: sulla prima riga, occupa entrambe le colonne
                ax1 = fig.add_subplot(gs[0, :])
                ax1.scatter(cv1, cv2, c=t, cmap=cmap, marker='.', s=1, zorder=1)
                ax1.scatter(cv1[-1], cv2[-1], c='black', marker='x', s=25, zorder=2)
                ax1.scatter(cv1[0], cv2[0], c='black', marker='o', s=25, zorder=2)
                ax1.scatter(lows[:, 0], lows[:, 1], c='gray', marker='.', s=1, zorder=0)
                ax1.set(xlabel='cv1', ylabel='cv2')
                ax1.set_title(f'o: reference structure | x: step {last_step}')
                ax1.set_aspect('equal', adjustable='box')
    
                # Seconda riga, primo plot: RMSD nel latent space
                ax2 = fig.add_subplot(gs[1, 0])
                scatter_rmsd = ax2.scatter(l1, l2, c=lows_rmsd, cmap=cmap, marker='.', s=5)
                ax2.scatter(cv1[0], cv2[0], c='black', marker='o', s=25)
                ax2.scatter(cv1[-1], cv2[-1], c='black', marker='x', s=25)
                ax2.set_xlim(np.min(cv1), np.max(cv1))
                ax2.set_ylim(np.min(cv2), np.max(cv2))
                ax2.set(xlabel='cv1', ylabel='cv2')
                ax2.set_title('AAE - Latent Space - RMSD')
    
                # Seconda riga, secondo plot: Rg nel latent space
                ax3 = fig.add_subplot(gs[1, 1])
                scatter_rg = ax3.scatter(l1, l2, c=lows_rg, cmap=cmap, marker='.', s=5)
                ax3.scatter(cv1[0], cv2[0], c='black', marker='o', s=25)
                ax3.scatter(cv1[-1], cv2[-1], c='black', marker='x', s=25)
                ax3.set_xlim(np.min(cv1), np.max(cv1))
                ax3.set_ylim(np.min(cv2), np.max(cv2))
                ax3.set(xlabel='cv1', ylabel='cv2')
                ax3.set_title('AAE - Latent Space - RG')
    
                plt.tight_layout()
                display(fig)
                clear_output(wait=True)
                time.sleep(interval)
        
        except KeyboardInterrupt:
            plt.ioff()
            print("Visualization stopped by user")


    def calculate_rmsd(self, index):
        """Calculate and plot RMSD over trajectory."""
        conf_nh, traj_fit = self.prepare_trajectory()
       
        tr = md.load(traj_fit, top=conf_nh)
        
        tr.superpose(tr, atom_indices=index)

        # Calculate RMSD
        rmsds = md.rmsd(tr, tr, 0, atom_indices=index, precentered=True)
    
        # Plot RMSD
        plt.figure(figsize=(10, 4))
        plt.title(f'RMSD - {int(tr.time[-1]/1000)} ns')
        plt.scatter(tr.time/1000, rmsds, c=rmsds, s=0.1, cmap='jet')
        plt.xlabel('Time (ns)')
        plt.ylabel('RMSD (nm)')
        plt.colorbar(label='RMSD (nm)')
        plt.savefig('rmsd.png', dpi=300)
        plt.show()
        return rmsds
        
    def highlights_and_dynamic(self, cmap='rainbow', color_by_rmsd=True):
        """
        Interactive visualization of structure dynamics and CV space with synchronized controls.
        
        Parameters:
        -----------
        cmap : str
            Colormap for visualization
        color_by_rmsd : bool
            Whether to color the visualization by RMSD values
        """
        # Prepare trajectory
        conf_nh, traj_fit = self.prepare_trajectory()
        tr = md.load(traj_fit, top=conf_nh)
        
        # Align trajectory on CA atoms
        idx = tr.top.select("name CA")
        tr.superpose(tr, atom_indices=idx)
        
        # Calculate RMSD for coloring
        if color_by_rmsd:
            rmsds = md.rmsd(tr, tr, 0, precentered=True)
            # No need to normalize RMSD here, we'll use the raw values
        else:
            rmsds = None
        
        # Load HILLS data
        hills_data = np.loadtxt(self.hills)
        t = hills_data[:, 0] / 1000  # Time in ns
        cv1, cv2 = hills_data[:, 1], hills_data[:, 2]
        
        # Ensure time frames match between trajectory and hills data
        # We need to map trajectory frames to hills data time points
        traj_times = tr.time / 1000  # Convert to ns for consistency
        
        # Build a mapping between trajectory times and hills times
        # Create bidirectional mappings for synchronization
        traj_to_hills = {}  # Maps from trajectory frame to closest hills index
        hills_to_traj = {}  # Maps from hills index to closest trajectory frame
        
        for i, hill_time in enumerate(t):
            closest_frame = np.argmin(np.abs(traj_times - hill_time))
            hills_to_traj[i] = closest_frame
            
        for i, traj_time in enumerate(traj_times):
            closest_hill = np.argmin(np.abs(t - traj_time))
            traj_to_hills[i] = closest_hill
        
        # Setup NGLView visualization with GUI enabled for play button
        v = nv.show_mdtraj(tr, default=False, gui=True)  # Enable GUI for play button
        v.add_cartoon(selection="protein", color='red')
        c = v.add_component(tr[0], default=False, gui=False)
        c.add_cartoon(selection="protein", color='blue')
        v.center(selection='protein')
        v.render_image(trim=True, factor=3)
        v.camera = 'orthographic'
        v.background = 'white'
        
        # Create output for plot and info
        out_plot = Output()
        info_output = Output()
        
        # Track current frame for both visualizations
        current_hills_index = 0
        current_traj_frame = 0
        
        # Function to update the plot based on slider value and coloring mode
        def update_plot(hills_index):
            nonlocal current_hills_index
            current_hills_index = hills_index
            
            # Get current coloring mode
            coloring_mode = color_selector.value
            
            with out_plot:
                clear_output(wait=True)
    
                # Make the figure larger
                fig, ax = plt.subplots(figsize=(10, 8))
                # Plot full trajectory path
                ax.scatter(cv1, cv2, c='grey', marker='.', s=0.1)
                
                # Determine coloring scheme based on dropdown selection
                if coloring_mode == 'RMSD' and rmsds is not None:
                    # Map hills data points to trajectory frames for RMSD coloring
                    # Use raw RMSD values for coloring
                    color_data = np.array([rmsds[hills_to_traj[i]] for i in range(hills_index+1)])
                    cmap_to_use = 'rainbow'  # Good colormap for RMSD
                else:
                    # Default coloring by time
                    color_data = t[:hills_index+1]
                    cmap_to_use = cmap
                
                # Plot path up to current frame with coloring
                scatter = ax.scatter(cv1[:hills_index+1], cv2[:hills_index+1], 
                           c=color_data, cmap=cmap_to_use, marker='.', s=1)
                
                # No colorbar as requested
                
                # Mark current position
                ax.scatter(cv1[hills_index], cv2[hills_index], c='black', marker='x', s=50, zorder=2,  edgecolors='black', linewidths=5)
                ax.set_title(f'Time: {t[hills_index]:.2f} ns')
                ax.set_xlabel('cv1')
                ax.set_ylabel('cv2')
                plt.tight_layout()
                plt.show()
                
            # Update info display
            with info_output:
                clear_output(wait=True)
                print(f"Hills data time: {t[hills_index]:.2f} ns")
                print(f"Corresponding trajectory frame: {hills_to_traj[hills_index]}")
                print(f"Trajectory time: {traj_times[hills_to_traj[hills_index]]:.2f} ns")
                
                if rmsds is not None:
                    traj_frame = hills_to_traj[hills_index]
                    print(f"RMSD at current frame: {rmsds[traj_frame]:.4f} nm")
            
            # Update NGLView to the corresponding frame
            traj_frame = hills_to_traj[hills_index]
            v._remote_call('setFrame', target='Stage', args=[traj_frame])
            
        # Function to handle NGLView frame changes
        def on_frame_change(change):
            nonlocal current_traj_frame
            new_frame = change['new']
            current_traj_frame = new_frame
            
            # Only update if the frame change is significant
            if new_frame in traj_to_hills:
                # Update the slider without triggering the slider's own callback
                hills_index = traj_to_hills[new_frame]
                slider.value = hills_index
                
                # Update the plot directly
                update_plot(hills_index)
        
        # Calculate time increments for different steps
        total_time_range = t[-1] - t[0]
        time_increments = {}
        
        # Create step sizes in nanoseconds
        if total_time_range > 10:  # If simulation is longer than 10ns
            possible_steps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]  # in nanoseconds
        else:
            possible_steps = [0.01, 0.05, 0.1, 0.5, 1.0]  # smaller steps for shorter simulations
        
        # Convert time steps to frame indices
        for step_ns in possible_steps:
            # Find the approximate number of indices to skip to achieve this time step
            avg_time_per_idx = total_time_range / len(t)
            idx_step = max(1, int(step_ns / avg_time_per_idx))
            time_increments[f"{step_ns} ns"] = idx_step
        
        # Default step size
        current_step_size = max(1, len(t)//100)
        
        # Create interactive slider to control both visualizations
        slider = IntSlider(
            min=0, 
            max=len(t)-1, 
            step=current_step_size,
            value=0, 
            description='Time:',
            continuous_update=False
        )
        
        # Display slider value as time in nanoseconds
        slider.description_tooltip = "Move to adjust time position"
        
        # Create a dropdown to select coloring mode
        from ipywidgets import Dropdown, Button, HBox, VBox, Layout, Label
        
        color_options = ['Time', 'RMSD'] if color_by_rmsd else ['Time']
        
        color_selector = Dropdown(
            options=color_options,
            value='RMSD' if color_by_rmsd else 'Time',
            description='Color by:',
            disabled=False,
        )
        
        # Create a dropdown to select the time step size
        step_selector = Dropdown(
            options=list(time_increments.keys()),
            value=list(time_increments.keys())[1],  # Default to second option
            description='Step size:',
            disabled=False,
        )
        
        # Create navigation buttons
        prev_button = Button(description='◀ Previous')
        next_button = Button(description='Next ▶')
        
        # Update plot when coloring mode changes
        def on_color_change(change):
            update_plot(slider.value)
        
        color_selector.observe(on_color_change, names=['value'])
        
        # Update slider step when dropdown selection changes
        def on_step_change(change):
            nonlocal current_step_size
            step_name = change['new']
            current_step_size = time_increments[step_name]
            slider.step = current_step_size
        
        step_selector.observe(on_step_change, names=['value'])
        
        # Button click handlers
        def on_prev_click(b):
            new_value = max(0, slider.value - current_step_size)
            slider.value = new_value
            
        def on_next_click(b):
            new_value = min(len(t)-1, slider.value + current_step_size)
            slider.value = new_value
            
        prev_button.on_click(on_prev_click)
        next_button.on_click(on_next_click)
        
        # Navigation controls layout
        nav_controls = HBox([prev_button, next_button])
        
        # Organize all controls
        visualization_controls = VBox([
            HBox([step_selector, color_selector]),
            slider,
            nav_controls
        ])
        
        # Connect slider to update function
        interact(lambda i: update_plot(i), i=slider)
        
        # Register callback for NGLView frame changes
        v.observe(on_frame_change, names=['frame'])
        
        # Initial update
        update_plot(0)
    
        # Display visualizations and info
        display(info_output)
        display(VBox([
            visualization_controls,
            HBox([out_plot, v])
        ]))
            
    
  