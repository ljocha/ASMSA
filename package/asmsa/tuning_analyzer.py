from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors        
import numpy as np
import pickle
import os
import glob
from pickle import UnpicklingError

class TuningAnalyzer():
    def __init__(self,tuning=None):
        # get latest tuning dir or the one selected by user
        self.tuning_dir = max(glob.glob(os.path.join(f'{os.getcwd()}/analysis', '*/')), key=os.path.getmtime) if tuning==None else tuning
        self.sorted_trials = self._analyze()
        print(f'Analyzing tuning from: {self.tuning_dir}')

    def populate_TB(self, out_dir=None, trials=[], num_trials=10):
        # add trailing slash if not present to go one subdir below
        self.tuning_dir = self.tuning_dir if self.tuning_dir[-1]=='/' else self.tuning_dir+'/'
        summary_writer = SummaryWriter(f'{self.tuning_dir}_TB' if not out_dir else out_dir)

        first_model = list(self.sorted_trials[0]['results'].items())[0][0]
        num_epochs = len(self.sorted_trials[0]['results'][first_model]['ae_loss'])

        for i in range(len(self.sorted_trials)):
            if i == num_trials:
                break

            trial = self.sorted_trials[i]
            if trials and trial['trial_id'] not in trials:
                continue
            for epoch in range(num_epochs):
                ae_losses={};dn_losses={};kl_divs={}
                for k,v in trial['results'].items():
                    ae_losses[k] = v['ae_loss'][epoch]
                    dn_losses[k] = v['dn_loss'][epoch]
                    kl_divs[k] = v['kl_div'][epoch]
                summary_writer.add_scalars(f'{trial["trial_id"]}/ae_loss', ae_losses, epoch)
                summary_writer.add_scalars(f'{trial["trial_id"]}/Discriminator_loss', dn_losses, epoch)
                summary_writer.add_scalars(f'{trial["trial_id"]}/KL_divergence', kl_divs, epoch)

        summary_writer.close()


    def visualize_tuning(self, trial=None, num_trials=10):
        if trial:
            num_trials = len(self.sorted_trials)
        
        # populate x axis with number of epochs from first model of first trial
        first_model = list(self.sorted_trials[0]['results'].items())[0][0]
        num_epochs = len(self.sorted_trials[0]['results'][first_model]['ae_loss'])
        x = np.arange(start=0, stop=num_epochs, step=1)
        
        # Create a distinct color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        colors = np.vstack([colors, plt.cm.Set3(np.linspace(0, 1, 12))])
        
        first_plot_saved = False
    
        for i in range(len(self.sorted_trials)):
            if i == num_trials:
                break
            _trial = self.sorted_trials[i]
            if trial and trial != _trial["trial_id"]:
                continue
            print(f'Trial ID: {_trial["trial_id"]}')
            
            # Create a figure with 3 subplots
            fig = plt.figure(figsize=(14, 8))  # Increased width for legend space
            
            # Dictionary to map models to colors
            model_colors = {}
            color_idx = 0
            
            # Per gestire una sola legenda comune
            handles = []
            labels = []
            
            for k, v in _trial['results'].items():
                # Assign a unique color to each model
                if k not in model_colors:
                    model_colors[k] = colors[color_idx % len(colors)]
                    color_idx += 1
                    
                for plot, measure in [(311, 'ae_loss'), (312, 'dn_loss'), (313, 'kl_div')]:
                    ax = plt.subplot(plot)
                    line, = plt.plot(x, _trial['results'][k][measure], label=k, color=model_colors[k])
                    plt.ylabel(measure, fontsize=12)
                    plt.xlabel("epoch", fontsize=12)
                    plt.ylim([0, max(_trial['results'][k][measure])])
                    plt.autoscale()
                    if plot != 313:
                        plt.xticks([])
                    # raccogli handle e label per una sola legenda finale
                    handles.append(line)
                    labels.append(k)
            
            # rimuove duplicati nelle legende
            unique = dict(zip(labels, handles))
            
            # riduce lo spazio verticale tra i plot
            plt.subplots_adjust(hspace=0.15, right=0.83)  # più vicino e più spazio a destra
            
            # legenda unica, leggermente più spostata a destra
            fig.legend(unique.values(), unique.keys(),
                       loc='center left', bbox_to_anchor=(0.88, 0.5),
                       fontsize=12, title="Models")
            
            plt.xticks(range(0, num_epochs))
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # assicura che la legenda non tagli il grafico
            
            # Save only the first generated image
            if not first_plot_saved:
                plt.savefig(f'trial_{_trial["trial_id"]}_tuning_first_results.png', 
                           bbox_inches='tight', dpi=600)
                first_plot_saved = True
                print(f"Saved first plot: trial_{_trial['trial_id']}_tuning_first_results.png")
            
            plt.show()

    def get_best_hp(self,num_trials=10):
        def _get_beautiful_output(d):
            for k,v in d.items():
                if k == 'trial_id':
                    print(f'Trial ID: {v}')
                elif k == 'hp':
                    print('Hyperparameters:')
                    for k1, v1 in v.items():
                        print(f'{k1}: {v1}')
                elif k == 'score':
                    print(f'Score: {v}')

        print(f'Printing results of tuning: {self.tuning_dir}')

        for i in range(num_trials):
            if i == len(self.sorted_trials):
                break
            print(f'-----({i+1})-----')
            _get_beautiful_output(self.sorted_trials[i])


    def _analyze(self):
        trials = []
        for trial in os.listdir(self.tuning_dir):
            trial_path = f'{self.tuning_dir}/{trial}'
            if not os.path.isdir(trial_path):
                try:
                    d = pickle.load(open(trial_path, 'rb'))
                except UnpicklingError as e:
                    raise Exception(f'No files to Unpickle - probably chose wrong directory where there are no pickle objects. Double check TuningAnalyzer\'s tuning path.')
                trials.append(d)

        return sorted(trials, key=lambda d: d['score'])

