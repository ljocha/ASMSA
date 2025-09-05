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

    
    def visualize_tuning(self, trial=None, num_trials=10, save=False, save_path=None, dpi=300):
        """
        Visualizza i risultati del tuning dei modelli.
        
        Args:
            trial: ID del trial specifico da visualizzare (opzionale)
            num_trials: Numero di trial da visualizzare se trial non è specificato
            save: Se True, salva le immagini generate
            save_path: Percorso base per salvare le immagini. Se None, usa 'tuning_plots/'
            dpi: Risoluzione delle immagini salvate
        """
        if trial:
            num_trials = len(self.sorted_trials)
        first_model = list(self.sorted_trials[0]['results'].items())[0][0]
        num_epochs = len(self.sorted_trials[0]['results'][first_model]['ae_loss'])
        x = np.arange(start=0, stop=num_epochs, step=1)
    
        colors_list = []
        
        colors_list.extend(plt.cm.tab10(np.linspace(0, 1, 10)))      
        
        colors_list.extend(plt.cm.Set1(np.linspace(0, 1, 9)))        
        colors_list.extend(plt.cm.Set2(np.linspace(0, 1, 8)))        
        colors_list.extend(plt.cm.Set3(np.linspace(0, 1, 12)))       
        
        colors_list.extend(plt.cm.Dark2(np.linspace(0, 1, 8)))       
        colors_list.extend(plt.cm.Paired(np.linspace(0, 1, 12)))     
        
        colors_list.extend(plt.cm.viridis(np.linspace(0, 1, 8)))     
        colors_list.extend(plt.cm.plasma(np.linspace(0, 1, 8)))      
        colors_list.extend(plt.cm.inferno(np.linspace(0, 1, 8)))     
        colors_list.extend(plt.cm.cividis(np.linspace(0, 1, 8)))     
        
        colors = np.array(colors_list)
        
        # Setup save path base se necessario
        if save and save_path is None:
            save_path = 'tuning_plots'
        
        if save and save_path:
            # Crea la directory se non esiste
            os.makedirs(save_path, exist_ok=True)
        
        measures = ['ae_loss', 'dn_loss', 'kl_div']
        for i in range(len(self.sorted_trials)):
            if i == num_trials:
                break
            _trial = self.sorted_trials[i]
            if trial and trial != _trial["trial_id"]:
                continue
            print(f'Trial ID: {_trial["trial_id"]}')
            
            # Pre-calcolo dei limiti per metrica
            limits = {}
            for m in measures:
                all_vals = np.concatenate([
                    np.asarray(v[m]) for v in _trial['results'].values()
                ])
                min_v, max_v = float(np.min(all_vals)), float(np.max(all_vals))
                if m in ('ae_loss', 'dn_loss'):
                    if np.isclose(min_v, max_v):
                        # caso piatto → apri un range "carino"
                        if max_v == 0:
                            ymin, ymax = -0.1, 0.1
                        else:
                            delta = abs(max_v) * 0.1
                            ymin, ymax = max_v - delta, max_v + delta
                    else:
                        pad = 0.1 * (max_v - min_v)
                        ymin, ymax = min_v - pad, max_v + pad
                else:
                    # kl_div: 0 → max con margine
                    ymax = max_v if max_v > 0 else 1.0
                    ymin, ymax = 0.0, ymax * 1.1
                limits[m] = (ymin, ymax)
            
            # Figura con 3 subplot
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.subplots_adjust(right=0.87)
            
            # Colori per modello - ora con molte più opzioni
            model_colors, color_idx = {}, 0
            for k, v in _trial['results'].items():
                if k not in model_colors:
                    model_colors[k] = colors[color_idx % len(colors)]
                    color_idx += 1
                for ax, m in zip(axes, measures):
                    y = np.asarray(v[m])
                    ax.plot(x, y, label=k, color=model_colors[k], 
                           linewidth=2.5,  
                           alpha=0.9)      
                    ax.set_ylabel(m)
                    ax.set_ylim(*limits[m])
                    ax.grid(True, alpha=0.3)  
            
            axes[-1].set_xticks(np.arange(0, num_epochs + 1, 10))
            axes[-1].set_xlabel("Epochs")
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
            
            handles, labels = axes[0].get_legend_handles_labels()
            uniq_h, uniq_l = [], []
            seen = set()
            for h, lab in zip(handles, labels):
                if lab not in seen:
                    uniq_h.append(h); uniq_l.append(lab); seen.add(lab)
            
            fig.legend(
                uniq_h, uniq_l,
                loc='center left',
                bbox_to_anchor=(0.89, 0.5),
                fontsize=11,
                frameon=True, fancybox=True, framealpha=0.95,
                edgecolor='0.4',
                borderpad=0.8, labelspacing=0.6, handletextpad=0.7, handlelength=2.5
            )
            
            plt.tight_layout(rect=[0, 0, 0.83, 1])
            
            # Salva l'immagine se richiesto
            if save and save_path:
                try:
                    # Nome file con trial ID
                    filename = f"trial_{_trial['trial_id']}_tuning_plot.png"
                    full_path = os.path.join(save_path, filename)
                    
                    plt.savefig(full_path, dpi=dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    print(f"Plot salvato in: {full_path}")
                    
                except Exception as e:
                    print(f"Errore nel salvare il plot per trial {_trial['trial_id']}: {e}")
            
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

