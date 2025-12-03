import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt


def plots(conf = "trpcage_correct.pdb",traj = "trpcage_red.xtc", lows="lowdim_uniform.txt", torsions="torsion_trp", s=0.5, save_png='analysis.png', dpi=300):
    
    # input conformation .pdb
    conf = conf
    traj = traj

    tr = md.load(traj,top=conf)
    idx=tr[0].top.select("name CA")
    tr.superpose(tr[0],atom_indices=idx)
    geom = np.moveaxis(tr.xyz ,0,-1)
    geom.shape
    
    rg = md.compute_rg(tr)
    base = md.load(conf)
    rmsd = md.rmsd(tr,base[0])
    
    t = np.linspace(0,len(rmsd),len(rmsd))
    
    # Read the data
    proj = np.loadtxt(lows)
    torsions = np.loadtxt(torsions)
    
    # Extract columns
    
    chi1 = np.degrees(torsions[:, 1])
    chi2 = np.degrees(torsions[:, 2])
    omega = np.degrees(torsions[:, 3])
    phi = np.degrees(torsions[:, 4])
    psi = np.degrees(torsions[:, 5])

    
    # Translate angles greater than 150 degrees to the negative range
    chi1[chi1 > 150] -= 360
    chi2[chi2 > 150] -= 360
    
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))

    s = s
    
    axs[0, 0].scatter(proj[:, 0], proj[:, 1], c=rmsd, cmap='rainbow', s=s)
    axs[0, 0].set_title('c by rmsd',size='15')
    axs[0, 0].set_xlabel('CV1', size="15")
    axs[0, 0].set_ylabel('CV2', size="15")
    
    # Plot chi2
    
    axs[0, 1].scatter(proj[:, 0], proj[:, 1], c=rg, cmap='rainbow', s=s)
    axs[0, 1].set_title('c by rgy',size='15')
    axs[0, 1].set_xlabel('CV1', size="15")
    axs[0, 1].set_ylabel('CV2', size="15")
    
    
    axs[0, 2].scatter(t,rmsd, c=rmsd, cmap='rainbow', s=s)
    axs[0, 2].set_title('c by rmsd',size='15')
    axs[0, 2].set_xlabel('CV1', size="15")
    axs[0, 2].set_ylabel('CV2', size="15")
    axs[0, 2].set_ylim(min(rmsd), max(rg))
    
    
    axs[0, 3].scatter(t,rg, c=rg, cmap='rainbow', s=s)
    axs[0, 3].set_title('c by rgy',size='15')
    axs[0, 3].set_xlabel('CV1', size="15")
    axs[0, 3].set_ylabel('CV2', size="15")
    
    # Plot chi1
    axs[1, 0].scatter(proj[:, 0], proj[:, 1], c=chi1, cmap='rainbow', s=s)
    axs[1, 0].set_title('c by Trp-6 - χ1',size='15')
    axs[1, 0].set_xlabel('CV1', size="15")
    axs[1, 0].set_ylabel('CV2', size="15")
    
    # Plot chi2
    axs[1, 1].scatter(proj[:, 0], proj[:, 1], c=chi2, cmap='rainbow', s=s)
    axs[1, 1].set_title('c by Trp-6 - χ2',size='15')
    axs[1, 1].set_xlabel('CV1', size="15")
    axs[1, 1].set_ylabel('CV2', size="15")
    
    # Plot chi1-chi2
    axs[1, 2].scatter(chi1, chi2, c=chi1, cmap='rainbow', s=s)
    axs[1, 2].set_title('c by Trp-6 - χ1',size='15')
    axs[1, 2].set_xlabel('chi1', size="15")
    axs[1, 2].set_ylabel('chi2', size="15")
    
    # Plot chi1-chi2
    axs[1, 3].scatter(chi1, chi2, c=chi2, cmap='rainbow', s=s)
    axs[1, 3].set_title('c by Trp-6 - χ2',size='15')
    axs[1, 3].set_xlabel('chi1', size="15")
    axs[1, 3].set_ylabel('chi2', size="15")
    
    # Plot phi
    axs[2, 0].scatter(proj[:, 0], proj[:, 1], c=phi, cmap='rainbow', s=s)
    axs[2, 0].set_title('c by Trp-6 - Φ',size='15')
    axs[2, 0].set_xlabel('CV1', size="15")
    axs[2, 0].set_ylabel('CV2', size="15")
    
    # Plot psi
    axs[2, 1].scatter(proj[:, 0], proj[:, 1], c=psi, cmap='rainbow', s=s)
    axs[2, 1].set_title('c by Trp-6 - Ψ',size='15')
    axs[2, 1].set_xlabel('CV1', size="15")
    axs[2, 1].set_ylabel('CV2', size="15")
    
    # Plot phi-psi
    axs[2, 2].scatter(phi, psi,c=phi, cmap='rainbow', s=s)
    axs[2, 2].set_title('c by Trp-6 - Φ',size='15')
    axs[2, 2].set_xlabel('phi', size="15")
    axs[2, 2].set_ylabel('psi', size="15")
    
    # Plot phi-psi
    axs[2, 3].scatter(phi, psi,c=psi, cmap='rainbow', s=s)
    axs[2, 3].set_title('c by Trp-6 - Ψ',size='15')
    axs[2, 3].set_xlabel('phi', size="15")
    axs[2, 3].set_ylabel('psi', size="15")
    
    
    
    plt.tight_layout()
    
    plt.savefig(save_png,dpi=dpi)