import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def extract_dihedral_indices_local(pdb_file):
    """
    Extracts atom indices (0-based, local to this PDB) required to calculate
    backbone (phi, psi, omega) and sidechain (chi) dihedral angles.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        list[tuple[int,int,int,int]]: List of quadruples of local indices (0-based)
                                      ready to index the geometry tensor.
    """

    CHI_DEFINITIONS = {
        'ARG': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'NE'), ('CG', 'CD', 'NE', 'CZ')],
        'ASN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
        'ASP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'OD1')],
        'CYS': [('N', 'CA', 'CB', 'SG')],
        'GLN': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'OE1')],
        'GLU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'OE1')],
        'HIS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'ND1')],
        'ILE': [('N', 'CA', 'CB', 'CG1'), ('CA', 'CB', 'CG1', 'CD1')],
        'LEU': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'LYS': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD'),
                ('CB', 'CG', 'CD', 'CE'), ('CG', 'CD', 'CE', 'NZ')],
        'MET': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'SD'),
                ('CB', 'CG', 'SD', 'CE')],
        'PHE': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'PRO': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD')],
        'SER': [('N', 'CA', 'CB', 'OG')],
        'THR': [('N', 'CA', 'CB', 'OG1')],
        'TRP': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'TYR': [('N', 'CA', 'CB', 'CG'), ('CA', 'CB', 'CG', 'CD1')],
        'VAL': [('N', 'CA', 'CB', 'CG1')],
    }

    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_seq = int(line[22:26].strip())

                atoms.append({
                    'index': len(atoms),   # local index (0-based)
                    'name': atom_name,
                    'res_name': res_name,
                    'res_seq': res_seq,
                    'chain': chain_id
                })

    # organize by chain and residue
    chains = {}
    for atom in atoms:
        chain = atom['chain']
        res_seq = atom['res_seq']
        if chain not in chains:
            chains[chain] = {}
        if res_seq not in chains[chain]:
            chains[chain][res_seq] = {'res_name': atom['res_name'], 'atoms': {}}
        chains[chain][res_seq]['atoms'][atom['name']] = atom['index']

    phi_indices, psi_indices, omega_indices, chi_indices = [], [], [], []

    for chain_id, residues in chains.items():
        res_list = sorted(residues.keys())
        for i, res_num in enumerate(res_list):
            curr_res = residues[res_num]
            curr_atoms = curr_res['atoms']
            res_name = curr_res['res_name']

            # φ
            if i > 0:
                prev_atoms = residues[res_list[i - 1]]['atoms']
                if 'C' in prev_atoms and all(a in curr_atoms for a in ('N', 'CA', 'C')):
                    phi_indices.append((
                        prev_atoms['C'], curr_atoms['N'], curr_atoms['CA'], curr_atoms['C']
                    ))

            # ψ
            if i < len(res_list) - 1:
                next_atoms = residues[res_list[i + 1]]['atoms']
                if all(a in curr_atoms for a in ('N', 'CA', 'C')) and 'N' in next_atoms:
                    psi_indices.append((
                        curr_atoms['N'], curr_atoms['CA'], curr_atoms['C'], next_atoms['N']
                    ))

            # ω
            if i < len(res_list) - 1:
                next_atoms = residues[res_list[i + 1]]['atoms']
                if all(a in curr_atoms for a in ('CA', 'C')) and all(a in next_atoms for a in ('N', 'CA')):
                    omega_indices.append((
                        curr_atoms['CA'], curr_atoms['C'], next_atoms['N'], next_atoms['CA']
                    ))

            # χ
            if res_name in CHI_DEFINITIONS:
                for chi_def in CHI_DEFINITIONS[res_name]:
                    if all(a in curr_atoms for a in chi_def):
                        chi_indices.append(tuple(curr_atoms[a] for a in chi_def))

    # remove duplicates while preserving order
    all_dihedrals = phi_indices + psi_indices + omega_indices + chi_indices
    seen, unique_dihedrals = set(), []
    for q in all_dihedrals:
        if q not in seen:
            unique_dihedrals.append(q)
            seen.add(q)

    return unique_dihedrals

def extract_atom_indices(pdb_path):
    """
    Estrae diversi insiemi di indici atomici da un file PDB.

    Parametri
    ----------
    pdb_path : str
        Percorso al file PDB.

    Restituisce
    ----------
    dict
        Un dizionario con quattro chiavi:
        - "backbone": indici di N, C e CA
        - "nC": indici di N, O e S
        - "alpha": indici di CA
        - "alphabeta": indici di CA e CB
    """
    # Helper interno per evitare di riscrivere il parsing 4 volte
    def parse_pdb_for_atoms(pdb_path, target_atoms):
        indices = []
        with open(pdb_path) as f:
            atom_counter = 0
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                name = line[12:16].strip()
                if name in target_atoms:
                    indices.append(atom_counter)
                atom_counter += 1
        return indices

    backbone = parse_pdb_for_atoms(pdb_path, {"N", "C", "CA"})
    nC = parse_pdb_for_atoms(pdb_path, {"N", "O", "S"})
    alpha = parse_pdb_for_atoms(pdb_path, {"CA"})
    alphabeta = parse_pdb_for_atoms(pdb_path, {"CA", "CB"})

    print(f'Backbone({len(backbone)}): {backbone}')
    print(f'nC({len(nC)}): {nC}')
    print(f'Alpha C ({len(alpha)}): {alpha}')
    print(f'Alpha and Beta ({len(alphabeta)}): {alphabeta}')

    return {
        "backbone": backbone,
        "nC": nC,
        "alpha": alpha,
        "alphabeta": alphabeta,
    }



tfd = tfp.distributions


def create_gaussian_mixture_on_circle(K=10, radius=1.5, scale_std=0.15):
    """
    Crea una distribuzione gaussiana mista con K componenti disposte su un cerchio.

    Parametri
    ----------
    K : int, opzionale
        Numero di componenti nella miscela.
    radius : float, opzionale
        Raggio del cerchio su cui disporre le medie.
    scale_std : float, opzionale
        Deviazione standard (uguale per tutte le componenti) per ogni asse.

    Restituisce
    ----------
    tfd.MixtureSameFamily
        Una distribuzione di tipo MixtureSameFamily con componenti gaussiane 2D.
    """
    # --- means on a circle (float32) ---
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False).astype(np.float32)
    means_np = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(np.float32)
    means = tf.constant(means_np, dtype=tf.float32)

    # --- consistent diagonal scales (float32) ---
    scales = tf.fill([K, 2], tf.constant(scale_std, tf.float32))

    # --- uniform categorical weights ---
    mix = tfd.Categorical(logits=tf.zeros([K], dtype=tf.float32))

    # --- Gaussian components ---
    components = tfd.MultivariateNormalDiag(loc=means, scale_diag=scales)

    # --- final mixture distribution ---
    prior = tfd.MixtureSameFamily(mixture_distribution=mix, components_distribution=components)

    return prior

