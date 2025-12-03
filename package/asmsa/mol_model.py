"""
Molecular Internal Coordinates Module

This module provides PyTorch-based models for representing molecular structures using internal coordinates.
It defines classes for calculating various molecular features like bond lengths, angles, and dihedral angles
from atomic coordinates, which are essential for molecular modeling and simulation.
"""

import torch
import numpy as np
import tensorflow as tf

from networkx.generators import chordal_cycle_graph
from networkx.generators.classic import complete_graph
import networkx as nx
from sympy import nextprime


def _norm_axis_1(x):
    return torch.sqrt(torch.sum(x*x,dim=1))


"""
Base molecule features - internal coordinates
"""

class BondsModel(torch.nn.Module):
    """
    Calculates bond lengths between pairs of atoms.
    
    Parameters:
    -----------
    n_atoms : int
        Total number of atoms in the molecule.
    bonds : list or array
        List of atom index pairs (i,j) representing bonds between atoms.
        
    Methods:
    --------
    forward(geoms)
        Calculates the Euclidean distances between bonded atoms.
        
    Arguments:
        geoms: Tensor of shape [n_atoms, 3, n_conformations] representing atomic coordinates.
        
    Returns:
        Tensor of bond lengths for each conformation.
    """
    def __init__(self, n_atoms, bonds):
        super().__init__()
        self.n_atoms = n_atoms
        self.bonds = np.array(bonds).reshape(-1, 2)

    def forward(self, geoms):
#        geoms = input.reshape(self.n_atoms, 3, -1)
        diffs = geoms[self.bonds[:, 0]] - geoms[self.bonds[:, 1]]
#        return torch.linalg.norm(diffs, axis=1)
        return _norm_axis_1(diffs)
        


class AnglesModel(torch.nn.Module):
    """
    Calculates bond angles between triplets of atoms.
    
    Parameters:
    -----------
    n_atoms : int
        Total number of atoms in the molecule.
    angles : list or array
        List of atom index triplets (i,j,k) representing angles centered at atom j.
    angles_th0 : array or None
        Reference/equilibrium values of the angles (in radians) for force field calculations.
        If None, simple cosine of angles is returned.
        
    Methods:
    --------
    forward(geoms)
        Calculates bond angles or normalized deviations from reference angles.
        
    Arguments:
        geoms: Tensor of shape [n_atoms, 3, n_conformations] representing atomic coordinates.
        
    Returns:
        Tensor of angle values (or normalized deviations) for each conformation.
    """
    def __init__(self, n_atoms, angles, angles_th0):
        super().__init__()
        self.n_atoms = n_atoms
        self.angles = np.array(angles).reshape(-1, 3)
        if angles_th0 is not None:
            self.angles_th0 = torch.tensor(angles_th0, requires_grad=False)
            self.angles_2rth0 = 2 * torch.reciprocal(self.angles_th0)
        else:
            self.angles_th0 = None
            self.angles_2rth0 = None

    def forward(self, geoms):
#        geoms = input.reshape(self.n_atoms, 3, -1)
        v1 = geoms[self.angles[:,0]] - geoms[self.angles[:,1]]
        v2 = geoms[self.angles[:,2]] - geoms[self.angles[:,1]]
        n1 = torch.linalg.norm(v1,axis=1)
        n2 = torch.linalg.norm(v2,axis=1)

        dot = torch.sum(v1 * v2, axis=1) / (n1 * n2)

        # if force field was specified, map the angle around it's relaxed value; use just it's cosine otherwise
        if self.angles_th0 is not None:
            aa = torch.arccos(dot * 0.999999) # numerical stability of arccos
            return (aa - .75 * self.angles_th0[:,None]) * self.angles_2rth0[:,None] # map 0.75 a0 -- 1.25 a0 to 0 -- 1
        else:
            return dot


class DihedralModel(torch.nn.Module):
    """
    Calculates dihedral (torsion) angles between quartets of atoms.
    
    Parameters:
    -----------
    n_atoms : int
        Total number of atoms in the molecule.
    atoms : list or array
        List of atom index quartets (i,j,k,l) representing dihedral angles.
        
    Methods:
    --------
    forward(geoms)
        Calculates the sine and cosine components of dihedral angles.
        
    Arguments:
        geoms: Tensor of shape [n_atoms, 3, n_conformations] representing atomic coordinates.
        
    Returns:
        Tensor containing paired sine and cosine values for each dihedral angle and conformation.
    """
    def __init__(self, n_atoms, atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.atoms = np.array(atoms).reshape(-1, 4)

    def forward(self, geoms):
#        geoms = input.reshape(self.n_atoms, 3, -1)
        a12 = geoms[self.atoms[:, 1]] - geoms[self.atoms[:, 0]]
        a23 = geoms[self.atoms[:, 2]] - geoms[self.atoms[:, 1]]
        a34 = geoms[self.atoms[:, 3]] - geoms[self.atoms[:, 2]]

#        a12 = torch.nn.functional.normalize(a12, p=2, dim=1)
#        a23 = torch.nn.functional.normalize(a23, p=2, dim=1)
#        a34 = torch.nn.functional.normalize(a34, p=2, dim=1)

        vp1 = torch.nn.functional.normalize(torch.cross(a12,a23,axis=1))
        vp2 = torch.nn.functional.normalize(torch.cross(a23,a34,axis=1))
        vp3 = torch.nn.functional.normalize(torch.cross(vp1,a23,axis=1))

        sp1 = torch.sum(vp1 * vp2, axis=1)
        sp2 = torch.sum(vp3 * vp2, axis=1)

        """ original:
        # output for i-th dihedral angle
            aa = np.arctan2(sp1,sp2) - np.pi * .5
            return np.sin(aa), np.cos(aa)
        """

        #NOTE: Why adding two variables that determine each other? It the angle better?
        # return torch.nn.functional.normalize(torch.stack([-sp2, sp1]), p=2, dim=0).reshape(2*len(self.atoms), geoms.shape[2])
        return torch.stack([-sp2, sp1]).reshape(2*len(self.atoms), geoms.shape[2])


"""
Extra molecule features - Non-binding distances
"""

class NBDistancesSparse(BondsModel):
    """
    Calculates a sparse set of non-bonded atom-atom distances.
    
    This class uses a chordal cycle graph to generate a sparse but representative 
    set of atom-atom distances for molecular representation.
    
    Parameters:
    -----------
    all_atoms : int
        Total number of atoms in the molecule.
    density : int
        Controls the density of the distance network (higher values = more connections).
    atoms : list or None
        Subset of atom indices to consider. If None, all atoms are used.
    """
    def __init__(self, all_atoms, density=1, atoms=None):
        if atoms is None:
            atoms = list(range(all_atoms))

        used_atoms = len(atoms)

        p = nextprime(used_atoms)
        assert(1 <= density < p)

        edges = []

        for i in range(1, density + 1):
            G = chordal_cycle_graph(p)
            G.remove_edges_from(nx.selfloop_edges(G))

            edges += [((a*i) % p, (b*i) % p) for a,b in G.edges()]

        E = np.array(list(set(
            filter(lambda p: p[0]!=p[1],[
                tuple(sorted([min(a, used_atoms-1),min(b, used_atoms-1)]))
                for a,b in edges
            ])
        )))

        E = [ (atoms[a],atoms[b]) for a,b in E ]

        super().__init__(all_atoms, np.array(E))


class NBDistancesDense(BondsModel):
    """
    Calculates all possible pairwise atom-atom distances.
    
    This class generates a complete graph of all atom-atom distances.
    
    Parameters:
    -----------
    all_atoms : int
        Total number of atoms in the molecule.
    atoms : list or None
        Subset of atom indices to consider. If None, all atoms are used.
    """
    def __init__(self, all_atoms, atoms=None):
        if atoms is None:
            atoms = list(range(all_atoms))
        
        # Generate a complete graph of the atom subset
        used_atoms = len(atoms)
        G = complete_graph(used_atoms)
        
        # Map the edges back to the original atom indices
        E = np.array([(atoms[a], atoms[b]) for a, b in G.edges()])
        
        super().__init__(all_atoms, np.array(E))


"""
Molecule model
"""

class MoleculeModel(torch.nn.Module):
    """
    Comprehensive model for molecular internal coordinates representation.
    
    Combines multiple feature representations (bonds, angles, dihedrals, etc.)
    into a unified molecular representation.
    
    Parameters:
    -----------
    n_atoms : int
        Total number of atoms in the molecule.
    bonds : list or array, optional
        List of atom index pairs representing bonds.
    angles : list or array, optional
        List of atom index triplets representing angles.
    angles_th0 : array, optional
        Reference values for angles (in radians).
    dihed4 : list or array, optional
        List of atom index quartets for type-4 dihedrals.
    dihed9 : list or array, optional
        List of atom index quartets for type-9 dihedrals.
    feature_maps : list, optional
        Additional feature extraction modules to apply.
        
    Methods:
    --------
    forward(input)
        Calculates all internal coordinate features for the given atomic coordinates.
        
    get_indices()
        Returns a dictionary mapping feature types to their index ranges in the output vector.
    """
    def __init__(self, n_atoms, bonds=None, angles=None, angles_th0=None, dihed4=None, dihed9=None, feature_maps=None):
        super().__init__()
        self.n_atoms = n_atoms
        self.bonds = bonds
        self.angles = angles
        self.angles_th0 = angles_th0
        self.dihed4 = dihed4
        self.dihed9 = dihed9
        self.feature_maps = feature_maps

        self.bonds_model = None
        self.angles_model = None
        self.dihed4_model = None
        self.dihed9_model = None

        if bonds is not None: self.bonds_model = BondsModel(self.n_atoms, self.bonds)
        if angles is not None: self.angles_model = AnglesModel(self.n_atoms, self.angles, self.angles_th0)
        if dihed4 is not None: self.dihed4_model = DihedralModel(self.n_atoms, self.dihed4)
        if dihed9 is not None: self.dihed9_model = DihedralModel(self.n_atoms, self.dihed9)

    def forward(self, input):
        """
        Calculate all internal coordinate features for the given atomic coordinates.
        
        Parameters:
        -----------
        input : Tensor
            Atomic coordinates of shape [n_atoms, 3, n_conformations]
            
        Returns:
        --------
        Tensor
            Concatenated feature vector for all enabled internal coordinates
        """
        assert input.shape[0] == self.n_atoms
        assert input.shape[1] == 3
        outputs = []
        if self.bonds_model: outputs.append(self.bonds_model(input))
        if self.angles_model: outputs.append(self.angles_model(input))
        if self.dihed4_model: outputs.append(self.dihed4_model(input))
        if self.dihed9_model: outputs.append(self.dihed9_model(input))
        if self.feature_maps: outputs += [fm(input) for fm in self.feature_maps]

        return torch.cat(outputs, axis=0)

    def get_indices(self):
        """
        Get indices for the different feature types in the output vector.
        
        Returns:
        --------
        dict
            Dictionary mapping feature types to (start_index, end_index) tuples
        """
        out = dict()
        bl = 0
        al = 0
        d4 = 0
        d9 = 0
        if self.bonds is not None:
            bl=len(self.bonds)
            out['bonds'] = (0,bl)
        if self.angles is not None:
            al=len(self.angles)
            out['angles'] = (bl,bl+al)
        if self.dihed4 is not None:
            d4 = len(self.dihed4)
            out['dihed4'] = (bl+al,bl+al+d4)
        if self.dihed9 is not None:
            d9 = len(self.dihed9)
            out['dihed9'] = (bl+al+d4,bl+al+d4+d9)

        return out
