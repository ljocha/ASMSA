#! vim: expandtabs ts=4 ai:

import torch
import numpy as np
import tensorflow as tf

from networkx.generators import chordal_cycle_graph
from networkx.generators.classic import complete_graph
import networkx as nx
from sympy import nextprime

"""
This module defines PyTorch models to compute basic molecular geometry features:
bonds, angles, dihedrals, and non-bonded distances. It also provides a MoleculeModel
that combines these features into a single output.
"""

class BondsModel(torch.nn.Module):
    """
    Computes bond lengths between pairs of atoms.

    Args:
        n_atoms (int): Number of atoms (for reference).
        bonds (array-like): Pairs of atom indices (shape: [num_bonds, 2]).
    """
    def __init__(self, n_atoms, bonds):
        super().__init__()
        self.n_atoms = n_atoms
        self.bonds = np.array(bonds).reshape(-1, 2)

    def forward(self, geoms):
        """
        Args:
            geoms (torch.Tensor): Coordinates of shape (n_atoms, 3, ...).
                                  Each row is an (x, y, z) coordinate.

        Returns:
            torch.Tensor: Bond lengths for each pair in 'self.bonds'.
                          Shape: (num_bonds, ...) depending on extra dimensions.
        """
        diffs = geoms[self.bonds[:, 0]] - geoms[self.bonds[:, 1]]
        return torch.linalg.norm(diffs, axis=1)


class AnglesModel(torch.nn.Module):
    """
    Computes angles (in either cos() form or mapped around a reference angle).

    Args:
        n_atoms (int): Number of atoms.
        angles (array-like): Triplets of atom indices (shape: [num_angles, 3]).
        angles_th0 (array-like, optional): Preferred angles for each triplet.
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
        """
        Args:
            geoms (torch.Tensor): Coordinates of shape (n_atoms, 3, ...).

        Returns:
            torch.Tensor: Angles in one of two forms:
              - If angles_th0 is set, returns angles mapped around the reference.
              - Otherwise returns cos(theta).
        """
        v1 = geoms[self.angles[:, 0]] - geoms[self.angles[:, 1]]
        v2 = geoms[self.angles[:, 2]] - geoms[self.angles[:, 1]]
        n1 = torch.linalg.norm(v1, axis=1)
        n2 = torch.linalg.norm(v2, axis=1)
        dot = torch.sum(v1 * v2, axis=1) / (n1 * n2)

        if self.angles_th0:
            aa = torch.arccos(dot * 0.999999)  # help numerical stability
            return (aa - 0.75 * self.angles_th0[:, None]) * self.angles_2rth0[:, None]
        else:
            return dot


class DihedralModel(torch.nn.Module):
    """
    Computes dihedral angles for quadruples of atoms.

    Args:
        n_atoms (int): Number of atoms.
        atoms (array-like): Quadruples of atom indices (shape: [num_dihedrals, 4]).
    """
    def __init__(self, n_atoms, atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.atoms = np.array(atoms).reshape(-1, 4)

    def forward(self, geoms):
        """
        Args:
            geoms (torch.Tensor): Coordinates of shape (n_atoms, 3, ...).

        Returns:
            torch.Tensor: A stack of sine and cosine components for each
                          dihedral angle (shape: 2*num_dihedrals, ...).
        """
        a12 = geoms[self.atoms[:, 1]] - geoms[self.atoms[:, 0]]
        a23 = geoms[self.atoms[:, 2]] - geoms[self.atoms[:, 1]]
        a34 = geoms[self.atoms[:, 3]] - geoms[self.atoms[:, 2]]

        vp1 = torch.nn.functional.normalize(torch.cross(a12, a23, axis=1))
        vp2 = torch.nn.functional.normalize(torch.cross(a23, a34, axis=1))
        vp3 = torch.nn.functional.normalize(torch.cross(vp1, a23, axis=1))

        sp1 = torch.sum(vp1 * vp2, axis=1)
        sp2 = torch.sum(vp3 * vp2, axis=1)

        # Return an angle representation (two values per angle)
        return torch.stack([-sp2, sp1]).reshape(2 * len(self.atoms), geoms.shape[2])


class NBDistancesSparse(BondsModel):
    """
    Computes sparse non-bonded distances using a chordal cycle graph pattern.

    Args:
        all_atoms (int): Total number of atoms in the system.
        density (int): A parameter controlling how many edges to generate.
        atoms (list, optional): Which atom indices to use. If None, uses all_atoms.
    """
    def __init__(self, all_atoms, density=1, atoms=None):
        if atoms is None:
            atoms = list(range(all_atoms))

        used_atoms = len(atoms)
        p = nextprime(used_atoms)
        assert (1 <= density < p)

        edges = []
        for i in range(1, density + 1):
            G = chordal_cycle_graph(p)
            G.remove_edges_from(nx.selfloop_edges(G))
            edges += [((a * i) % p, (b * i) % p) for a, b in G.edges()]

        # Remove duplicates and map them to the valid range
        E = np.array(list(set(
            filter(lambda pair: pair[0] != pair[1], [
                tuple(sorted([min(a, used_atoms - 1), min(b, used_atoms - 1)]))
                for a, b in edges
            ])
        )))

        E = [(atoms[a], atoms[b]) for a, b in E]

        super().__init__(all_atoms, np.array(E))


class NBDistancesDense(BondsModel):
    """
    Computes dense non-bonded distances by connecting all pairs of atoms (complete graph).

    Args:
        n_atoms (int): Number of atoms.
    """
    def __init__(self, n_atoms):
        G = complete_graph(n_atoms)
        E = np.array([e for e in G.edges()])
        super().__init__(n_atoms, np.array(E))


class MoleculeModel(torch.nn.Module):
    """
    A container model that can combine:
      - BondsModel
      - AnglesModel
      - DihedralModel (optional for different sets of dihedrals)
      - Custom feature maps

    Args:
        n_atoms (int): Number of atoms.
        bonds (list/array, optional): Bond indices.
        angles (list/array, optional): Angle triplets.
        angles_th0 (list/array, optional): Preferred angles for each triplet.
        dihed4 (list/array, optional): Dihedral quadruples.
        dihed9 (list/array, optional): Another set of dihedrals (if needed).
        feature_maps (list of callables, optional): Additional functions
            that take geoms as input and return a tensor.
    """
    def __init__(self, n_atoms, bonds=[], angles=None, angles_th0=None, dihed4=None, dihed9=None, feature_maps=[]):
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

        # Create sub-models
        self.bonds_model = BondsModel(self.n_atoms, self.bonds) if self.bonds else None
        if self.angles is not None:
            self.angles_model = AnglesModel(self.n_atoms, self.angles, self.angles_th0)
        if self.dihed4 is not None:
            self.dihed4_model = DihedralModel(self.n_atoms, self.dihed4)
        if self.dihed9 is not None:
            self.dihed9_model = DihedralModel(self.n_atoms, self.dihed9)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Coordinates of shape (n_atoms, 3, ...).

        Returns:
            torch.Tensor: Concatenation of bond lengths, angles, dihedrals,
                          and additional feature maps, if available.
        """
        assert input.shape[0] == self.n_atoms
        assert input.shape[1] == 3
        outputs = []

        if self.bonds_model:
            outputs.append(self.bonds_model(input))
        if self.angles_model:
            outputs.append(self.angles_model(input))
        if self.dihed4_model:
            outputs.append(self.dihed4_model(input))
        if self.dihed9_model:
            outputs.append(self.dihed9_model(input))

        # Append optional feature maps
        if self.feature_maps:
            outputs += [fm(input) for fm in self.feature_maps]

        return torch.cat(outputs, axis=0)

    def get_indices(self):
        """
        Returns a dictionary with the start/end positions of each feature type
        in the output tensor. This is useful to locate where bonds, angles, and
        dihedrals appear in the concatenated output.

        Returns:
            dict: Keys are 'bonds', 'angles', 'dihed4', 'dihed9' mapping to
                  (start_index, end_index) in the final output.
        """
        out = {}
        bl = 0
        al = 0
        d4 = 0
        d9 = 0

        if self.bonds is not None:
            bl = len(self.bonds)
            out['bonds'] = (0, bl)

        if self.angles is not None:
            al = len(self.angles)
            out['angles'] = (bl, bl + al)

        if self.dihed4 is not None:
            d4 = len(self.dihed4)
            out['dihed4'] = (bl + al, bl + al + d4)

        if self.dihed9 is not None:
            d4 = len(self.dihed9)
            out['dihed9'] = (bl + al + d4, bl + al + d4 + d9)

        return out
