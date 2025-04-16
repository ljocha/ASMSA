import mdtraj as md
import numpy as np
import re
import os.path
import logging
import torch

from networkx.generators import chordal_cycle_graph
from networkx.generators.classic import complete_graph
import networkx as nx

from .mol_model import MoleculeModel


def _parse_top(top, ndx=None):
    """
    Parse a GROMACS-style topology file (top) and an optional index file (ndx).
    
    This function extracts:
      - atom indices and types
      - bonded interactions (bonds, angles, dihedrals)
    
    It can also reorder or filter out hydrogens if an index file is provided.

    Args:
        top (str): Path to the .top file.
        ndx (str, optional): Path to the index file.
    
    Returns:
        tuple:
            - list[str]: Atom types (filtered if 'ndx' is used).
            - np.ndarray: Bond pairs.
            - np.ndarray: Angle triplets.
            - np.ndarray: Dihedral quadruples (type 4).
            - np.ndarray: Dihedral quadruples (type 9).
    """
    anums = []
    types = []
    bonds = []
    angles = []
    dihedrals = []
    sect = 'unknown'
    
    # Read the topology line by line
    with open(top) as tf:
        for l in tf:
            # Skip comments or empty lines
            if re.match(r'\s*[;#]', l) or re.match(r'\s*$', l):
                continue

            # Check for section headers in the .top file
            m = re.match(r'\[\s*(\w+)\s*\]', l)
            if m:
                sect = m.group(1)
                continue
            elif sect == 'atoms':
                # Example line: "1  C"
                m = re.match(r'\s*(\d+)\s+(\w+)', l)
                if m:
                    anums.append(int(m.group(1)))
                    types.append(m.group(2))
                    continue
            elif sect == 'bonds':
                m = re.match(r'\s*(\d+)\s+(\d+)', l)
                if m:
                    bonds.append((int(m.group(1)), int(m.group(2))))
                    continue
            elif sect == 'angles':
                m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)', l)
                if m:
                    angles.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
                    continue
            elif sect == 'dihedrals':
                # Example line might have 5 numbers, with the last indicating the type (4 or 9)
                m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', l)
                if m:
                    dihedrals.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                      int(m.group(4)), int(m.group(5))))
                    continue
            else:
                # Unrecognized line
                logging.error(f"unexpected in [{sect}]: {l}")

    # Build a map from GROMACS atom numbering to internal indexing
    aidx = [-1] * (max(anums) + 1)
    for i, n in enumerate(anums):
        aidx[n] = i

    # Convert bond/angle/dihedral indices to internal indexing
    bonds = map(lambda b: [aidx[b[0]], aidx[b[1]]], bonds)
    angles = map(lambda a: [aidx[a[0]], aidx[a[1]], aidx[a[2]]], angles)
    dihed4 = map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]],
                 filter(lambda d: d[4] == 4, dihedrals))
    dihed9 = map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]],
                 filter(lambda d: d[4] == 9, dihedrals))

    # If an index file is provided, apply reindexing or filtering (e.g., remove hydrogens)
    if ndx:
        with open(ndx) as f:
            f.readline()  # Skip header line
            ndxs = " ".join(f)

        idx = np.fromstring(ndxs, dtype=int, sep=' ')
        idx -= 1  # Shift to 0-based indexing

        # Mark and re-map non-hydrogen atoms
        isheavy = [False] * len(types)
        sqidx = [-1] * len(types)
        hnum = 0
        for i, t in enumerate(types):
            if t[0] != 'H':
                isheavy[i] = True
                sqidx[i] = idx[hnum]
                hnum += 1

        # Keep only non-hydrogen types
        types = list(filter(lambda t: t[0] != 'H', types))
        ntypes = ['*'] * len(types)
        for i, t in enumerate(types):
            ntypes[idx[i]] = t
        types = ntypes

        # Remap bonds, angles, dihedrals (exclude those with hydrogens)
        bonds = map(
            lambda b: [sqidx[b[0]], sqidx[b[1]]],
            filter(lambda b: isheavy[b[0]] and isheavy[b[1]], bonds)
        )
        angles = map(
            lambda a: [sqidx[a[0]], sqidx[a[1]], sqidx[a[2]]],
            filter(lambda a: isheavy[a[0]] and isheavy[a[1]] and isheavy[a[2]], angles)
        )
        dihed4 = map(
            lambda d: [sqidx[d[0]], sqidx[d[1]], sqidx[d[2]], sqidx[d[3]]],
            filter(lambda d: isheavy[d[0]] and isheavy[d[1]] and isheavy[d[2]] and isheavy[d[3]], dihed4)
        )
        dihed9 = map(
            lambda d: [sqidx[d[0]], sqidx[d[1]], sqidx[d[2]], sqidx[d[3]]],
            filter(lambda d: isheavy[d[0]] and isheavy[d[1]] and isheavy[d[2]] and isheavy[d[3]], dihed9)
        )

    return (
        types,
        np.array(list(bonds), dtype=np.int32),
        np.array(list(angles), dtype=np.int32),
        np.array(list(dihed4), dtype=np.int32),
        np.array(list(dihed9), dtype=np.int32)
    )


def _parse_ff(itpfile):
    """
    Parse a force-field file (itp) to extract bond, angle, and dihedral parameters.
    
    Args:
        itpfile (str): Path to the .itp file.

    Returns:
        tuple:
            - list: Bond type definitions (btypes).
            - list: Angle type definitions (atypes).
            - list: Dihedral type 4 definitions (d4types).
            - list: Dihedral type 9 definitions (d9types).
    """
    btypes = []
    atypes = []
    d4types = []
    d9types = []

    sect = 'unknown'
    with open(itpfile) as itp:
        for l in itp:
            if re.match(r'\s*[;#]', l) or re.match(r'\s*$', l):
                continue
            m = re.match(r'\[\s*(\S+)\s*\]', l)
            if m:
                sect = m.group(1)
                continue
            elif sect == 'bondtypes':
                # Example line: "type1 type2 <int> length kb"
                m = re.match(r'\s*(\S+)\s+(\S+)\s+\d+\s+(\S+)\s+(\S+)', l)
                if m:
                    btypes.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
                    continue
            elif sect == 'angletypes':
                # Example line: "type1 type2 type3 <int> angle force"
                m = re.match(r'\s*(\S+)\s+(\S+)\s+(\S+)\s+\d+\s+(\S+)\s+(\S+)', l)
                if m:
                    atypes.append((m.group(1), m.group(2), m.group(3),
                                   float(m.group(4)), float(m.group(5))))
                    continue
            elif sect == 'dihedraltypes':
                # Example line: "type1 type2 type3 type4 <int> phase force"
                m = re.match(r'\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)', l)
                if m:
                    if m.group(5) == '4':
                        d4types.append((m.group(1), m.group(2), m.group(3), m.group(4),
                                        float(m.group(6)), float(m.group(7))))
                    elif m.group(5) == '9':
                        d9types.append((m.group(1), m.group(2), m.group(3), m.group(4),
                                        float(m.group(6)), float(m.group(7))))
                    continue
            else:
                logging.error(f"unexpected in [{sect}]: {l}")

    return (btypes, atypes, d4types, d9types)


def _match_type(atom, pattern):
    """
    Checks if 'atom' matches a particular 'pattern' based on:
      - exact match
      - wildcard pattern (e.g., C* matches anything starting with 'C')
      - or 'X' as a universal match
    """
    return (
        atom == pattern
        or (len(pattern) > 1 and atom[0] == pattern[0] and pattern[1] == '*')
        or pattern == 'X'
    )


class Molecule:
    """
    High-level molecule handler that reads topology and force-field data, matches
    parameters, and constructs a 'MoleculeModel' for internal coordinate computations.

    Args:
        pdb (str, optional): Path to a PDB file (needed if using an index file).
        top (str, optional): Path to a .top file for bonds/angles/dihedrals.
        ndx (str, optional): Path to an index file for reindexing or filtering atoms.
        ff (str, optional): Path to a .itp file with force-field data.
        fms (list, optional): Additional feature maps to apply in the MoleculeModel.
        n_atoms (int, optional): Number of atoms (if no .top file is used).
        bonds, angles, diheds (array-like, optional): Custom definitions if .top is not used.
    """
    def __init__(
        self,
        pdb=None,
        top=None,
        ndx=None,
        ff=None,
        fms=[],
        n_atoms=None,
        bonds=None,
        angles=None,
        diheds=None
    ):
        if not top and not n_atoms:
            raise ValueError("`top` or `n_atoms` must be provided")

        if top is not None and (bonds is not None or angles is not None or diheds is not None):
            raise ValueError("Specify either `top` or manual (bonds/angles/diheds), not both")

        if top is None and fms is None and bonds is None and angles is None and diheds is None:
            raise ValueError("Without `top`, at least one of `fms/bonds/angles/diheds` must be provided")

        self.angles_th0 = None
        self.angles = None
        self.bonds = None
        self.dihed4 = None
        self.dihed9 = None

        # Parse the topology if provided
        if top:
            if ndx:
                # If we have an index file, we expect a matching PDB
                if not pdb:
                    raise ValueError("PDB required with index")

                self.ref = md.load_pdb(pdb)
                hs = self.ref[0].top.select("element == H")
                if hs:
                    raise ValueError("Reindexing not reliable with explicit hydrogens")

            self.atypes, self.bonds, self.angles, self.dihed4, self.dihed9 = _parse_top(top, ndx)

            # Remove duplicates from each interaction type
            self.bonds = np.unique(self.bonds, axis=0)
            self.angles = np.unique(self.angles, axis=0)
            self.dihed4 = np.unique(self.dihed4, axis=0)
            self.dihed9 = np.unique(self.dihed9, axis=0)

            # Match them against the force-field if provided
            if ff:
                btypes, atypes, d4types, d9types = _parse_ff(ff)
                self._match_bonds(btypes)
                self._match_angles(atypes)
                self._match_dihed(d4types, d9types)
        else:
            # No .top file, just use what's given
            self.bonds = np.unique(bonds, axis=0) if bonds is not None else []
            self.angles = np.unique(angles, axis=0) if angles is not None else []
            self.dihed4 = np.unique(diheds, axis=0) if diheds is not None else []
            self.atypes = ['*'] * n_atoms  # placeholder if no .top

        self.fms = fms

        # Create the MoleculeModel (PyTorch)
        self.model = MoleculeModel(
            len(self.atypes) if top else n_atoms,
            bonds=self.bonds,
            angles=self.angles,
            angles_th0=self.angles_th0,
            dihed4=self.dihed4,
            dihed9=self.dihed9,
            feature_maps=self.fms
        )

    def _match_bonds(self, btypes):
        """
        Match each bond in 'self.bonds' with its force-field parameters in 'btypes'.
        """
        self.bonds_b0 = np.empty(self.bonds.shape[0], dtype=np.float32)
        self.bonds_kb = np.empty(self.bonds.shape[0], dtype=np.float32)

        for i in range(self.bonds.shape[0]):
            matched = False
            for b in btypes:
                t0 = self.atypes[self.bonds[i, 0]]
                t1 = self.atypes[self.bonds[i, 1]]
                if ((_match_type(t0, b[0]) and _match_type(t1, b[1])) or
                        (_match_type(t0, b[1]) and _match_type(t1, b[0]))):
                    self.bonds_b0[i] = b[2]
                    self.bonds_kb[i] = b[3]
                    matched = True
                    break
            if not matched:
                self.bonds_b0[i] = np.nan
                self.bonds_kb[i] = np.nan
                logging.warning(f"bond {i} ({self.bonds[i]}) unmatched")

    def _match_angles(self, atypes):
        """
        Match each angle in 'self.angles' with its force-field parameters in 'atypes'.
        """
        self.angles_th0 = np.empty(self.angles.shape[0], dtype=np.float32)
        self.angles_cth = np.empty(self.angles.shape[0], dtype=np.float32)

        for i in range(self.angles.shape[0]):
            matched = False
            for a in atypes:
                t0 = self.atypes[self.angles[i, 0]]
                t1 = self.atypes[self.angles[i, 1]]
                t2 = self.atypes[self.angles[i, 2]]
                if ((_match_type(t0, a[0]) and _match_type(t1, a[1]) and _match_type(t2, a[2])) or
                        (_match_type(t0, a[2]) and _match_type(t1, a[1]) and _match_type(t2, a[0]))):
                    self.angles_th0[i] = a[3] / 180.0 * np.pi
                    self.angles_cth[i] = a[4]
                    matched = True
                    break
            if not matched:
                self.angles_th0[i] = np.nan
                self.angles_cth[i] = np.nan
                logging.warning(f"angle {i} ({self.angles[i]}) unmatched")

        self.angles_2rth0 = 2.0 * np.reciprocal(self.angles_th0)

    def _match_dihed(self, d4types, d9types):
        """
        Match dihedral type 9 in 'self.dihed9' with force-field parameters in 'd9types'.
        (Type 4 dihedrals are not matched here, but are included if present in the topology.)
        """
        self.dihed9_phase = np.empty(self.dihed9.shape[0], dtype=np.float32)
        self.dihed9_kd = np.empty(self.dihed9.shape[0], dtype=np.float32)

        for i in range(self.dihed9.shape[0]):
            matched = False
            for d in d9types:
                t0 = self.atypes[self.dihed9[i, 0]]
                t1 = self.atypes[self.dihed9[i, 1]]
                t2 = self.atypes[self.dihed9[i, 2]]
                t3 = self.atypes[self.dihed9[i, 3]]
                if ((_match_type(t0, d[0]) and _match_type(t1, d[1]) and
                     _match_type(t2, d[2]) and _match_type(t3, d[3])) or
                        (_match_type(t0, d[3]) and _match_type(t1, d[2]) and
                         _match_type(t2, d[1]) and _match_type(t3, d[0]))):
                    self.dihed9_phase[i] = d[4] / 180.0 * np.pi
                    self.dihed9_kd[i] = d[5]
                    matched = True
                    break
            if not matched:
                self.dihed9_phase[i] = np.nan
                self.dihed9_kd[i] = np.nan
                logging.warning(f"dihed9 {i} ({self.dihed9[i]}) unmatched")

    def describe_features(self):
        """
        Print the number of bonds, angles, and dihedrals in this molecule.
        """
        print(f"bonds: {len(self.bonds)}")
        print(f"angles: {len(self.angles)}")
        print(f"dihed4: {len(self.dihed4)}")
        print(f"dihed9: {len(self.dihed9)}")

    def get_model(self):
        """
        Returns the internal MoleculeModel (PyTorch module) used for computing
        internal coordinates.
        """
        return self.model

    def intcoord(self, geoms):
        """
        Compute the molecule's internal coordinates (bonds, angles, dihedrals, etc.)
        for a set of geometries.

        Args:
            geoms (np.ndarray): Atomic coordinates of shape [n_atoms, 3, ...].
        
        Returns:
            np.ndarray: The computed internal coordinates as a NumPy array.
        """
        return np.array(self.model(torch.tensor(geoms)))


if __name__ == '__main__':
    # Example usage to test functionality
    mol = Molecule('alaninedipeptide_H.pdb', 'topol.top')
    print(mol.atypes)
    print(mol.bonds)
    print(mol.angles)
    print(mol.dihed4)
    print(mol.dihed9)

    ff = os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp'
    btypes, atypes, d4types, d9types = _parse_ff(ff)
    print(d9types)

    print(mol.bonds_b0)
    print(mol.angles_th0)
