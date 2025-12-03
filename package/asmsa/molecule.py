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

def _parse_top(top, ndx = None):
    """
    Parses a topology file and extracts atom types, bonds, angles, and dihedrals.
    
    This function reads a molecular topology file in a format similar to GROMACS,
    extracting information about atoms, bonds, angles, and dihedrals. It can 
    optionally filter the data using an index file.
    
    Parameters:
    -----------
    top : str
        Path to the topology file to parse
    ndx : str, optional
        Path to an index file for filtering atoms (typically to select non-hydrogen atoms)
        
    Returns:
    --------
    tuple
        A 5-tuple containing:
        - list of atom types (strings)
        - numpy array of bonds (pairs of atom indices)
        - numpy array of angles (triplets of atom indices)
        - numpy array of type 4 dihedrals (quadruplets of atom indices)
        - numpy array of type 9 dihedrals (quadruplets of atom indices)
    """
    anums = []
    types = []
    bonds = []
    angles = []
    dihedrals = []
    sect = 'unknown'
    with open(top) as tf:
        for l in tf:
            if re.match('\s*[;#]',l) or re.match('\s*$',l): continue
            m = re.match('\[\s*(\w+)\s*\]',l)
            if m:
                sect = m.group(1)
                continue
            elif sect == 'atoms':
                m = re.match('\s*(\d+)\s+(\w+)',l)
                if m:
                    anums.append(int(m.group(1)))
                    types.append(m.group(2))
                    continue
            elif sect == 'bonds':
                m = re.match('\s*(\d+)\s+(\d+)',l)
                if m:
                    bonds.append((int(m.group(1)),int(m.group(2))))
                    continue
            elif sect == 'angles':
                m = re.match('\s*(\d+)\s+(\d+)\s+(\d+)',l)
                if m:
                    angles.append((int(m.group(1)),int(m.group(2)),int(m.group(3))))
                    continue
            elif sect == 'dihedrals':
                m = re.match('\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',l)
                if m:
                    dihedrals.append((int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4)),int(m.group(5))))
                    continue
            else: continue
                    
            log.error(f"unexpected in [{sect}]: {l}")
    
    aidx = [ -1 ] * (max(anums) + 1)
    for i,n in enumerate(anums):
        aidx[n] = i


    bonds = map(lambda b: [aidx[b[0]], aidx[b[1]]], bonds)
    angles = map(lambda a: [aidx[a[0]], aidx[a[1]], aidx[a[2]]], angles)
    dihed4 = map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]],
                                  filter(lambda d: d[4] == 4, dihedrals)
                              )
    dihed9 = map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]],
                                  filter(lambda d: d[4] == 9, dihedrals)
                              )

    if ndx:
        # Read index file to filter atoms (typically to remove hydrogens)
        with open(ndx) as f:
            f.readline()
            ndxs = " ".join(f)

        idx = np.fromstring(ndxs,dtype=int,sep=' ')
        idx -= 1

        # Filter out hydrogens and apply the index
        isheavy = [ False ] * len(types)
        sqidx = [ -1 ] * len(types)
        hnum = 0
        for i,t in enumerate(types):
            if types[i][0] != 'H':
                isheavy[i] = True
                sqidx[i] = idx[hnum]
                hnum += 1

        types = list(filter(lambda t: t[0] != 'H', types))
        ntypes = ['*'] * len(types)
        for i,t in enumerate(types):
            ntypes[idx[i]] = t
        types = ntypes
        
        bonds = map(lambda b: [sqidx[b[0]],sqidx[b[1]]],
                            filter(lambda b: isheavy[b[0]] and isheavy[b[1]], bonds)
                        )
        angles = map(lambda a: [sqidx[a[0]],sqidx[a[1]],sqidx[a[2]]],
                            filter(lambda a: isheavy[a[0]] and isheavy[a[1]] and isheavy[a[2]], angles)
                        )
        dihed4 = map(lambda d: [sqidx[d[0]],sqidx[d[1]],sqidx[d[2]],sqidx[d[3]]],
                            filter(lambda d: isheavy[d[0]] and isheavy[d[1]] and isheavy[d[2]] and isheavy[d[3]], dihed4)
                        )
        dihed9 = map(lambda d: [sqidx[d[0]],sqidx[d[1]],sqidx[d[2]],sqidx[d[3]]],
                            filter(lambda d: isheavy[d[0]] and isheavy[d[1]] and isheavy[d[2]] and isheavy[d[3]], dihed9)
                        )


    return (types,
        np.array(list(bonds),dtype=np.int32),
        np.array(list(angles),dtype=np.int32),
        np.array(list(dihed4),dtype=np.int32),
        np.array(list(dihed9),dtype=np.int32)
    )


def _parse_ff(itpfile):
    """
    Parses a force field parameter file to extract bond, angle, and dihedral parameters.
    
    This function reads a force field file in GROMACS .itp format and extracts 
    parameters for bonds, angles, and dihedrals that will be used for molecular modeling.
    
    Parameters:
    -----------
    itpfile : str
        Path to the force field parameter file
        
    Returns:
    --------
    tuple
        A 4-tuple containing:
        - list of bond types with parameters (atom1_type, atom2_type, equilibrium_length, force_constant)
        - list of angle types with parameters (atom1_type, atom2_type, atom3_type, equilibrium_angle, force_constant)
        - list of type 4 dihedral parameters (atom1_type, atom2_type, atom3_type, atom4_type, phase, force_constant)
        - list of type 9 dihedral parameters (atom1_type, atom2_type, atom3_type, atom4_type, phase, force_constant)
    """
    btypes = []
    atypes = []
    d4types = []
    d9types = []

    sect = 'unknown'
    with open(itpfile) as itp:
        for l in itp:
            if re.match('\s*[;#]',l) or re.match('\s*$',l): continue
            m = re.match('\[\s*(\S+)\s*\]',l)
            if m:
                sect = m.group(1)
                continue
            elif sect == 'bondtypes':
                m = re.match('\s*(\S+)\s+(\S+)\s+\d+\s+(\S+)\s+(\S+)',l)
                if m:
                    btypes.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
                    continue
            elif sect == 'angletypes':
                m = re.match('\s*(\S+)\s+(\S+)\s+(\S+)\s+\d+\s+(\S+)\s+(\S+)',l)
                if m:
                    atypes.append((m.group(1), m.group(2), m.group(3), float(m.group(4)), float(m.group(5))))
                    continue
            elif sect == 'dihedraltypes':
                m = re.match('\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)',l)
                if m:
                    if m.group(5) == '4':
                        d4types.append((m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(6)), float(m.group(7))))
                    elif m.group(5) == '9':
                        d9types.append((m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(6)), float(m.group(7))))
                    continue
            else: continue

            log.error(f"unexpected in [{sect}]: {l}")
    
    return (btypes, atypes, d4types, d9types)


def _match_type(atom, pattern):
    """
    Determines if an atom type matches a pattern from the force field.
    
    This function is used to match atom types from the molecule with patterns
    from the force field. It supports exact matches, wildcard matches (if pattern
    has '*' as second character), and the generic 'X' atom type.
    
    Parameters:
    -----------
    atom : str
        The atom type from the molecule
    pattern : str
        The atom type pattern from the force field
        
    Returns:
    --------
    bool
        True if the atom matches the pattern, False otherwise
    """
    return atom == pattern or (len(pattern) > 1 and atom[0] == pattern[0] and pattern[1] == '*') or pattern == 'X'


class Molecule:
    """
    A class representing a molecule with its topology and force field parameters.
    
    This class handles the molecular structure, including atoms, bonds, angles, and
    dihedrals. It can parse topology and force field files to create a complete
    molecular model that supports energy calculations and internal coordinate
    transformations.
    
    Parameters:
    -----------
    pdb : str, optional
        Path to a PDB file containing the molecular structure
    top : str, optional
        Path to a topology file describing the molecular connectivity
    ndx : str, optional
        Path to an index file for atom selection
    ff : str, optional
        Path to a force field parameter file
    fms : list, optional
        List of feature maps to use in the molecular model
    n_atoms : int, optional
        Number of atoms (required if top is not provided)
    bonds : numpy.ndarray, optional
        Array of bond pairs (required if top is not provided)
    angles : numpy.ndarray, optional
        Array of angle triplets
    diheds : numpy.ndarray, optional
        Array of dihedral quadruplets
    """

    # ff = os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp',
    def __init__(self, pdb=None, top=None, ndx=None, ff=None, fms=[], n_atoms=None,
            bonds=None, angles=None, diheds=None):

        if not top and not n_atoms:
            raise ValueError("`top` or `n_atoms` must be provided")

        if top is not None and (bonds is not None or angles is not None or diheds is not None):
            raise ValueError("Either `top` or `bonds/angles/diheds` can be specified, not both")

        if top is None and fms is None and bonds is None and angles is None and diheds is None:
            raise ValueError("Without `top`, at least one of `fms/bonds/angles/diheds` must be provided")

        self.angles_th0 = None
        self.angles = None
        self.bonds = None
        self.dihed4 = None
        self.dihed9 = None

        if top:
            if ndx:
                if not pdb:
                    raise ValueError("PDB required with index")

                self.ref = md.load_pdb(pdb)
                hs = self.ref[0].top.select("element == H")
                if hs:
                    raise ValueError("Reindexing not reliable with explicit hydrogens")
                
            self.atypes, self.bonds, self.angles, self.dihed4, self.dihed9 = _parse_top(top, ndx)
            self.bonds = np.unique(self.bonds, axis=0)
            self.angles = np.unique(self.angles, axis=0)
            self.dihed4 = np.unique(self.dihed4, axis=0)
            self.dihed9 = np.unique(self.dihed9, axis=0)

            if (ff):
              btypes, atypes, d4types, d9types = _parse_ff(ff)
              self._match_bonds(btypes)
              self._match_angles(atypes)
              self._match_dihed(d4types, d9types)

        else: # not top
            if bonds is not None: self.bonds = np.unique(bonds, axis=0)
            if angles is not None: self.angles = np.unique(angles, axis=0)
            if diheds is not None: self.dihed4 = np.unique(diheds, axis=0)

        self.fms = fms

        # Initialize the molecular model with the parsed data
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
        Matches bonds in the molecule to bond types in the force field.
        
        For each bond in the molecule, this method finds a matching bond type
        in the force field and assigns equilibrium bond lengths and force constants.
        
        Parameters:
        -----------
        btypes : list
            List of bond types from the force field, each containing
            (atom1_type, atom2_type, equilibrium_length, force_constant)
        """
        self.bonds_b0 = np.empty(self.bonds.shape[0], dtype=np.float32)
        self.bonds_kb = np.empty(self.bonds.shape[0], dtype=np.float32)

        for i in range(self.bonds.shape[0]):
            matched = False
            for b in btypes:
                t0 = self.atypes[self.bonds[i,0]]
                t1 = self.atypes[self.bonds[i,1]]
                if ((_match_type(t0, b[0]) and _match_type(t1, b[1]))
                        or (_match_type(t0, b[1]) and _match_type(t1, b[0]))):
                    self.bonds_b0[i] = b[2]
                    self.bonds_kb[i] = b[3]
                    matched = True
                    break   # first match only
            if not matched:
                self.bonds_b0[i] = np.nan
                self.bonds_kb[i] = np.nan
                log.warn(f"bond {i} ({self.bonds[i]}) unmatched")


    def _match_angles(self, atypes):
        """
        Matches angles in the molecule to angle types in the force field.
        
        For each angle in the molecule, this method finds a matching angle type
        in the force field and assigns equilibrium angle values and force constants.
        It also precomputes some values used in energy calculations.
        
        Parameters:
        -----------
        atypes : list
            List of angle types from the force field, each containing
            (atom1_type, atom2_type, atom3_type, equilibrium_angle, force_constant)
        """
        self.angles_th0 = np.empty(self.angles.shape[0], dtype=np.float32)
        self.angles_cth = np.empty(self.angles.shape[0], dtype=np.float32)

        for i in range(self.angles.shape[0]):
            matched = False
            for a in atypes:
                t0 = self.atypes[self.angles[i,0]]
                t1 = self.atypes[self.angles[i,1]]
                t2 = self.atypes[self.angles[i,2]]
                if ((_match_type(t0, a[0]) and _match_type(t1, a[1]) and _match_type(t2, a[2]))
                        or (_match_type(t0, a[2]) and _match_type(t1, a[1]) and _match_type(t2, a[0]))):
                    self.angles_th0[i] = a[3] / 180. * np.pi
                    self.angles_cth[i] = a[4]
                    matched = True
                    break # first match only
            if not matched:
                self.angles_th0[i] = np.nan
                self.angles_cth[i] = np.nan
                log.warn(f"angle {i} ({self.angles[i]}) unmatched")

        # Precompute reciprocal values for efficiency
        self.angles_2rth0 = 2. * np.reciprocal(self.angles_th0)
        


    def _match_dihed(self, d4types, d9types):
        """
        Matches dihedrals in the molecule to dihedral types in the force field.
        
        For each dihedral in the molecule, this method finds a matching dihedral type
        in the force field and assigns phase and force constant values.
        
        Parameters:
        -----------
        d4types : list
            List of type 4 dihedral parameters from the force field
        d9types : list
            List of type 9 dihedral parameters from the force field
        """
        self.dihed9_phase = np.empty(self.dihed9.shape[0], dtype=np.float32)
        self.dihed9_kd = np.empty(self.dihed9.shape[0], dtype=np.float32)

        # XXX: type4 are not matched to FF, they are always included if present in topology

        for i in range(self.dihed9.shape[0]):
            matched = False
            for d in d9types:
                t0 = self.atypes[self.dihed9[i,0]]
                t1 = self.atypes[self.dihed9[i,1]]
                t2 = self.atypes[self.dihed9[i,2]]
                t3 = self.atypes[self.dihed9[i,3]]

                if ((_match_type(t0, d[0]) and _match_type(t1, d[1]) and _match_type(t2, d[2]) and _match_type(t3, d[3])) 
                        or (_match_type(t0, d[3]) and _match_type(t1, d[2]) and _match_type(t2, d[1]) and _match_type(t3, d[0]))):
                    self.dihed9_phase[i] = d[4] / 180. * np.pi
                    self.dihed9_kd[i] = d[5]
                    matched = True
                    break
            if not matched:
                self.dihed9_phase[i] = np.nan
                self.dihed9_kd[i] = np.nan
                log.warn(f"dihed9 {i} ({self.dihed9[i]}) unmatched")



# ....

    """
    bonds[:,2] 
    bonds_b0[:]
    angles[:,3]
    angles_th0[:]   (rad)
    angles_2rth0[:] (precomputed 2 * reciprocal th0)
    dihed4[:,4]
    dihed4_phase[:]
    dihed4_kd[:]
    dihed4_pn[:]
    dihed9[:,4]
    dihed9_phase[:]
    dihed9_kd[:]
    dihed9_pn[:]


    nb = asmsa.NonBond(self) ... ic(geoms)
    """

    def describe_features(self):
        """
        Prints a summary of the molecular features.
        
        This method outputs the count of bonds, angles, and dihedrals
        in the molecular structure.
        """
        print(f"bonds: {len(self.bonds)}")
        print(f"angles: {len(self.angles)}")
        print(f"dihed4: {len(self.dihed4)}")
        print(f"dihed9: {len(self.dihed9)}")


# XXX: unmatched bonds/angles/dihedrals (i.e. nans in their properties) are not handled yet

    def get_model(self):
        """
        Returns the molecular model.
        
        Returns:
        --------
        MoleculeModel
            The molecular model object associated with this molecule
        """
        return self.model

# geoms[atom][xyz][conf]
    def intcoord(self, geoms):
        """
        Computes internal coordinates from Cartesian coordinates.
        
        This method transforms Cartesian coordinates into internal coordinates
        (bonds, angles, dihedrals) using the molecular model.
        
        Parameters:
        -----------
        geoms : numpy.ndarray
            Array of atomic coordinates with shape (n_atoms, 3, n_conformations)
            
        Returns:
        --------
        numpy.ndarray
            Array of internal coordinates
        """
        return np.array(self.model(torch.tensor(geoms)))


if __name__ == '__main__':
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