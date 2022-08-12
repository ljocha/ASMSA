import mdtraj as md
import numpy as np
import re
import os.path
import logging

from networkx.generators import chordal_cycle_graph
from networkx.generators.classic import complete_graph
import networkx as nx

# TODO: replace by sympy.next_prime
_max_prime = 10000
_sieve = np.full(_max_prime+1, True)
_sieve[0] = _sieve[1] = False
for i in range(2, int(np.sqrt(_max_prime+1)) + 2):
    if _sieve[i]:
        _sieve[np.arange(2*i, _max_prime+1, i)] = False
_primes = np.arange(0,_max_prime+1, 1)[_sieve]

def next_prime(n):
	assert(n <= _max_prime)
	return int(_primes[np.searchsorted(_primes, n)])


def _parse_top(top):
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

	return (types,
		np.array(list(map(lambda b: [aidx[b[0]], aidx[b[1]]], bonds)),dtype=np.int32),
		np.array(list(map(lambda a: [aidx[a[0]], aidx[a[1]], aidx[a[2]]], angles)),dtype=np.int32),
		np.array(list(map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]], 
																	filter(lambda d: d[4] == 4, dihedrals)
															)),dtype=np.int32),
		np.array(list(map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]], 
																	filter(lambda d: d[4] == 9, dihedrals)
															)),dtype=np.int32)
	)


def _parse_ff(itpfile):
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


def _match_type(atom,pattern):
	return atom == pattern or (len(pattern) > 1 and atom[0] == pattern[0] and pattern[1] == '*') or pattern == 'X'


class FeatureMap:
	def ic(self, geoms):
		return np.empty([0,geoms.shape[2]],dtype=np.float32)


class NBDistancesSparse(FeatureMap):
	def __init__(self, n_atoms, density=1):
		p = next_prime(n_atoms)
		assert(1 <= density < p)

		edges = []

		for i in range(1, density + 1):
			G = chordal_cycle_graph(p)
			G.remove_edges_from(nx.selfloop_edges(G))

			edges += [((a*i) % p, (b*i) % p) for a,b in G.edges()]

		E = np.array(list(set(
            [tuple(sorted([min(a, n_atoms-1),min(b, n_atoms-1)])) for a,b in edges]
        )))

		self.edges_s = np.array(E)[:, 0]
		self.edges_t = np.array(E)[:, 1]

	def ic(self, geoms):
		dist_vecs = geoms[self.edges_s] - geoms[self.edges_t]
		return np.linalg.norm(dist_vecs, axis=1)


class NBDistancesDense(FeatureMap):
	def __init__(self, n_atoms):
		G = complete_graph(n_atoms)

		E = np.array([e for e in G.edges()])
		self.edges_s = E[:, 0]
		self.edges_t = E[:, 1]

	def ic(self,geoms):
		dist_vecs = geoms[self.edges_s] - geoms[self.edges_t]
		return np.linalg.norm(dist_vecs, axis=1)


class Molecule:

	def __init__(self,pdb,top,ff = os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp',fms=[]):
		# XXX: unused self.ref = md.load_pdb(pdb)

		if not top and not fms:
			raise ValueError("At least one of `top` or `fms` must be provided")

		if top:
			self.atypes,self.bonds,self.angles,self.dihed4,self.dihed9 = _parse_top(top)
			btypes,atypes,d4types,d9types = _parse_ff(ff)
			self._match_bonds(btypes)
			self._match_angles(atypes)
			self._match_dihed(d4types,d9types)

		self.fms = fms

	def _match_bonds(self,btypes):
		self.bonds_b0 = np.empty(self.bonds.shape[0],dtype=np.float32)
		self.bonds_kb = np.empty(self.bonds.shape[0],dtype=np.float32)

		for i in range(self.bonds.shape[0]):
			matched = False
			for b in btypes:
				t0 = self.atypes[self.bonds[i,0]]
				t1 = self.atypes[self.bonds[i,1]]
				if ((_match_type(t0,b[0]) and _match_type(t1,b[1]))
						or (_match_type(t0,b[1]) and _match_type(t1,b[0]))):
					self.bonds_b0[i] = b[2]
					self.bonds_kb[i] = b[3]
					matched = True
					break	# first match only
			if not matched:
				self.bonds_b0[i] = np.nan
				self.bonds_kb[i] = np.nan
				log.warn(f"bond {i} ({self.bonds[i]}) unmatched")


	def _match_angles(self,atypes):
		self.angles_th0 = np.empty(self.angles.shape[0],dtype=np.float32)
		self.angles_cth = np.empty(self.angles.shape[0],dtype=np.float32)

		for i in range(self.angles.shape[0]):
			matched = False
			for a in atypes:
				t0 = self.atypes[self.angles[i,0]]
				t1 = self.atypes[self.angles[i,1]]
				t2 = self.atypes[self.angles[i,2]]
				if ((_match_type(t0,a[0]) and _match_type(t1,a[1]) and _match_type(t2,a[2]))
						or (_match_type(t0,a[2]) and _match_type(t1,a[1]) and _match_type(t2,a[0]))):
					self.angles_th0[i] = a[3] / 180. * np.pi
					self.angles_cth[i] = a[4]
					matched = True
					break # first match only
			if not matched:
				self.angles_th[i] = np.nan
				self.angles_cth[i] = np.nan
				log.warn(f"angle {i} ({self.angles[i]}) unmatched")


		self.angles_2rth0 = 2. * np.reciprocal(self.angles_th0)
		


	def _match_dihed(self,d4types,d9types):
		self.dihed9_phase = np.empty(self.dihed9.shape[0],dtype=np.float32)
		self.dihed9_kd = np.empty(self.dihed9.shape[0],dtype=np.float32)

		# XXX: type4 are not matched to FF, they are always included if present in topology

		for i in range(self.dihed9.shape[0]):
			matched = False
			for d in d9types:
				t0 = self.atypes[self.dihed9[i,0]]
				t1 = self.atypes[self.dihed9[i,1]]
				t2 = self.atypes[self.dihed9[i,2]]
				t3 = self.atypes[self.dihed9[i,3]]

				if ((_match_type(t0,d[0]) and _match_type(t1,d[1]) and _match_type(t2,d[2]) and _match_type(t3,d[3])) 
						or (_match_type(t0,d[3]) and _match_type(t1,d[2]) and _match_type(t2,d[1]) and _match_type(t3,d[0]))):
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
	angles_th0[:]	(rad)
	angles_2rth0[:]	(precomputed 2 * reciprocal th0)
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




# XXX: unmatched bonds/angles/dihedrals (i.e. nans in their properties) are not handled yet

# geoms[atom][xyz][conf]
	def _ic_bonds(self,geoms):
		out = np.empty([self.bonds.shape[0],geoms.shape[2]],dtype=np.float32)

		for i in range(self.bonds.shape[0]):
			l3 = geoms[self.bonds[i,0],:,:] - geoms[self.bonds[i,1],:,:]

			# this is what original code does but shouldn't these be normalized (as angles) too?
			out[i] = np.linalg.norm(l3,axis=0) # - self.bonds_b0[i]	/ ...

		return out

	def _ic_angles(self,geoms):
		out = np.empty([self.angles.shape[0],geoms.shape[2]],dtype=np.float32)

		for i in range(self.angles.shape[0]):
			v1 = geoms[self.angles[i,0],:,:] - geoms[self.angles[i,1],:,:]
			v2 = geoms[self.angles[i,2],:,:] - geoms[self.angles[i,1],:,:]
			n1 = np.linalg.norm(v1,axis=0)
			n2 = np.linalg.norm(v2,axis=0)
			dot = np.einsum('ij,ij->j',v1,v2)
			dot /= n1 * n2
			aa = np.arccos(dot)
			out[i] = (aa - .75 * self.angles_th0[i]) * self.angles_2rth0[i] # map 0.75 a0 -- 1.25 a0 to 0 -- 1

		return out


	def _ic_dihedral(self,geoms,atoms):
		a12 = geoms[atoms[1],:,:] - geoms[atoms[0],:,:]
		a23 = geoms[atoms[2],:,:] - geoms[atoms[1],:,:]
		a34 = geoms[atoms[3],:,:] - geoms[atoms[2],:,:]

		a12 /= np.linalg.norm(a12,axis=0)
		a23 /= np.linalg.norm(a23,axis=0)
		a34 /= np.linalg.norm(a34,axis=0)

		vp1 = np.cross(a12,a23,axis=0)
		vp2 = np.cross(a23,a34,axis=0)
		vp3 = np.cross(vp1,a23,axis=0)

		sp1 = np.einsum('ij,ij->j',vp1,vp2)
		sp2 = np.einsum('ij,ij->j',vp3,vp2)

		""" original:

		aa = np.arctan2(sp1,sp2) - np.pi * .5
		return np.sin(aa), np.cos(aa)

		this is the same, IMHO, without expensive trigon:""" 

		return (1.-sp2)*.5, (1.+sp1)*.5


	def _ic_dihed4(self,geoms):
		out = np.empty([self.dihed4.shape[0] * 2, geoms.shape[2]],dtype=np.float32)

		for i in range(self.dihed4.shape[0]):
			s,c = self._ic_dihedral(geoms,self.dihed4[i])
			out[2*i] = s
			out[2*i+1] = c

		return out


	def _ic_dihed9(self,geoms):
		out = np.empty([self.dihed9.shape[0] * 2, geoms.shape[2]],dtype=np.float32)

		for i in range(self.dihed9.shape[0]):
			s,c = self._ic_dihedral(geoms,self.dihed9[i])
			out[2*i] = s
			out[2*i+1] = c

		return out


# TODO nbdistance

	def intcoord(self,geoms):
		if not hasattr(self,'atypes'):
			return np.concatenate([fm.ic(geoms) for fm in self.fms],axis=0)

		if geoms.shape[0] != len(self.atypes):
			raise ValueError(f"Number of atoms ({geoms.shape[0]}) does not match topology ({len(self.atypes)})")

		if geoms.shape[1] != 3:
			raise ValueError(f"3D coordinates expected, {geoms.shape[1]} given")

		return np.concatenate([
			self._ic_bonds(geoms),
			self._ic_angles(geoms),
			self._ic_dihed4(geoms),
			self._ic_dihed9(geoms),
			] + [fm.ic(geoms) for fm in self.fms],axis=0)


logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)


if __name__ == '__main__':
	mol = Molecule('alaninedipeptide_H.pdb','topol.top')

	print(mol.atypes)
	print(mol.bonds)
	print(mol.angles)
	print(mol.dihed4)
	print(mol.dihed9)


	ff = os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp'
	btypes,atypes,d4types,d9types = _parse_ff(ff)
	print(d9types)

	print(mol.bonds_b0)
	print(mol.angles_th0)
