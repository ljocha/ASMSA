import mdtraj as md
import numpy as np
import re
import os.path
import logging

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
			elif sect == 'dihedrealtypes':
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
	return atom == pattern or (len(pattern) > 1 and atom[0] == pattern[0] and pattern[1] == '*')


class Molecule:

	def __init__(self,pdb,top,ff = os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp'):
		self.ref = md.load_pdb(pdb)
		self.atypes,self.bonds,self.angles,self.dihed4,self.dihed9 = _parse_top(top)
		btypes,atypes,d4types,d9types = _parse_ff(ff)
		self._match_bonds(btypes)
		self._match_angles(atypes)
		self._match_dihed(d4types,d9types)
		self.nb = None

	def _match_bonds(self,btypes):
		self.bonds_b0 = np.empty(self.bonds.shape[0],dtype=np.float32)
		self.bonds_kb = np.empty(self.bonds.shape[0],dtype=np.float32)

		for i in range(self.bonds.shape[0]):
			for b in btypes:
				t0 = self.atypes[self.bonds[i,0]]
				t1 = self.atypes[self.bonds[i,1]]
				if (_match_type(t0,b[0]) and _match_type(t1,b[1])) or (_match_type(t0,b[1]) and _match_type(t1,b[0])):
					 self.bonds_b0[i] = b[2]
					 self.bonds_kb[i] = b[3]
					 break	# first match only

	def _match_angles(self,atypes):
		self.angles_th0 = np.empty(self.angles.shape[0],dtype=np.float32)
		self.angles_cth = np.empty(self.angles.shape[0],dtype=np.float32)

		for i in range(self.angles.shape[0]):
			for a in atypes:
				t0 = self.atypes[self.angles[i,0]]
				t1 = self.atypes[self.angles[i,1]]
				t2 = self.atypes[self.angles[i,2]]
				if (_match_type(t0,a[0]) and _match_type(t1,a[1]) and _match_type(t2,a[2])) or (_match_type(t0,a[2]) and _match_type(t1,a[1]) and _match_type(t2,a[0])):
					self.angles_th0[i] = a[3] / 180. * np.pi
					self.angles_cth[i] = a[4]
					break # first match only

		self.angles_2rth0 = 2. * np.reciprocal(self.angles_th0)
		


	def _match_dihed(self,d4types,d9types):
		# TODO
		pass


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
			rn1 = np.reciprocal(np.linalg.norm(v1,axis=0))
			rn2 = np.reciprocal(np.linalg.norm(v2,axis=0))
			dot = np.einsum('ij,ij->j',v1,v2)
			aa = np.arccos(dot * rn1 * rn2)
			out[i] = (aa - .75 * self.angles_th0[i]) * self.angles_2rth0[i] # map 0.75 a0 -- 1.25 a0 to 0 -- 1

		return out

# TODO dihedrals 4/9, nbdistance

	def intcoord(self,geoms):
		if self.nb is not None:
			nb = self.nb.ic(geoms)
		else:
			nb = np.empty([0,geoms.shape[2]],dtype=np.float32)
			
		return np.concatenate((
			self._ic_bonds(geoms),
			self._ic_angles(geoms),
			nb
			),axis=0)


logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)


if __name__ == '__main__':
	mol = Molecule('alaninedipeptide_H.pdb','topol.top')

	print(mol.atypes)
	print(mol.bonds)
	print(mol.angles)
	print(mol.dihed4)
	print(mol.dihed9)


	print(_parse_ff(os.path.dirname(os.path.abspath(__file__)) + '/ffbonded.itp'))

	print(mol.bonds_b0)
	print(mol.angles_th0)
