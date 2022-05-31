import mdtraj as md
import numpy as np
import re

class Molecule:

	def __init__(self,pdb,top,ff):
		self.ref = md.load_pdb(pdb)
		self.parse_top(top)
		self.parse_ff(ff)
		self.nb = None

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

	def parse_top(self,top):
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
		
		self.atypes = types
		aidx = [ -1 ] * (max(anums) + 1)
		for i,n in enumerate(anums):
			aidx[n] = i

		self.bonds = np.array(list(map(lambda b: [aidx[b[0]], aidx[b[1]]], bonds)),dtype=np.int32)
		self.angles = np.array(list(map(lambda a: [aidx[a[0]], aidx[a[1]], aidx[a[2]]], angles)),dtype=np.int32)
		self.dihed4 = np.array(list(map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]], 
																		filter(lambda d: d[4] == 4, dihedrals)
																)),dtype=np.int32)
		self.dihed9 = np.array(list(map(lambda d: [aidx[d[0]], aidx[d[1]], aidx[d[2]], aidx[d[3]]], 
																		filter(lambda d: d[4] == 9, dihedrals)
																)),dtype=np.int32)



	def parse_ff(self,ff):
		pass



# geoms[atom][xyz][conf]
	def ic_bonds(self,geoms):
		out = np.empty([self.bonds.shape[0],geoms.shape[2]],dtype=np.float32)

		for i in range(self.bonds.shape[0]):
			l3 = geoms[self.bonds[i,0],:,:] - geoms[self.bonds[i,1],:,:]
			out[i] = np.linalg.norm(l3,axis=0) - self.bonds_b0[i]

		return out

	def ic_angles(self,geoms):
		out = np.empty([self.angles.shape[0],geoms.shape[2]],dtype=np.float32)

		for i in range(self.angles.shape[0]):
			v1 = geoms[self.angles[i,0],:,:] - geoms[self.angles[i,1],:,:]
			v2 = geoms[self.angles[i,2],:,:] - geoms[self.angles[i,1],:,:]
			rn1 = np.reciprocal(np.linalg.norm(v1,axis=0))
			rn2 = np.reciprocal(np.linalg.norm(v2,axis=0))
			aa = np.arccos(v1 * v2 * rn1 * rn2)
			out[i] = (aa - .75 * self.angles_th0[i]) * self.angles_2rth0[i] # map 0.75 a0 -- 1.25 a0 to 0 -- 1

		return out

# TODO dihedrals 4/9, nbdistance

	def intcoord(self,geoms):
		if self.nb is not None:
			nb = self.nb.ic(geoms)
		else:
			nb = np.empty([0,geoms.shape[2]],dtype=float32)
			
		return np.concatenate((
			self.ic_bonds(geoms),
			self.ic_angles(geoms),
			nb
			),axis=0)


import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)


if __name__ == '__main__':
	mol = Molecule('alaninedipeptide_H.pdb','topol.top',None)

	print(mol.atypes)
	print(mol.bonds)
	print(mol.angles)
	print(mol.dihed4)
	print(mol.dihed9)
