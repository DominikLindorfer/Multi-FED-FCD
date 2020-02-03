#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:47:14 2020

@author: lindorfer

Implementation of FED-Couplings from Hsu J. Phys. Chem. C (2008)
"""
from pyscf import gto

mol = gto.Mole()

mol.atom = '''C     0.670518    0.000000    0.000000;
   H     1.241372    0.927754    0.000000;
   H     1.241372   -0.927754    0.000000;
   C    -0.670518    0.000000    0.000000;
   H    -1.241372   -0.927754    0.000000;
   H    -1.241372    0.927754    0.000000;
C       1.3862000000     0.0000000000    2.5;
C       0.6931000000     1.2004844147    2.5;
C      -0.6931000000     1.2004844147    2.5;
C      -1.3862000000     0.0000000000    2.5;
C      -0.6931000000    -1.2004844147    2.5;
C       0.6931000000    -1.2004844147    2.5;
H       2.4618000000     0.0000000000    2.5;
H       1.2309000000     2.1319813390    2.5;
H      -1.2309000000     2.1319813390    2.5;
H      -2.4618000000     0.0000000000    2.5;
H      -1.2309000000    -2.1319813390    2.5;
H       1.2309000000    -2.1319813390    2.5'''

# mol.atom = '''O 0 0 0; H  0 1 0; H 0 0 1'''
mol.basis = '6-31G'
mol.cart = True
mol.charge = 0
mol.spin = 0
mol.symmetry = False
nstates = 10
mol.build(dump_input=False, parse_arg=False)

#######################
#HF-CIS Part with PySCF
#######################
import pyscf
from pyscf import gto, scf, dft, tddft
import pyscf.tdscf
mf = scf.RHF(mol)
mf.kernel()

tdhf = pyscf.tdscf.rhf.TDA(mf)
_, xy1 = tdhf.kernel(nstates=nstates)
tdhf.analyze()

#Mullikens stimmen mit den Werten aus QChem überein!
mullikens = mf.mulliken_pop()

import numpy as np
import numpy
#Mulliken-Charges per Hand für 1. C-Atom wie auf extra-Zettel / Wiki
6 - np.sum(mullikens[0][0:9])

from pyscf.lib import logger

# from https://sunqm.github.io/pyscf/_modules/pyscf/scf/hf.html#SCF.make_rdm1
# From Documentation: full density matrix for RHF
def make_rdm1_dom(mo_coeff, mo_occ, **kwargs):
    '''One-particle density matrix in AO representation

    Args:
        mo_coeff : 2D ndarray
            Orbital coefficients. Each column is one orbital.
        mo_occ : 1D ndarray
            Occupancy
    '''
    mocc = mo_coeff[:,mo_occ>0]
    return numpy.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)

def get_ovlp_dom(mol):
    '''Overlap matrix
    '''
    return mol.intor_symmetric('int1e_ovlp')

def mulliken_pop_dom(mol, dm, s=None):
    r'''Mulliken population analysis

    .. math:: M_{ij} = D_{ij} S_{ji}

    Mulliken charges

    .. math:: \delta_i = \sum_j M_{ij}

    Returns:
        A list : pop, charges

        pop : nparray
            Mulliken population on each atomic orbitals
        charges : nparray
            Mulliken charges
    '''

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        pop = numpy.einsum('ij,ji->i', dm, s).real

    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
    chg = mol.atom_charges() - chg

    return pop, chg

#Calc Mullikens with own routine
dm = mf.make_rdm1()
s = mf.get_ovlp()
mulliken_pop_dom(mol, dm, s)

#Check Einstein-Summation in mulliken_pop_dom()
np.dot(dm[0][0:],s[0][0:])

#Check Occupancy, MO-Coeffs, Density-Matrix etc.
mf.mo_occ
mf.mo_coeff

mf.mo_occ[mf.mo_occ>0]

rdm = mf.make_rdm1()
rdm_dom = make_rdm1_dom(mf.mo_coeff, mf.mo_occ)

# from https://github.com/pyscf/pyscf/blob/master/examples/tddft/22-density.py
#Make Density Matrix for Excited State
def tda_denisty_matrix_dom(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm


# mf_dom = tdhf._scf
# dm = np.diag(mf.mo_occ)
# nocc = xy1[0][0].shape[0]
# dm[:nocc,:nocc]
# dm[nocc:,nocc:]
# mo = mf.mo_coeff
# dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())

# Density matrix for the excited state and Mullikens
#Ergebnis stimmt mit QChem überein!
dm = tda_denisty_matrix_dom(tdhf, 0)
mulliken_pop_dom(mol, dm, s)

#Build the transition density matrix (GS -> XS)
'''
dm_ia = MO_i * MO_a  of molA
'''
moA = mf.mo_coeff
o_A = moA[:,mf.mo_occ!=0]
v_A = moA[:,mf.mo_occ==0]

state_id = 0  # The second excited state (non-dark exciton state)
t1_A = xy1[state_id][0]

dm_ia = o_A.dot(t1_A).dot(v_A.T)
dm_ia_symm = (dm_ia + np.transpose(dm_ia)) / 2

def mulliken_pop_dom_transition(mol, dm, s=None):

    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        pop = numpy.einsum('ij,ji->i', dm, s).real
    # else: # ROHF
    #     pop = numpy.einsum('ij,ji->i', dm[0]+dm[1], s).real

    chg = numpy.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        chg[s[0]] += pop[i]
    # chg = mol.atom_charges() - chg

    return pop, chg

def dip_moment_dom(mol, dm, unit='Debye', verbose=logger.NOTE, **kwargs):
    r''' Dipole moment calculation

    .. math::

        \mu_x = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|x|\mu) + \sum_A Q_A X_A\\
        \mu_y = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|y|\mu) + \sum_A Q_A Y_A\\
        \mu_z = -\sum_{\mu}\sum_{\nu} P_{\mu\nu}(\nu|z|\mu) + \sum_A Q_A Z_A

    where :math:`\mu_x, \mu_y, \mu_z` are the x, y and z components of dipole
    moment

    Args:
         mol: an instance of :class:`Mole`
         dm : a 2D ndarrays density matrices

    Return:
        A list: the dipole moment on x, y and z component
    '''

    log = logger.new_logger(mol, verbose)

    if 'unit_symbol' in kwargs:  # pragma: no cover
        log.warn('Kwarg "unit_symbol" was deprecated. It was replaced by kwarg '
                 'unit since PySCF-1.5.')
        unit = kwargs['unit_symbol']

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = numpy.einsum('xij,ji->x', ao_dip, dm).real

    # charges = mol.atom_charges()
    # coords  = mol.atom_coords()
    # nucl_dip = numpy.einsum('i,ix->x', charges, coords)
    mol_dip = el_dip
    # mol_dip *= nist.AU2DEBYE
    
    log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    
    # if unit.upper() == 'DEBYE':
    #     mol_dip *= nist.AU2DEBYE
    #     log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    # else:
    #     log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    return mol_dip

#Transition Mullikens
mulliken_pop_dom(mol, dm_ia, s)
#Same result from own dipole moment routine and own transition density matrix and from PySCF routine
dip_dom = dip_moment_dom(mol, dm_ia)
dip_dom * 2
#Wrong result from PySCF routine bc nuc. dipolemoment is included 
# -> modified dipole_moment_dom() routine below
dip = mf.dip_moment(mol, dm)

mulliken_pop_dom_transition(mol, dm_ia, s)
mulliken_pop_dom_transition(mol, dm_ia_symm, s)
    
tdhf.transition_dipole()

def xs2xs_denisty_matrix_dom(td, state_id1, state_id2):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the XS->XS density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id1][0]
    cis_t2 = td.xy[state_id2][0]
    
    dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t2.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm


dmxs2xs = xs2xs_denisty_matrix_dom(tdhf, 0, 2)
#has nuclear contribution, the 0.17 seem to be an addition from nuclei?! is unclear!
dip_xs2xs = dip_moment_dom(mol, dmxs2xs)
mf.dip_moment(mol, dmxs2xs, unit='A.U.')

def difference_density_matrix(td, state_id1, state_id2):
    '''
    Difference Density Matrix from Herbert 2011
    '''
    cis_t1 = tdhf.xy[state_id1][0]
    cis_t2 = tdhf.xy[state_id2][0]
        
    dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
    # The ground state density matrix in mo_basis
    mf = tdhf._scf
    dm = np.diag(mf.mo_occ)
    dm_gs = np.diag(mf.mo_occ)
    
    diff_dm = dm - dm_gs
    
    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2
    
    diff_dm[:nocc,:nocc] += dm_oo * 2
    diff_dm[nocc:,nocc:] += dm_vv * 2
    
    nocc = cis_t1.shape[0]    
    # Transform density matrices to AO basis
    mo = mf.mo_coeff
    dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    dm_gs = np.einsum('pi,ij,qj->pq', mo, dm_gs, mo.conj())
    diff_dm = np.einsum('pi,ij,qj->pq', mo, diff_dm, mo.conj())
    
    return np.array([dm, dm_gs, diff_dm])

diffdms = difference_density_matrix(tdhf, 0, 0)

#Difference charges - selbes Ergebnis wie QChem

mulliken_pop_dom(mol, diffdms[0], s)
mulliken_pop_dom(mol, diffdms[1], s)
mulliken_pop_dom_transition(mol, diffdms[0] - diffdms[1], s)
mulliken_pop_dom_transition(mol, diffdms[2], s)


def excitation_denisty_matrix(td, state_id1, state_id2):
    '''
    Excitation Density Matrix from Hsu 2008
    '''
    cis_t1 = tdhf.xy[state_id1][0]
    cis_t2 = tdhf.xy[state_id2][0]
        
    dm_oo = np.einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
    # Create density matrix in mo_basis
    dm_hole = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    dm_part = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    
    nocc = cis_t1.shape[0]    
    # Add CIS contribution for hole and particle density (detachment)
    dm_hole[:nocc,:nocc] += dm_oo * 2
    dm_part[nocc:,nocc:] += dm_vv * 2
    
    # Transform density matrices to AO basis
    mo = mf.mo_coeff
    dm_hole = np.einsum('pi,ij,qj->pq', mo, dm_hole, mo.conj())
    dm_part = np.einsum('pi,ij,qj->pq', mo, dm_part, mo.conj())
    
    return np.array([dm_hole, dm_part])



def FED_Couplings(tdhf, s0_max, s1_max):
    s = mf.get_ovlp()
    STS_D = 6
    
    print("----------------------------------------------------------------------------")
    print("States ", " FED-Coupling (eV) ", " dX12 "," dX11 "," dX22 ")
    print("----------------------------------------------------------------------------")
    
    for s0 in range(s0_max):
        for s1 in range(s1_max):
            
            #XS States & Populations for Donor & Acceptor
            dm_s1 = excitation_denisty_matrix(tdhf, s0, s0)
            dm_s2 = excitation_denisty_matrix(tdhf, s1, s1)
            
            mulliken_hole_s1 = mulliken_pop_dom_transition(mol, dm_s1[0], s)
            mulliken_part_s1 = mulliken_pop_dom_transition(mol, dm_s1[1], s)
            
            mulliken_hole_s2 = mulliken_pop_dom_transition(mol, dm_s2[0], s)
            mulliken_part_s2 = mulliken_pop_dom_transition(mol, dm_s2[1], s)
            
            s1_D_hole = mulliken_hole_s1[1][:STS_D]
            s1_D_part = mulliken_part_s1[1][:STS_D]
            s1_A_hole = mulliken_hole_s1[1][STS_D:]
            s1_A_part = mulliken_part_s1[1][STS_D:]
            
            s2_D_hole = mulliken_hole_s2[1][:STS_D]
            s2_D_part = mulliken_part_s2[1][:STS_D]
            s2_A_hole = mulliken_hole_s2[1][STS_D:]
            s2_A_part = mulliken_part_s2[1][STS_D:]
            
            dx11 = np.sum(s1_D_hole) + np.sum(s1_D_part) - (np.sum(s1_A_hole)+np.sum(s1_A_part))
            dx22 = np.sum(s2_D_hole) + np.sum(s2_D_part) - (np.sum(s2_A_hole)+np.sum(s2_A_part))
            
            #XS -> XS Excitation Matrix
            dm_s12 = excitation_denisty_matrix(tdhf, s0, s1)
            dm_s21 = excitation_denisty_matrix(tdhf, s1, s0)
            
            dms_12 = (dm_s12 + dm_s21) / 2
            
            mulliken_hole_s12 = mulliken_pop_dom_transition(mol, dms_12[0], s)
            mulliken_part_s12 = mulliken_pop_dom_transition(mol, dms_12[1], s)
            s12_D_hole = mulliken_hole_s12[1][:STS_D]
            s12_D_part = mulliken_part_s12[1][:STS_D]
            s12_A_hole = mulliken_hole_s12[1][STS_D:]
            s12_A_part = mulliken_part_s12[1][STS_D:]
            
            dx12 = np.sum(s12_D_hole) + np.sum(s12_D_part) - (np.sum(s12_A_hole)+np.sum(s12_A_part))
            
            energies = (tdhf.e_tot - mf.e_tot) * 27.211386
            denergy = energies[s1] - energies[s0]
            
            fed = denergy * dx12 / np.sqrt((dx11 - dx22)**2 + 4* dx12**2)
            
            if(s1 > s0):
                print(s0+1, "  ", s1+1, "  ", fed, "  ", dx12, "  ", dx11, "  ", dx22)

    
FED_Couplings(tdhf, 10,10)





###########################################################
#Comments
###########################################################

STS_D = 6

dm_s1 = excitation_denisty_matrix(tdhf, 0, 0)
dm_s2 = excitation_denisty_matrix(tdhf, 5, 5)

mulliken_hole_s1 = mulliken_pop_dom_transition(mol, dm_s1[0], s)
mulliken_part_s1 = mulliken_pop_dom_transition(mol, dm_s1[1], s)

mulliken_hole_s2 = mulliken_pop_dom_transition(mol, dm_s2[0], s)
mulliken_part_s2 = mulliken_pop_dom_transition(mol, dm_s2[1], s)

s1_D_hole = mulliken_hole_s1[1][:STS_D]
s1_D_part = mulliken_part_s1[1][:STS_D]
s1_A_hole = mulliken_hole_s1[1][STS_D:]
s1_A_part = mulliken_part_s1[1][STS_D:]

s2_D_hole = mulliken_hole_s2[1][:STS_D]
s2_D_part = mulliken_part_s2[1][:STS_D]
s2_A_hole = mulliken_hole_s2[1][STS_D:]
s2_A_part = mulliken_part_s2[1][STS_D:]

dx11 = np.sum(s1_D_hole) + np.sum(s1_D_part) - (np.sum(s1_A_hole)+np.sum(s1_A_part))
dx22 = np.sum(s2_D_hole) + np.sum(s2_D_part) - (np.sum(s2_A_hole)+np.sum(s2_A_part))


dm_s12 = excitation_denisty_matrix(tdhf, 0, 5)
dm_s21 = excitation_denisty_matrix(tdhf, 5, 0)

dms_12 = (dm_s12 + dm_s21) / 2

mulliken_hole_s12 = mulliken_pop_dom_transition(mol, dm_s12[0], s)
mulliken_part_s12 = mulliken_pop_dom_transition(mol, dm_s12[1], s)
s12_D_hole = mulliken_hole_s12[1][:STS_D]
s12_D_part = mulliken_part_s12[1][:STS_D]
s12_A_hole = mulliken_hole_s12[1][STS_D:]
s12_A_part = mulliken_part_s12[1][STS_D:]

dx12 = np.sum(s12_D_hole) + np.sum(s12_D_part) - (np.sum(s12_A_hole)+np.sum(s12_A_part))


# import FED_Functions as FED
# FED.test()








state_id1 = 0
state_id2 = 0

cis_t1 = tdhf.xy[state_id1][0]
cis_t2 = tdhf.xy[state_id2][0]
    
dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t2.conj())

# The ground state density matrix in mo_basis
mf = tdhf._scf
dm = np.diag(mf.mo_occ)
dm_gs = np.diag(mf.mo_occ)

# dm = dm - dm_gs
dm = np.diag(np.zeros(92))

# Add CIS contribution
nocc = cis_t1.shape[0]
dm[:nocc,:nocc] += dm_oo * 2
dm[nocc:,nocc:] += dm_vv * 2
# dm1[nocc:,nocc:] += dm_vv * 2

# Transform density matrix to AO basis
mo = mf.mo_coeff
dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
dm_gs = np.einsum('pi,ij,qj->pq', mo, dm_gs, mo.conj())
# dm1 = np.einsum('pi,ij,qj->pq', mo, dm1, mo.conj())

# mulliken_pop_dom(mol, dm+dm1, s)

mulliken_pop_dom(mol, dm, s)
mulliken_pop_dom(mol, dm_gs, s)

#difference charge - selbes Ergebnis wie QChem
mulliken_pop_dom_transition(mol, dm, s)
mulliken_pop_dom_transition(mol, dm - dm_gs, s)


mulliken_pop_dom(mol, dm1, s)

s_occ = s[:nocc,:nocc]
s_vir = s[nocc:, nocc:]

numpy.einsum('ij,ji->i', dm+dm1, s).real
numpy.zeros(mol.natm)


for i, s in enumerate(mol.ao_labels(fmt=None)):
    chg[s[0]] += pop[i]
chg = mol.atom_charges() - chg









nocc = cis_t1.shape[0]
# The ground state density matrix in mo_basis
mf = tdhf._scf
# dm = np.diag(mf.mo_occ)
dm = np.diag(np.zeros(92))

# Add CIS contribution
dm[:nocc,:nocc] += dm_oo * 2
# dm[nocc:,nocc:] += dm_vv * 2

# Transform density matrix to AO basis
mo = mf.mo_coeff
mo_occ = mo[:nocc, :nocc]
mo_mat = np.diag(np.zeros(92))

mo_mat[:nocc, :nocc] += mo_occ * 2

dm = np.einsum('pi,ij,qj->pq', mo_mat, dm, mo_mat.conj())
s = mf.get_ovlp()

mulliken_pop_dom_transition(mol,dm,s)














mo_coeff1 =  mf.mo_coeff[:nocc,:]
mo_coeff2 =  mf.mo_coeff[:,:nocc]

dm_particle = np.einsum('pi,ij,qj->pq', mo_coeff2, dm_oo, mo_coeff2.conj())

pop = numpy.einsum('ij,ji->i', dm_particle, s).real

chg = numpy.zeros(mol.natm)
for i, s in enumerate(mol.ao_labels(fmt=None)):
    chg[s[0]] += pop[i]
chg = mol.atom_charges() - chg






# The ground state density matrix in mo_basis
mf = td._scf
dm = np.diag(mf.mo_occ)
nocc = cis_t1.shape[0]

dm_occ = dm[:nocc,:nocc]
dm_vir = dm[nocc:,nocc:]

dm_occ += 2* dm_oo
dm_vir += 2* dm_vv

dm_occ_fin = np.einsum('pi,ij,qj->pq', mo[:nocc,:nocc], dm_occ, mo[:nocc,:nocc].conj())
dm_vir_fin = np.einsum('pi,ij,qj->pq', mo[nocc:,nocc:], dm_vir, mo[nocc:,nocc:].conj())






dm_occ = xs2xs_excitation_denisty_matrix_dom(tdhf, s0, s0)
mp_occ = mulliken_pop_dom(mol, dm_occ_fin, s[:nocc,:nocc])


































s0 = 0
s1 = 1

dm0 = xs2xs_excitation_density_matrix(tdhf, s0, s0)
dm1 = xs2xs_excitation_density_matrix(tdhf, s1, s1)
dm01 = xs2xs_excitation_density_matrix(tdhf, s0, s1)
dm10 = xs2xs_excitation_density_matrix(tdhf, s1, s0)

dm1001 = (dm01 + dm10) / 2

mp0 = mulliken_pop_dom(mol, dm0, s)

mf.mulliken_pop(mol, dm0, s)

mp1 = mulliken_pop_dom(mol, dm1, s)
mp01 = mulliken_pop_dom(mol, dm01, s)
mp10 = mulliken_pop_dom(mol, dm10, s)

mp1001 = mulliken_pop_dom(mol, dm1001, s)

dx01q = np.sum(mp01[0][0:26]) - np.sum(mp01[0][26:])
dx10q = np.sum(mp10[0][0:26]) - np.sum(mp10[0][26:])

dx00q = np.sum(mp0[0][0:26])  - np.sum(mp0[0][26:])
dx11q = np.sum(mp1[0][0:26]) - np.sum(mp1[0][26:]) 
dx1001q = np.sum(mp1001[0][0:26])  - np.sum(mp1001[0][26:])

dx01 = np.sum(mp01[1][0:6]) - np.sum(mp01[1][6:18])
dx10 = np.sum(mp10[1][0:6]) - np.sum(mp10[1][6:18])
dx00 = np.sum(mp0[1][0:6]) - np.sum(mp0[1][6:18]) 
dx11 = np.sum(mp1[1][0:6]) - np.sum(mp1[1][6:18]) 

energies = (tdhf.e_tot - mf.e_tot) * 27.211386
denergy = energies[s1] - energies[s0]

dx1001q = (dx01q + dx10q) / 2
denergy * dx1001q / np.sqrt((dx11q - dx00q)**2 + 4* dx1001q**2)

dx1001 = (dx01 + dx10) / 2
denergy * dx1001 / np.sqrt((dx11 - dx00)**2 + 4* dx1001**2)


def FED_Couplings(s0_max, s1_max):
    
    for s0 in range(s0_max):
        for s1 in range(s1_max):
            
            dm0 = xs2xs_excitation_density_matrix(tdhf, s0, s0)
            dm1 = xs2xs_excitation_density_matrix(tdhf, s1, s1)
            dm01 = xs2xs_excitation_density_matrix(tdhf, s0, s1)
            dm10 = xs2xs_excitation_density_matrix(tdhf, s1, s0)
            
            dm1001 = (dm01 + dm10) / 2
            
            mp0 = mulliken_pop_dom(mol, dm0, s)
            mp1 = mulliken_pop_dom(mol, dm1, s)
            mp01 = mulliken_pop_dom(mol, dm01, s)
            mp10 = mulliken_pop_dom(mol, dm10, s)
            
            mp1001 = mulliken_pop_dom(mol, dm1001, s)
            
            dx01q = np.sum(mp01[0][0:26]) - np.sum(mp01[0][26:])
            dx10q = np.sum(mp10[0][0:26]) - np.sum(mp10[0][26:])
            
            dx00q = np.sum(mp0[0][0:26])  - np.sum(mp0[0][26:])
            dx11q = np.sum(mp1[0][0:26]) - np.sum(mp1[0][26:]) 
            dx1001q = np.sum(mp1001[0][0:26])  - np.sum(mp1001[0][26:])
            
            # dx01 = np.sum(mp01[1][0:6]) - np.sum(mp01[1][6:18])
            # dx10 = np.sum(mp10[1][0:6]) - np.sum(mp10[1][6:18])
            # dx00 = np.sum(mp0[1][0:6]) - np.sum(mp0[1][6:18]) 
            # dx11 = np.sum(mp1[1][0:6]) - np.sum(mp1[1][6:18]) 
            
            energies = (tdhf.e_tot - mf.e_tot) * 27.211386
            denergy = energies[s1] - energies[s0]
            
            #dx1001q = (dx01q + dx10q) / 2
            fed = denergy * dx1001q / np.sqrt((dx11q - dx00q)**2 + 4* dx1001q**2)
            if(s1 > s0):
                print(s0+1, "  ", s1+1, "  ", fed)

    
FED_Couplings(10,10)
    















#denergy * 0.000124 / np.sqrt((1.999999 - 1.999481)**2 + 4* 0.000124**2) 




#Test-Parts
#Test transition density by calc. transition dipole moment
def _charge_center(mol):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    return numpy.einsum('z,zr->r', charges, coords)/charges.sum()

with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric('int1e_r', comp=3)

from pyscf import lib

nstates1 = len(xy1)
pol_shape = ints.shape[:-2]
nao = ints.shape[-1]
intsre = ints.reshape(-1,nao,nao)

ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), o_A.conj(), v_A)
pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy1])

pol = pol.reshape((nstates,)+pol_shape)

tdhf.transition_dipole()

def _contract_multipole(tdobj, ints, hermi=True, xy=None):
    if xy is None: xy = tdobj.xy
    mo_coeff = tdobj._scf.mo_coeff
    mo_occ = tdobj._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]

    nstates = len(xy)
    pol_shape = ints.shape[:-2]
    nao = ints.shape[-1]

    #Incompatible to old numpy version
    #ints = numpy.einsum('...pq,pi,qj->...ij', ints, orbo.conj(), orbv)
    ints = lib.einsum('xpq,pi,qj->xij', ints.reshape(-1,nao,nao), orbo.conj(), orbv)
    pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) * 2 for x,y in xy])
    if isinstance(xy[0][1], numpy.ndarray):
        if hermi:
            pol += [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
        else:  # anti-Hermitian
            pol -= [numpy.einsum('xij,ij->x', ints, y) * 2 for x,y in xy]
    pol = pol.reshape((nstates,)+pol_shape)
    return pol

def transition_dipole(tdobj, xy=None):
    '''Transition dipole moments in the length gauge'''
    mol = tdobj.mol
    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric('int1e_r', comp=3)
    return tdobj._contract_multipole(ints, hermi=True, xy=xy)



# CIS calculations for the excited states of two molecules
molA = gto.M(atom='H 0.5 0.2 0.1; F 0 -0.1 -0.2', basis='ccpvdz')
mfA = scf.RHF(molA).run()
moA = mfA.mo_coeff
o_A = moA[:,mfA.mo_occ!=0]
v_A = moA[:,mfA.mo_occ==0]
tdA = mfA.TDA().run()

molB = gto.M(atom='C 0.9 0.2 0; O 0.1 .2 .1', basis='ccpvtz')
mfB = scf.RHF(molB).run()
moB = mfB.mo_coeff
o_B = moB[:,mfB.mo_occ!=0]
v_B = moB[:,mfB.mo_occ==0]
tdB = mfB.TDA().run()

# CIS coeffcients
state_id = 2  # The third excited state
t1_A = tdA.xy[state_id][0]
t1_B = tdB.xy[state_id][0]

'''
dm_ia = MO_i * MO_a  of molA
dm_jb = MO_j * MO_b  of molB
'''
dm_ia = o_A.dot(t1_A).dot(v_A.T)
dm_jb = o_B.dot(t1_B).dot(v_B.T)




###################
#TDDFT / B3LYP Part
###################
from pyscf import gto, scf, dft, tddft
import pyscf.tdscf
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.kernel()

mf._numint.libxc = pyscf.dft.xcfun
td = pyscf.tdscf.TDA(mf)
td_rpa = pyscf.tdscf.RPA(mf)

_, xy1 = td.kernel(nstates=nstates)

td.analyze()

dm = mf.make_rdm1()
scf.hf.energy_elec(mf, dm)
mf.energy_elec(dm)

#Mullikens stimmen mit den Werten aus QChem überein!
mullikens = mf.mulliken_pop()

import numpy as np

#Mulliken-Charges per Hand wie auf extra-Zettel / Wiki
6 - np.sum(mullikens[0][0:9])

1 - np.sum(mullikens[0][10:12])