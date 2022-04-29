#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:47:14 2020

@author: lindorfer

Implementation of FED-Couplings from Hsu J. Phys. Chem. C (2008)

This file contains function definitions used in the main-part
"""
import pyscf
import pyscf.dft
import pyscf.tdscf
from pyscf.lib import logger
import numpy as np
import numpy
from pyscf.lib import misc
from numpy import asarray  # For backward compatibility
from numpy_helper import *
try:
# Import tblis before libnp_helper to avoid potential dl-loading conflicts
    from pyscf.lib import tblis_einsum
    FOUND_TBLIS = True
except (ImportError, OSError):
    FOUND_TBLIS = False

_np_helper = misc.load_library('libnp_helper')

print(FOUND_TBLIS)
#from numba import jit

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


# from https://github.com/pyscf/pyscf/blob/master/examples/tddft/22-density.py
#Make Density Matrix for Excited State
def tda_denisty_matrix_dom(td, state_id):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id][0]
    dm_oo =-einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
    dm_vv = einsum('ia,ic->ac', cis_t1, cis_t1.conj())

    # The ground state density matrix in mo_basis
    mf = td._scf
    dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

def mulliken_pop_dom_transition(mol, dm, s=None):
    '''
    Calculate Mullikens without substracting the atomic charges
    '''
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

def transition_density_matrix(mf, tdhf, state_id):
    '''
    Build the (symmetric) transition density matrix (GS -> XS)
    dm_ia = MO_i * MO_a  of molA
    '''
    moA = mf.mo_coeff
    o_A = moA[:,mf.mo_occ!=0]
    v_A = moA[:,mf.mo_occ==0]
    
    t1_A = tdhf.xy[state_id][0]
    
    dm_ia = o_A.dot(t1_A).dot(v_A.T)
    dm_ia_symm = (dm_ia + np.transpose(dm_ia)) / 2
    
    return dm_ia_symm

def xs2xs_denisty_matrix_dom(td, state_id1, state_id2):
    '''
    Taking the TDA amplitudes as the CIS coefficients, calculate the XS->XS density
    matrix (in AO basis) of the excited states
    '''
    cis_t1 = td.xy[state_id1][0]
    cis_t2 = td.xy[state_id2][0]
    
    dm_oo =-einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
    # Create density matrix in mo_basis
    mf = td._scf
    dm = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    # Add the ground state density matrix in mo_basisif states are equal
    # see Hsu 2014 eq. 46
    if(state_id1 == state_id2):
        dm = np.diag(mf.mo_occ)

    # Add CIS contribution
    nocc = cis_t1.shape[0]
    dm[:nocc,:nocc] += dm_oo * 2
    dm[nocc:,nocc:] += dm_vv * 2

    # Transform density matrix to AO basis
    mo = mf.mo_coeff
    dm = einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    return dm

# def xs2xs_denisty_matrix_dom2(tdhf, state_id1, state_id2):
#     '''
#     Taking the TDA amplitudes as the CIS coefficients, calculate the XS->XS density
#     matrix (in AO basis) of the excited states
#     '''
#     mf = tdhf._scf

#     cis_t1 = tdhf.xy[state_id1][0]
#     cis_t2 = tdhf.xy[state_id2][0]
        
#     dm_oo = -einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
#     dm_vv = einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
#     # Create density matrix in mo_basis
#     dm = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    
#     nocc = cis_t1.shape[0]    
#     # Add CIS contribution for hole and particle density (detachment)
#     dm[:nocc,:nocc] += dm_oo * 2
#     dm[nocc:,nocc:] += dm_vv * 2
    
#     # Transform density matrices to AO basis
#     mo = mf.mo_coeff
#     dm = einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    
#     return dm

def difference_density_matrix(tdhf, state_id1, state_id2):
    '''
    Difference Density Matrix from Herbert J. Chem. Theor. Comp. (2011)
    Calculates the XS-GS density matrix which can be used e.g. for difference charges
    This can be used to check the Mulliken particle & hole contributions
    '''
    cis_t1 = tdhf.xy[state_id1][0]
    cis_t2 = tdhf.xy[state_id2][0]
        
    dm_oo = -einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
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
    dm = einsum('pi,ij,qj->pq', mo, dm, mo.conj())
    dm_gs = einsum('pi,ij,qj->pq', mo, dm_gs, mo.conj())
    diff_dm = einsum('pi,ij,qj->pq', mo, diff_dm, mo.conj())
    
    return np.array([dm, dm_gs, diff_dm])

def excitation_denisty_matrix(tdhf, mf, state_id1, state_id2):
    '''
    Excitation Density Matrix from Hsu 2008
    '''
    cis_t1 = tdhf.xy[state_id1][0]
    cis_t2 = tdhf.xy[state_id2][0]
    dm_oo = einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
    # Create density matrix in mo_basis
    dm_hole = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    dm_part = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    
    nocc = cis_t1.shape[0]    
    # Add CIS contribution for hole and particle density (detachment)
    dm_hole[:nocc,:nocc] += dm_oo * 2
    dm_part[nocc:,nocc:] += dm_vv * 2
    
    # Transform density matrices to AO basis
    mo = mf.mo_coeff
    dm_hole = einsum('pi,ij,qj->pq', mo, dm_hole, mo.conj())
    dm_part = einsum('pi,ij,qj->pq', mo, dm_part, mo.conj())
    
    return np.array([dm_hole, dm_part])

def excitation_denisty_matrix_pickle(cis_t1, cis_t2, mo, state_id1, state_id2):
    '''
    Excitation Density Matrix from Hsu 2008
    '''
    # cis_t1 = tdhf.xy[state_id1][0]
    # cis_t2 = tdhf.xy[state_id2][0]
    dm_oo = einsum('ia,ka->ik', cis_t1.conj(), cis_t2)
    dm_vv = einsum('ia,ic->ac', cis_t1, cis_t2.conj())
    
    # Create density matrix in mo_basis
    dm_hole = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    dm_part = np.diag(np.zeros(cis_t1.shape[0] + cis_t1.shape[1]))
    
    nocc = cis_t1.shape[0]    
    # Add CIS contribution for hole and particle density (detachment)
    dm_hole[:nocc,:nocc] += dm_oo * 2
    dm_part[nocc:,nocc:] += dm_vv * 2
    
    # Transform density matrices to AO basis
    # mo = mf.mo_coeff
    dm_hole = einsum('pi,ij,qj->pq', mo, dm_hole, mo.conj())
    dm_part = einsum('pi,ij,qj->pq', mo, dm_part, mo.conj())
    
    return np.array([dm_hole, dm_part])
