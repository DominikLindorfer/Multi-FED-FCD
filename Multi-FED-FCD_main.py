#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:47:14 2020

@author: lindorfer

Implementation of Multi-FED-FCD-Couplings from Menuccii Photosynth. Res. (2018)
Test-Case is an Ethylene-Benzene Dimer

Function-definitions are given in  FED_FCD_Functions.py
"""
import FED_FCD_Functions as FED
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

import numpy as np

def FCD_Couplings(tdhf, mf, s0_max, s1_max):
    s = mf.get_ovlp()
    STS_D = 6
    
    print("----------------------------------------------------------------------------")
    print("States ", " FCD-Coupling (eV) ", " dX12 "," dX11 "," dX22 ")
    print("----------------------------------------------------------------------------")
    
    for s0 in range(s0_max):
        for s1 in range(s1_max):
    
            dmxs0 = FED.xs2xs_denisty_matrix_dom(tdhf, s0, s0)
            chg0 = FED.mulliken_pop_dom(mol, dmxs0, s)[1]
            
            dqs0 = np.sum(chg0[:STS_D])-np.sum(chg0[STS_D:])
            
            
            dmxs1 = FED.xs2xs_denisty_matrix_dom(tdhf, s1, s1)
            chg1 = FED.mulliken_pop_dom(mol, dmxs1, s)[1]
            
            dqs1 = np.sum(chg1[:STS_D])-np.sum(chg1[STS_D:])
            
            
            dmxs2xs05 = FED.xs2xs_denisty_matrix_dom(tdhf, s0, s1)
            dmxs2xs50 = FED.xs2xs_denisty_matrix_dom(tdhf, s1, s0)
            dmxs2xs_symm = (dmxs2xs05 + dmxs2xs50) / 2
            
            chg01 = FED.mulliken_pop_dom_transition(mol, dmxs2xs_symm, s)[1]
            
            dqs0s1 = np.sum(chg01[:STS_D])-np.sum(chg01[STS_D:])
            
            energies = (tdhf.e_tot - mf.e_tot) * 27.211386
            denergy = energies[s1] - energies[s0]
            
            fcd = denergy * dqs0s1 / np.sqrt((dqs0 - dqs1)**2 + 4* dqs0s1**2)
            
            if(s1 > s0):
                print(s0+1, "  ", s1+1, "  ", fcd, "  ", dqs0s1, "  ", dqs0, "  ", dqs1)

def FED_Couplings(tdhf, mf, s0_max, s1_max):
    s = mf.get_ovlp()
    STS_D = 6
    
    print("----------------------------------------------------------------------------")
    print("States ", " FED-Coupling (eV) ", " dX12 "," dX11 "," dX22 ")
    print("----------------------------------------------------------------------------")
    
    for s0 in range(s0_max):
        for s1 in range(s1_max):
            
            #XS States & Populations for Donor & Acceptor
            dm_s1 = FED.excitation_denisty_matrix(tdhf, mf, s0, s0)
            dm_s2 = FED.excitation_denisty_matrix(tdhf, mf, s1, s1)
            
            mulliken_hole_s1 = FED.mulliken_pop_dom_transition(mol, dm_s1[0], s)
            mulliken_part_s1 = FED.mulliken_pop_dom_transition(mol, dm_s1[1], s)
            
            mulliken_hole_s2 = FED.mulliken_pop_dom_transition(mol, dm_s2[0], s)
            mulliken_part_s2 = FED.mulliken_pop_dom_transition(mol, dm_s2[1], s)
            
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
            dm_s12 = FED.excitation_denisty_matrix(tdhf, mf, s0, s1)
            dm_s21 = FED.excitation_denisty_matrix(tdhf, mf, s1, s0)
            
            dms_12 = (dm_s12 + dm_s21) / 2
            
            mulliken_hole_s12 = FED.mulliken_pop_dom_transition(mol, dms_12[0], s)
            mulliken_part_s12 = FED.mulliken_pop_dom_transition(mol, dms_12[1], s)
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

FED_Couplings(tdhf, mf, nstates, nstates)
FCD_Couplings(tdhf, mf, nstates, nstates)

#-----Build Matrices-----

s = mf.get_ovlp()
STS_D = 6

s0_max = 10
s1_max = 10

#-----Build Dx Matrix-----
qmat = np.zeros((s0_max, s1_max))

for s0 in range(s0_max):
    for s1 in range(s1_max):
        
        #XS States & Populations for Donor & Acceptor
        if(s0 == s1):
            dm_s1 = FED.excitation_denisty_matrix(tdhf, mf, s0, s0)
            
            mulliken_hole_s1 = FED.mulliken_pop_dom_transition(mol, dm_s1[0], s)
            mulliken_part_s1 = FED.mulliken_pop_dom_transition(mol, dm_s1[1], s)
            
            s1_D_hole = mulliken_hole_s1[1][:STS_D]
            s1_D_part = mulliken_part_s1[1][:STS_D]
            s1_A_hole = mulliken_hole_s1[1][STS_D:]
            s1_A_part = mulliken_part_s1[1][STS_D:]
            
            dx11 = np.sum(s1_D_hole) + np.sum(s1_D_part) - (np.sum(s1_A_hole)+np.sum(s1_A_part))
            
            qmat[s0, s1] = dx11
        else:
            #XS -> XS Excitation Matrix
            dm_s12 = FED.excitation_denisty_matrix(tdhf, mf, s0, s1)
            dm_s21 = FED.excitation_denisty_matrix(tdhf, mf, s1, s0)
            
            dms_12 = (dm_s12 + dm_s21) / 2
            
            mulliken_hole_s12 = FED.mulliken_pop_dom_transition(mol, dms_12[0], s)
            mulliken_part_s12 = FED.mulliken_pop_dom_transition(mol, dms_12[1], s)
            s12_D_hole = mulliken_hole_s12[1][:STS_D]
            s12_D_part = mulliken_part_s12[1][:STS_D]
            s12_A_hole = mulliken_hole_s12[1][STS_D:]
            s12_A_part = mulliken_part_s12[1][STS_D:]
            
            dx12 = np.sum(s12_D_hole) + np.sum(s12_D_part) - (np.sum(s12_A_hole)+np.sum(s12_A_part))
            
            qmat[s0, s1] = dx12

np.savetxt("qmat.txt", qmat)

#-----Build Dq Matrix-----
qmatFCD = np.zeros((s0_max, s1_max))
mullikens = mf.mulliken_pop()

for s0 in range(s0_max):
    for s1 in range(s1_max):
        
        if(s0 == s1):
            dmxs0 = FED.xs2xs_denisty_matrix_dom(tdhf, s0, s0)
            chg0 = FED.mulliken_pop_dom(mol, dmxs0, s)[1]
            dqs0 = np.sum(chg0[:STS_D])-np.sum(chg0[STS_D:])
            
            gs_contrib = np.sum(mullikens[1][:STS_D]) - np.sum(mullikens[1][STS_D:])
            qmatFCD[s0, s1] = dqs0 - gs_contrib
            
        else:
            dmxs2xs05 = FED.xs2xs_denisty_matrix_dom(tdhf, s0, s1)
            dmxs2xs50 = FED.xs2xs_denisty_matrix_dom(tdhf, s1, s0)
            dmxs2xs_symm = (dmxs2xs05 + dmxs2xs50) / 2
        
            chg01 = FED.mulliken_pop_dom_transition(mol, dmxs2xs_symm, s)[1]
            dqs0s1 = np.sum(chg01[:STS_D])-np.sum(chg01[STS_D:])
            
            qmatFCD[s0, s1] = dqs0s1
            
np.savetxt("qmatFCD.txt", qmatFCD)

qmat = qmat / 2
qmatFCD = -qmatFCD / 2

Dmat = np.dot(qmatFCD, qmatFCD) - np.dot(qmat, qmat)

(evals, evecs) = np.linalg.eig(Dmat)

a = np.dot( np.dot(np.transpose(evecs), Dmat), evecs)
a[np.abs(a) < 0.01] = 0

