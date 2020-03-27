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

#-----HF-CIS Part with PySCF-----
import pyscf
from pyscf import gto, scf, dft, tddft
import pyscf.tdscf
mf = scf.RHF(mol)
mf.kernel()
tdhf = pyscf.tdscf.rhf.TDA(mf)
_, xy1 = tdhf.kernel(nstates=nstates)
tdhf.analyze()

import numpy as np

#-----FCD and FED Couplings-----
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

# FED_Couplings(tdhf, mf, nstates, nstates)
# FCD_Couplings(tdhf, mf, nstates, nstates)

#-----Start of Multi-FED-FCD Scheme-----
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

# np.savetxt("qmat.txt", qmat)

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
            
            qmatFCD[s0, s1] = -dqs0s1
            
# np.savetxt("qmatFCD.txt", qmatFCD)

qmat = qmat / 2
qmatFCD = -qmatFCD / 2

#-----Build D Matrix-----
Dmat = np.dot(qmatFCD, qmatFCD) - np.dot(qmat, qmat)
(Devals, Devecs) = np.linalg.eig(Dmat)

# np.set_printoptions(precision = 15, suppress=True)
# print(repr(Dmat))

#-----Divide into CT / LE Subspaces-----
CT_subspace = np.array([])
LE_subspace = np.array([])

for i in range(Devals.size):
    if(Devals[i] < 0):
        LE_subspace = np.append(LE_subspace, i)
    if(Devals[i] > 0):
        CT_subspace = np.append(CT_subspace, i)

#-----Transform Dx and Dq to D-Basis-----
DxDbasis = np.dot( np.dot(np.transpose(Devecs), qmat), Devecs)
DqDbasis = np.dot( np.dot(np.transpose(Devecs), qmatFCD), Devecs)


DxDbasis_LE = DxDbasis[LE_subspace.astype(int),:][:,LE_subspace.astype(int)]
(evals_LE, evecs_LE) = np.linalg.eig(DxDbasis_LE)

DqDbasis_CT = DqDbasis[CT_subspace.astype(int),:][:,CT_subspace.astype(int)]
(evals_CT, evecs_CT) = np.linalg.eig(DqDbasis_CT)

#-----Build U2-----
U2 = np.zeros((s0_max, s1_max))
U2[0:evecs_CT.shape[0], 0:evecs_CT.shape[0]] = evecs_CT
U2[evecs_CT.shape[0]:s0_max, evecs_CT.shape[0]:s0_max] = evecs_LE

DxFinal = np.dot( np.dot(np.transpose(U2), DxDbasis), U2)
DqFinal = np.dot( np.dot(np.transpose(U2), DqDbasis), U2)

#-----Divide CT Subspace into A-B+ (AmBp) and A+B- (ApBm)-----
CT_subspace_CT1 = np.array([])
CT_subspace_CT2 = np.array([])

for i in range(CT_subspace.size):
    if(DqFinal[i,i] < 0):
        CT_subspace_CT2 = np.append(CT_subspace_CT2, i)
    if(DqFinal[i,i] > 0):
        CT_subspace_CT1 = np.append(CT_subspace_CT1, i)

#-----Divide LE Subspace into A*B (AsB) and AB* (ABs)-----
LE_subspace_LE1 = np.array([])
LE_subspace_LE2 = np.array([])
CT_size = evecs_CT.shape[0]

for i in range(LE_subspace.size):
    if(DxFinal[CT_size + i, CT_size + i] < 0):
        LE_subspace_LE2 = np.append(LE_subspace_LE2, CT_size + i)
    if(DxFinal[CT_size + i, CT_size + i] > 0):
        LE_subspace_LE1 = np.append(LE_subspace_LE1, CT_size + i)

#-----Order Matrix into Submatrices as in the SI of Cupellini 2018: LE1 LE2 CT1 CT2-----
Dmat_order = np.concatenate((CT_subspace_CT1, CT_subspace_CT2 , LE_subspace_LE1, LE_subspace_LE2))

#-----Initial Hamiltonian-----
H_init = np.zeros((tdhf.e.shape[0], tdhf.e.shape[0]))
np.fill_diagonal(H_init, tdhf.e / 0.0367493 * 8065.5)

#-----Hamiltonian in D-Basis-----
H_Dbasis = np.dot( np.dot(np.transpose(Devecs), H_init), Devecs)
# H_Dbasis[np.abs(H_Dbasis) < 1e-5] = 0

#-----Restructure Hamiltonian into LE / CT Subspaces-----
Dmat_order_CTLE = np.concatenate((CT_subspace, LE_subspace))
H_Dbasis_CTLE = H_Dbasis[Dmat_order_CTLE.astype(int),:][:,Dmat_order_CTLE.astype(int)]

H_final = np.dot( np.dot(np.transpose(U2), H_Dbasis_CTLE), U2)

#-----Restructure Hamiltonian into LE1, LE2 / CT1, CT2 Subspaces-----
H_final = H_final[Dmat_order.astype(int),:][:,Dmat_order.astype(int)]
#H_final[np.abs(H_final) < 1e-3] = 0
# print(H_final)

#-----Diagonalize Subspaces in the Hamiltonian to finally de-couple states-----
CT1_order = np.arange(0, CT_subspace_CT1.shape[0])
CT2_order = np.arange(CT_subspace_CT1.shape[0], CT_subspace_CT1.shape[0] + CT_subspace_CT2.shape[0])
LE1_order = np.arange(CT_subspace_CT1.shape[0] + CT_subspace_CT2.shape[0], CT_subspace_CT1.shape[0] + CT_subspace_CT2.shape[0] + LE_subspace_LE1.shape[0])
LE2_order = np.arange(CT_subspace_CT1.shape[0] + CT_subspace_CT2.shape[0] + LE_subspace_LE1.shape[0] , CT_subspace_CT1.shape[0] + CT_subspace_CT2.shape[0] + LE_subspace_LE1.shape[0] + LE_subspace_LE2.shape[0])


CT1_subspace = H_final[CT1_order,:][:,CT1_order]
(evals_CT1, evecs_CT1) = np.linalg.eig(CT1_subspace)

CT2_subspace = H_final[CT2_order,:][:,CT2_order]
(evals_CT2, evecs_CT2) = np.linalg.eig(CT2_subspace)

LE1_subspace = H_final[LE1_order,:][:,LE1_order]
(evals_LE1, evecs_LE1) = np.linalg.eig(LE1_subspace)

LE2_subspace = H_final[LE2_order,:][:,LE2_order]
(evals_LE2, evecs_LE2) = np.linalg.eig(LE2_subspace)

#-----Build U3-----
U3 = np.zeros((s0_max, s1_max))
CT1_size = CT1_subspace.shape[0]
CT2_size = CT2_subspace.shape[0]
LE1_size = LE1_subspace.shape[0]
LE2_size = LE2_subspace.shape[0]

U3[0:CT1_size, 0:CT1_size] = evecs_CT1

U3[CT1_size:CT1_size + CT2_size,
   CT1_size:CT1_size + CT2_size] = evecs_CT2

U3[CT1_size + CT2_size:CT1_size + CT2_size + LE1_size,
   CT1_size + CT2_size:CT1_size + CT2_size + LE1_size] = evecs_LE1

U3[CT1_size + CT2_size + LE1_size:CT1_size + CT2_size + LE1_size + LE2_size,
   CT1_size + CT2_size + LE1_size:CT1_size + CT2_size + LE1_size + LE2_size] = evecs_LE2


#-----Diagonalize Submatrixes in Hfinal-----
H_final = np.dot( np.dot(np.transpose(U3), H_final), U3)
H_final[np.abs(H_final) < 1e-3] = 0

#-----Output Energies & the Couplings between Subspaces-----
print(H_final)
print("Submatrix Sizes:\n")
print("CT1_Size: ", CT1_size, "\nCT2_Size: ", CT2_size,"\nLE1_Size: ", LE1_size,"\nLE2_Size: ", LE2_size,)

#-----CT1-Couplings-----
#CT1 - CT2
H_final[:CT1_size, CT1_size: CT1_size + CT2_size]
#CT1 - LE1
H_final[:CT1_size, CT1_size + CT2_size: CT1_size + CT2_size + LE1_size]
#CT1 - LE2
H_final[:CT1_size, CT1_size + CT2_size + LE1_size: CT1_size + CT2_size + LE1_size + LE2_size]

#-----CT2-Couplings-----
#CT2 - LE1
H_final[CT1_size: CT1_size + CT2_size, CT1_size + CT2_size: CT1_size + CT2_size + LE1_size]
#CT2 - LE2
H_final[CT1_size: CT1_size + CT2_size, CT1_size + CT2_size + LE1_size: CT1_size + CT2_size + LE1_size + LE2_size]

#-----LE1-LE2-Couplings-----
H_final[CT1_size + CT2_size: CT1_size + CT2_size + LE1_size, CT1_size + CT2_size + LE1_size: CT1_size + CT2_size + LE1_size + LE2_size]


# mat = np.arange(64).reshape(8,8)
# mat
# mat[0:3,2:]
# mat[3:5,2:]
# #CT1 - CT2
# H_final[CT1_size:CT1_size + CT1_size, :CT1_size]
# # H_final[CT2_size:CT2_size + LE1_size, :CT2_size]
# #CT2 - LE1
# H_final[CT1_size + CT2_size:CT2_size + LE1_size, :CT2_size]

# #CT2 - LE2
# H_final[CT2_size + LE1_size:CT2_size + LE1_size + LE2_size, :CT2_size]

# #LE1 - LE2
# H_final[CT2_size + LE1_size:CT2_size + LE1_size + LE2_size, CT2_size: CT2_size+LE1_size]
# H_final[4:,:2]
# H_final[2:,:2]

# DqFinal[np.abs(DqFinal) < 1e-10] = 0

# a = DxFinal[np.abs(DxFinal) < 1e-10] = 0

# np.sort(np.diag(DxFinal))
# np.sort(np.diag(DqFinal))


# DqDiag = np.diag(DqDbasis)
# DqDiag_size = DqDiag.size
# DqDiag = np.transpose(np.vstack((DqDiag, np.arange(DqDiag.size, dtype=int))))

# index_mapping = DqDiag[np.argsort(DqDiag[:, 0])]
# DqDiag_Restructured = np.zeros((DqDiag_size, DqDiag_size))

# for i in range(DqDiag_size):
#     DqDiag_Restructured[i, i] = index_mapping[i, 0]
# mat = np.arange(16).reshape(4,4)
# print(mat)
# ind = np.array([0,2,3,1])
# mat[ind,:][:,ind]

# ind2 = np.array([0,2])
# ind3 = np.array([3,1])

# mat[ind2,:][:,ind2]
# mat[ind3,:][:,ind3]


