import sys, platform, os
import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import healpy as hp
from astropy.io import fits
import time
import math
from scipy import interpolate
import pickle as pk
import treecorr
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/'

nrad = 20
minrad = 0.8
maxrad = 50.0
do_jk = True
njk_radec = 180
njk_z = 1
njk = njk_radec * njk_z
ds_g = 1
ds_m = 1
ds_g_inp = 1
ds_m_inp = 1
diag_plots = True


bins_all = [1, 2, 3, 4, 5]
# massbin_min = [12.0, 12.5, 13.0, 13.5, 14.0]
# massbin_max = [12.5, 13.0, 13.5, 14.0, 14.5]



filenames_mm = []


save_dir_mm_gm = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/'

for binval in bins_all:

    save_filename_mm = 'mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsm_' + str(ds_m_inp*ds_m) + '_th_nz_v2.pk'

    filenames_mm.append(save_dir_mm_gm + save_filename_mm)


print(len(filenames_mm))

xi_mm_big_combined = np.array([])
xi_mm_true = np.array([])
xi_mm_sigma = np.array([])
r_mm_all = np.array([])

for j in range(len(filenames_mm)):
    filename_mm = filenames_mm[j]

    mm_data = pk.load(open(filename_mm, "rb"))

    xi_mmtruth = mm_data['xi_mm_full']

    r_mm = mm_data['r_mm']
    xi_mmtruth_all = mm_data['ximm_big_all']
    xi_mmtruth_mean = np.tile(xi_mmtruth.transpose(), (njk, 1))
    xi_mmtruth_sub = xi_mmtruth_all - xi_mmtruth_mean
    xi_mmtruth_sigma = np.sqrt((1.0 * (njk - 1.) / njk) * (np.sum(np.square(xi_mmtruth_all - xi_mmtruth_mean), axis=0)))

    if len(xi_mm_big_combined) == 0:
        xi_mm_big_combined = xi_mmtruth_sub
    else:
        xi_mm_big_combined = np.hstack((xi_mm_big_combined, xi_mmtruth_sub))

    if len(xi_mm_big_combined) == 0:
        xi_mm_true = xi_mmtruth
        xi_mm_sigma = xi_mmtruth_sigma
    else:
        xi_mm_true = np.hstack((xi_mm_true, xi_mmtruth))
        xi_mm_sigma = np.hstack((xi_mm_sigma, xi_mmtruth_sigma))

    if len(r_mm_all) == 0:
        r_mm_all = r_mm
    else:
        r_mm_all = np.vstack((r_mm_all, r_mm))

xi_big_combined = xi_mm_big_combined
cov_combined = (1.0 * (njk - 1.) / njk) * np.matmul(xi_big_combined.T, xi_big_combined)
xi_true = xi_mm_true
xi_sigma_true = xi_mm_sigma
r_all = r_mm_all

th_f = np.load('/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/mice_xi_mm_all_bin.npz')
r_theory,xi_mm_theory = th_f['r'], th_f['xi']
nrad_th = len(r_theory)
# pdb.set_trace()

sig_diag = np.sqrt(np.diag(cov_combined))
nbins = len(bins_all)
fig, ax = plt.subplots(1, nbins, figsize=(nbins*4, 5), sharey=True)
for j in range(len(bins_all)):
    ax[j].errorbar(r_all[j], xi_mm_true[j * nrad:(j + 1) * nrad], sig_diag[j * nrad:(j + 1) * nrad], color='blue',
                   marker='*', linestyle='', label=r'Data')
    ax[j].plot(r_theory, xi_mm_theory[j * nrad_th:(j + 1) * nrad_th], color='red',
                   marker='', linestyle='-', label=r'Theory')
    ax[j].set_xlim(0.7,50.0)
    ax[j].set_ylim(5e-3,20.0)
    ax[j].set_yscale('log')
    ax[j].set_xscale('log')
    ax[j].set_xlabel(r'R  (Mpc/h)', size=16)
    ax[j].tick_params(axis='both', which='major', labelsize=15)
    ax[j].tick_params(axis='both', which='minor', labelsize=15)

ax[0].set_ylabel(r'$\xi_{\rm mm}(R)$ comparison', size=22)
ax[0].legend(fontsize=20)
plt.tight_layout()

plt.savefig(save_dir_mm_gm + 'xi_mm_MICE_comp_v2_fullsample.png')
plt.close()

