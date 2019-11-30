import sys, platform, os
import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
# import healpy as hp
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
njk_radec = 300
njk_z = 1
njk = njk_radec * njk_z
ds_g = 1
ds_m = 1
ds_g_inp = 1
ds_m_inp = 1.67
diag_plots = True


bins_all = [1, 2, 3, 4, 5]
# massbin_min = [12.0, 12.5, 13.0, 13.5, 14.0]
# massbin_max = [12.5, 13.0, 13.5, 14.0, 14.5]



filenames_gg = []
filenames_gm = []
filenames_mm = []
filenames_gg_mm = []
filenames_gm_mm = []

save_dir_gg_gm = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/'

for binval in bins_all:

    save_filename_gg = 'gg_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '_v2.pk'
    save_filename_mm = 'mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsm_' + str(ds_m_inp*ds_m) + '_th_nz_v2.pk'
    save_filename_gm = 'gm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '_v2.pk'
    save_filename_gg_mm = 'gg_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '_v2.pk'
    save_filename_gm_mm = 'gm_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '_v2.pk'

    filenames_gg.append(save_dir_gg_gm + save_filename_gg)
    filenames_mm.append(save_dir_gg_gm + save_filename_mm)
    filenames_gm.append(save_dir_gg_gm + save_filename_gm)
    filenames_gg_mm.append(save_dir_gg_gm + save_filename_gg_mm)
    filenames_gm_mm.append(save_dir_gg_gm + save_filename_gm_mm)

print(len(filenames_gg))

xi_gg_big_combined = np.array([])
xi_gm_big_combined = np.array([])

xi_gg_true = np.array([])
xi_gm_true = np.array([])

xi_gg_sigma = np.array([])
xi_gm_sigma = np.array([])

r_gg_all = np.array([])
r_gm_all = np.array([])

for j in range(len(filenames_gg)):
    print(j)
    filename_gg = filenames_gg[j]
    filename_gm = filenames_gm[j]

    gg_data = pk.load(open(filename_gg, "rb"))
    gm_data = pk.load(open(filename_gm, "rb"))

    xi_ggtruth = gg_data['xi_gg_full']
    xi_gmtruth = gm_data['xi_gm_full']

    r_gm = gm_data['r_gm']
    xi_gmtruth_all = gm_data['xigm_big_all']
    xi_gmtruth_mean = np.tile(xi_gmtruth.transpose(), (njk, 1))
    xi_gmtruth_sub = xi_gmtruth_all - xi_gmtruth_mean
    xi_gmtruth_sigma = np.sqrt((1.0 * (njk - 1.) / njk) * (np.sum(np.square(xi_gmtruth_all - xi_gmtruth_mean), axis=0)))

    r_gg = gg_data['r_gg']
    xi_ggtruth_all = gg_data['xigg_big_all']
    xi_ggtruth_mean = np.tile(xi_ggtruth.transpose(), (njk, 1))
    xi_ggtruth_sub = xi_ggtruth_all - xi_ggtruth_mean
    xi_ggtruth_sigma = np.sqrt((1.0 * (njk - 1.) / njk) * (np.sum(np.square(xi_ggtruth_all - xi_ggtruth_mean), axis=0)))

    if len(xi_gg_big_combined) == 0:
        xi_gg_big_combined = xi_ggtruth_sub
        xi_gm_big_combined = xi_gmtruth_sub
    else:
        xi_gg_big_combined = np.hstack((xi_gg_big_combined, xi_ggtruth_sub))
        xi_gm_big_combined = np.hstack((xi_gm_big_combined, xi_gmtruth_sub))

    if len(xi_gg_big_combined) == 0:
        xi_gg_true = xi_ggtruth
        xi_gm_true = xi_gmtruth
        xi_gm_sigma = xi_gmtruth_sigma
        xi_gg_sigma = xi_ggtruth_sigma
    else:
        xi_gg_true = np.hstack((xi_gg_true, xi_ggtruth))
        xi_gm_true = np.hstack((xi_gm_true, xi_gmtruth))
        xi_gm_sigma = np.hstack((xi_gm_sigma, xi_gmtruth_sigma))
        xi_gg_sigma = np.hstack((xi_gg_sigma, xi_ggtruth_sigma))

    if len(r_gg_all) == 0:
        r_gg_all = r_gg
        r_gm_all = r_gm
    else:
        r_gg_all = np.vstack((r_gg_all, r_gg))
        r_gm_all = np.vstack((r_gm_all, r_gm))

xi_big_combined = np.hstack((xi_gg_big_combined, xi_gm_big_combined))
cov_combined = (1.0 * (njk - 1.) / njk) * np.matmul(xi_big_combined.T, xi_big_combined)
xi_true = np.hstack((xi_gg_true, xi_gm_true))
xi_sigma_true = np.hstack((xi_gg_sigma, xi_gm_sigma))
r_all = np.vstack((r_gg_all, r_gm_all))

results_dict = {}
results_dict['sep'] = r_all
results_dict['mean'] = xi_true
results_dict['cov'] = cov_combined

filename_save = save_dir_gg_gm + 'gg_gm_datavec_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(
    nrad) + '_zbin_1_2_3_4_5_jk_True_njk_' + str(njk) + '_v2_fullsample.pk'
pk.dump(results_dict, open(filename_save, 'wb'), protocol=2)

xi_gg_mm_big_combined = np.array([])
xi_gm_mm_big_combined = np.array([])

xi_gg_mm_true = np.array([])
xi_gm_mm_true = np.array([])

xi_gg_mm_sigma = np.array([])
xi_gm_mm_sigma = np.array([])

r_gg_mm_all = np.array([])
r_gm_mm_all = np.array([])

for j in range(len(filenames_gg_mm)):
    filename_gg_mm = filenames_gg_mm[j]
    filename_gm_mm = filenames_gm_mm[j]

    gg_data = pk.load(open(filename_gg_mm, "rb"))
    gm_data = pk.load(open(filename_gm_mm, "rb"))

    xi_gg_mmtruth = gg_data['xi_gg_mm_full']
    xi_gm_mmtruth = gm_data['xi_gm_mm_full']

    r_gm_mm = gm_data['r_gm']
    xi_gm_mmtruth_all = gm_data['xi_gm_mm_big_all']
    xi_gm_mmtruth_mean = np.tile(xi_gm_mmtruth.transpose(), (njk, 1))
    xi_gm_mmtruth_sub = xi_gm_mmtruth_all - xi_gm_mmtruth_mean
    xi_gm_mmtruth_sigma = np.sqrt(
        (1.0 * (njk - 1.) / njk) * (np.sum(np.square(xi_gm_mmtruth_all - xi_gm_mmtruth_mean), axis=0)))

    r_gg_mm = gg_data['r_gg']
    xi_gg_mmtruth_all = gg_data['xi_gg_mm_big_all']
    xi_gg_mmtruth_mean = np.tile(xi_gg_mmtruth.transpose(), (njk, 1))
    xi_gg_mmtruth_sub = xi_gg_mmtruth_all - xi_gg_mmtruth_mean
    xi_gg_mmtruth_sigma = np.sqrt(
        (1.0 * (njk - 1.) / njk) * (np.sum(np.square(xi_gg_mmtruth_all - xi_gg_mmtruth_mean), axis=0)))

    if len(xi_gg_mm_big_combined) == 0:
        xi_gg_mm_big_combined = xi_gg_mmtruth_sub
        xi_gm_mm_big_combined = xi_gm_mmtruth_sub
    else:
        xi_gg_mm_big_combined = np.hstack((xi_gg_mm_big_combined, xi_gg_mmtruth_sub))
        xi_gm_mm_big_combined = np.hstack((xi_gm_mm_big_combined, xi_gm_mmtruth_sub))

    if len(xi_gg_mm_big_combined) == 0:
        xi_gg_mm_true = xi_gg_mmtruth
        xi_gm_mm_true = xi_gm_mmtruth
        xi_gm_mm_sigma = xi_gm_mmtruth_sigma
        xi_gg_mm_sigma = xi_gg_mmtruth_sigma
    else:
        xi_gg_mm_true = np.hstack((xi_gg_mm_true, xi_gg_mmtruth))
        xi_gm_mm_true = np.hstack((xi_gm_mm_true, xi_gm_mmtruth))
        xi_gm_mm_sigma = np.hstack((xi_gm_mm_sigma, xi_gm_mmtruth_sigma))
        xi_gg_mm_sigma = np.hstack((xi_gg_mm_sigma, xi_gg_mmtruth_sigma))

    if len(r_gg_mm_all) == 0:
        r_gg_mm_all = r_gg_mm
        r_gm_mm_all = r_gm_mm
    else:
        r_gg_mm_all = np.vstack((r_gg_mm_all, r_gg_mm))
        r_gm_mm_all = np.vstack((r_gm_mm_all, r_gm_mm))

xi_ratio_big_combined = np.hstack((xi_gg_mm_big_combined, xi_gm_mm_big_combined))
cov_ratio_combined = (1.0 * (njk - 1.) / njk) * np.matmul(xi_ratio_big_combined.T, xi_ratio_big_combined)
xi_ratio_true = np.hstack((xi_gg_mm_true, xi_gm_mm_true))
xi_ratio_sigma_true = np.hstack((xi_gg_mm_sigma, xi_gm_mm_sigma))
r_ratio_all = np.vstack((r_gg_mm_all, r_gm_mm_all))

results_dict = {}
results_dict['sep'] = r_ratio_all
results_dict['mean'] = xi_ratio_true
results_dict['cov'] = cov_ratio_combined

print(cov_ratio_combined.shape)

filename_save = save_dir_gg_gm + 'gg_mm__gm_mm_datavec_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(
    nrad) + '_zbin_1_2_3_4_5_jk_True_njk_' + str(njk) + '_v2_fullsample.pk'
pk.dump(results_dict, open(filename_save, 'wb'), protocol=2)


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in range(0, cov.shape[0]):
        for jj in range(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr


if diag_plots:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # corr = ax.imshow(np.log(np.abs(get_corr(cov_combined))),clim=(-12, 0.0))
    # fig.colorbar(corr, ax=ax)
    # fig.tight_layout()
    # fig.savefig(save_dir + 'gg_gm_3d_corr_mat_log.png')

    corr = ax.imshow(((get_corr(cov_combined))), clim=(-1.0, 1.0))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()

    fig.savefig(save_dir_gg_gm + 'gg_gm_3d_corr_mat_lin_v2_njk' + str(njk) + '.png')

    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # corr = ax.imshow(np.log(np.abs(get_corr(cov_ratio_combined))),clim=(-12, 0.0))
    # fig.colorbar(corr, ax=ax)
    # fig.tight_layout()
    # fig.savefig(save_dir + 'gg_mm__gm_mm_3d_corr_mat_log.png')

    corr = ax.imshow(((get_corr(cov_ratio_combined))), clim=(-1, 1.0))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()

    fig.savefig(save_dir_gg_gm + 'gg_mm__gm_mm_3d_corr_mat_lin_v2_njk' + str(njk) + '.png')

    plt.close()

    sig_diag = np.sqrt(np.diag(cov_combined))
    nbins = len(bins_all)
    fig, ax = plt.subplots(1, nbins, figsize=(nbins*4, 5), sharey=True)
    for j in range(len(bins_all)):
        ax[j].errorbar(r_all[j], xi_gg_true[j * nrad:(j + 1) * nrad], sig_diag[j * nrad:(j + 1) * nrad], color='blue',
                       marker='*', linestyle='', label=r'$\xi_{gg}$')
        ax[j].errorbar(r_all[j], xi_gm_true[j * nrad:(j + 1) * nrad],
                       sig_diag[j * nrad + (nrad * nbins):(j + 1) * nrad + (nrad * nbins)],
                       color='red', marker='*', linestyle='', label=r'$\xi_{gm}$')
        ax[j].set_yscale('log')
        ax[j].set_xscale('log')
        ax[j].set_xlabel(r'R  (Mpc/h)', size=16)
        ax[j].tick_params(axis='both', which='major', labelsize=15)
        ax[j].tick_params(axis='both', which='minor', labelsize=15)

    ax[0].set_ylabel(r'$\xi(R)$ comparison', size=22)
    ax[0].legend(fontsize=20)
    plt.tight_layout()

    plt.savefig(save_dir_gg_gm + 'xi_gg_gm_allbins_comp_v2_fullsample_njk' + str(njk) + '.png')
    plt.close()

    nbins = len(bins_all)
    sig_ratio_diag = np.sqrt(np.diag(cov_ratio_combined))
    fig, ax = plt.subplots(1, nbins, figsize=(nbins*4, 5), sharey=True)
    for j in range(len(bins_all)):
        ax[j].errorbar(r_ratio_all[j], xi_gg_mm_true[j * nrad:(j + 1) * nrad], sig_ratio_diag[j * nrad:(j + 1) * nrad],
                       color='blue',
                       marker='*', linestyle='', label=r'$\xi_{gg}/\xi_{mm}$')
        ax[j].errorbar(r_ratio_all[j], xi_gm_mm_true[j * nrad:(j + 1) * nrad],
                       sig_ratio_diag[j * nrad + (nrad * nbins):(j + 1) * nrad + (nrad * nbins)],
                       color='red', marker='*', linestyle='', label=r'$\xi_{gm}/\xi_{mm}$')
        ax[j].set_xscale('log')
        ax[j].set_xlabel(r'R  (Mpc/h)', size=16)
        ax[j].tick_params(axis='both', which='major', labelsize=15)
        ax[j].tick_params(axis='both', which='minor', labelsize=15)

    ax[0].set_ylabel(r'$\xi(R)$ comparison', size=22)
    ax[0].legend(fontsize=20)
    plt.tight_layout()

    plt.savefig(save_dir_gg_gm + 'xi_gg_mm__gm_mm_allbins_comp_v2_fullsample_njk' + str(njk) + '.png')
    plt.close()


    nbins = len(bins_all)
    sig_ratio_diag = np.sqrt(np.diag(cov_ratio_combined))
    fig, ax = plt.subplots(1, 1, figsize=(8,6), sharey=True)
    colors = ['r','b','orange','green','magenta']
    for j in range(len(bins_all)):
        # if j == 0:
        ax.errorbar(r_ratio_all[j]*(1.01)**j, np.sqrt(xi_gg_mm_true[j * nrad:(j + 1) * nrad]), np.sqrt(sig_ratio_diag[j * nrad:(j + 1) * nrad]),
                       color=colors[j],marker='*', linestyle='', label=r'Bin ' + str(j+1))
        ax.errorbar(r_ratio_all[j]*(1.02)**j, xi_gm_mm_true[j * nrad:(j + 1) * nrad],
                       sig_ratio_diag[j * nrad + (nrad * nbins):(j + 1) * nrad + (nrad * nbins)],
                       color=colors[j], marker='o', linestyle='')
        # else:
        #     ax.errorbar(r_ratio_all[j], xi_gg_mm_true[j * nrad:(j + 1) * nrad], sig_ratio_diag[j * nrad:(j + 1) * nrad],
        #                    color=colors[j],marker='*', linestyle='')
        #     ax.errorbar(r_ratio_all[j]*1.05, xi_gm_mm_true[j * nrad:(j + 1) * nrad],
        #                    sig_ratio_diag[j * nrad + (nrad * nbins):(j + 1) * nrad + (nrad * nbins)],
        #                    color=colors[j], marker='o', linestyle='')


    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xscale('log')
    ax.set_ylabel(r'$\xi(R)$ ratio', size=22)
    ax.legend(fontsize=14,ncol=2)
    ax.set_xlabel(r'R  (Mpc/h)', size=18)
    plt.tight_layout()

    plt.savefig(save_dir_gg_gm + 'xi_gg_mm__gm_mm_allbins_comp_onepanel_v2_fullsample_njk' + str(njk) + '.png')
    plt.close()
