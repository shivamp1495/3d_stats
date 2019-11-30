import sys, os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as integrate
import random
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from numpy.random import rand
import pickle as pk
import matplotlib.cm as cm
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import pdb
import time
import multiprocessing as mp
import argparse
import mycosmo as cosmodef

colors = ['r', 'b', 'k', 'orange', 'magenta']

stat_type = 'gg'
# stat_type = 'gg_mm'


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in xrange(0, cov.shape[0]):
        for jj in xrange(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr


class general_funcs:

    def __init__(self, cosmo_params):
        h = cosmo_params['H0'] / 100.
        cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                          cosmo_params['sigma8'])
        self.cosmo = cosmo_func

    def get_Dcom(self, zf):
        c = 3 * 10 ** 5
        Omega_m, Omega_L = self.cosmo.Om0, 1. - self.cosmo.Om0
        res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        return Dcom

    def get_diff(self, zf, chi):
        return chi - self.get_Dcom(zf)

    def root_find(self, init_x, chi):
        nll = lambda *args: self.get_diff(*args)
        result = op.root(nll, np.array([init_x]), args=chi, options={'maxfev': 50}, tol=0.01)
        return result.x[0]

    def get_z_from_chi(self, chi):
        valf = self.root_find(0., chi)
        return valf

    def get_vol(self, zmin, zmax, fsky):
        chimin, chimax = self.get_Dcom(zmin), self.get_Dcom(zmax)
        Vmin = (4. / 3.) * np.pi * chimin ** 3
        Vmax = (4. / 3.) * np.pi * chimax ** 3
        vol = (Vmax - Vmin) * fsky
        return vol


cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
fsky_mice = 1. / 8.

gnf = general_funcs(cosmo_params_dict)

ds_m_inp = 2
njk_radec = 180
njk_z = 1
njk = njk_radec * njk_z
save_plot_dir = '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/plots/test_theory_cov/'

load_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/'
load_filename_matter = 'matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_inp) + '.fits'
load_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_mice2_des_run_redmapper_v6.4.16_redmagic_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '.fits'

print 'loading g'
load_cat_g = fits.open(load_dir + load_filename_galaxy)

print 'loading m'
load_cat_m = fits.open(load_dir + load_filename_matter)

ra_g_all, dec_g_all, r_g_all, z_g_all, bin_g_all, jk_g_all = load_cat_g[1].data['RA'], load_cat_g[1].data['DEC'], \
                                                             load_cat_g[1].data['R'], \
                                                             load_cat_g[1].data['Z'], load_cat_g[1].data['BIN'], \
                                                             load_cat_g[1].data['JK']

ra_m_all, dec_m_all, r_m_all, z_m_all, bin_m_all, jk_m_all = load_cat_m[1].data['RA'], load_cat_m[1].data['DEC'], \
                                                             load_cat_m[1].data['R'], \
                                                             load_cat_m[1].data['Z'], load_cat_m[1].data['BIN'], \
                                                             load_cat_m[1].data['JK']

zmin_bins = [0.15, 0.3, 0.45, 0.6, 0.75]
zmax_bins = [0.3, 0.45, 0.6, 0.75, 0.9]

save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/'
minrad = 0.8
maxrad = 50.0
nrad = 20
file_jk_gg_gm = pk.load(open(save_dir + 'gg_gm_datavec_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(
    nrad) + '_zbin_1_2_3_4_5_jk_True_njk_' + str(njk) + '.pk', 'rb'))
file_jk_gg_mm__gm_mm = pk.load(open(
    save_dir + 'gg_mm__gm_mm_datavec_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(
        nrad) + '_zbin_1_2_3_4_5_jk_True_njk_' + str(njk) + '.pk', 'rb'))

if stat_type == 'gg':
    r_gg_data = file_jk_gg_gm['sep'][0]
    nr = len(r_gg_data)
    xi_gg_data = file_jk_gg_gm['mean'][0:5 * nr]
    cov_gg_data = file_jk_gg_gm['cov'][0:5 * nr, 0:5 * nr]

if stat_type == 'gg_mm':
    r_gg_data = file_jk_gg_mm__gm_mm['sep'][0]
    nr = len(r_gg_data)
    xi_gg_data = file_jk_gg_mm__gm_mm['mean'][0:5 * nr]
    cov_gg_data = file_jk_gg_mm__gm_mm['cov'][0:5 * nr, 0:5 * nr]

r_array = r_gg_data

pk_dir = '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/'

bins_all = [1, 2, 3, 4, 5]
# bins_all = [3]
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
for j in range(len(bins_all)):
    binval = bins_all[j]
    ind_bin_g = np.where(bin_g_all == binval)[0]
    ind_bin_m = np.where(bin_m_all == binval)[0]

    ra_g, dec_g, r_g, z_g, bin_g, jk_g = ra_g_all[ind_bin_g], dec_g_all[ind_bin_g], r_g_all[ind_bin_g], z_g_all[
        ind_bin_g], bin_g_all[ind_bin_g], jk_g_all[ind_bin_g]

    ra_m, dec_m, r_m, z_m, bin_m, jk_m = ra_m_all[ind_bin_m], dec_m_all[ind_bin_m], r_m_all[ind_bin_m], z_m_all[
        ind_bin_m], bin_m_all[ind_bin_m], jk_m_all[ind_bin_m]

    num_g, num_m = len(ra_g), len(ra_m)

    vol_bin = gnf.get_vol(zmin_bins[binval - 1], zmax_bins[binval - 1], fsky_mice)

    # get number density
    nbar_g, nbar_m = num_g / vol_bin, num_m / vol_bin

    print 'doing bin ', binval, ', nbar_g', nbar_g

    # load Pk for gg, gm and mm
    Pk_gg_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/Pk_gg_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')
    Pk_gm_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/Pk_gm_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')
    Pk_mm_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/Pk_mm_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')

    karray = Pk_gg_file['k']
    nk = len(karray)
    Pk_gg, Pk_gm, Pk_mm = Pk_gg_file['xi'], Pk_gm_file['xi'], Pk_mm_file['xi']
    k_mat = np.tile(karray.reshape(1, 1, nk), (nr, nr, 1))
    Pk_gg_mat = np.tile(Pk_gg.reshape(1, 1, nk), (nr, nr, 1))

    # load xi for gg, gm and mm
    xi_gg_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/xi_gg_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')
    xi_gm_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/xi_gm_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')
    xi_mm_file = np.load(
        '/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/measurements/xi_mm_total_bin_' + str(
            binval) + '_MICE_cosmogg_mm__gm_mm_nocov_crosszbinsgmgg_False_crosszbinsall_False_gmgg_False_covdiag_False_njk_180.npz')
    r_array_theory = xi_gg_file['r']
    xi_gg_th, xi_gm_th, xi_mm_th = xi_gg_file['xi'], xi_gm_file['xi'], xi_mm_file['xi']
    xi_gg_temp = intspline(r_array_theory, xi_gg_th)
    xi_gg = xi_gg_temp(r_gg_data)
    xi_gg_mat_diag = np.diag(xi_gg)

    xi_mm_temp = intspline(r_array_theory, xi_mm_th)
    xi_mm = xi_mm_temp(r_gg_data)
    xi_mm_mat_diag = np.diag(xi_mm)

    if stat_type == 'gg_mm':
        inv_xi_mm1 = np.tile((1. / xi_mm).reshape(nr, 1), (1, nr))
        inv_xi_mm2 = np.tile((1. / xi_mm).reshape(1, nr), (nr, 1))
        factor_mult = inv_xi_mm1 * inv_xi_mm2

    if stat_type == 'gg':
        factor_mult = 1.

    r1_mat = np.tile(r_array.reshape(nr, 1, 1), (1, nr, nk))
    r2_mat = np.tile(r_array.reshape(1, nr, 1), (nr, 1, nk))
    rdiag_mat = np.diag(r_array)

    integrand1 = (k_mat ** 2) * (np.sin(k_mat * r1_mat) / (k_mat * r1_mat)) * (
            np.sin(k_mat * r2_mat) / (k_mat * r2_mat)) * (Pk_gg_mat ** 2)
    T1 = factor_mult * (1. / (vol_bin * np.pi ** 2)) * sp.integrate.simps(integrand1, karray)

    integrand2 = (k_mat ** 2) * (np.sin(k_mat * r1_mat) / (k_mat * r1_mat)) * (
            np.sin(k_mat * r2_mat) / (k_mat * r2_mat)) * Pk_gg_mat
    T2 = factor_mult * (2. / (vol_bin * nbar_g * np.pi ** 2)) * sp.integrate.simps(integrand2, karray)

    T3 = factor_mult * np.diag((1. / (vol_bin * nbar_g ** 2)) * (2. / (4. * np.pi * r_array ** 2)))

    T4 = factor_mult * np.diag((1. / (vol_bin * nbar_g ** 2)) * (2. / (4. * np.pi * r_array ** 2)) * xi_gg)

    cov_total = T1 + T2 + T3 + T4

    # pdb.set_trace()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(T1)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_T1_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(T2)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_T2_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(T3)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_T3_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(T4)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_T4_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(cov_total)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_total_theory_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    cov_data = cov_gg_data[(binval - 1) * nr:binval * nr, (binval - 1) * nr:binval * nr]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(np.log(np.abs(cov_data)))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'logabs_cov_total_data_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(get_corr(cov_total), clim=(-1, 1))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'corr_total_theory_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    cov_data = cov_gg_data[(binval - 1) * nr:binval * nr, (binval - 1) * nr:binval * nr]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    corr = ax.imshow(get_corr(cov_data), clim=(-1, 1))
    fig.colorbar(corr, ax=ax)
    fig.tight_layout()
    fig.savefig(save_plot_dir + 'corr_total_data_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(r_array, np.diag(cov_data), color='red', ls='-', label='Data JK')
    ax.plot(r_array, np.diag(cov_total), color='blue', ls='-', label='Theory Total')
    ax.plot(r_array, np.diag(T1), color='orange', ls='-.', lw=1., label='T1')
    ax.plot(r_array, np.diag(T2), color='green', ls='--', lw=1., label='T2')
    ax.plot(r_array, np.diag(T3), color='black', ls=':', lw=1., label='T3')
    ax.plot(r_array, np.diag(T4), color='magenta', ls=':', lw=1., label='T4')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'R  (Mpc/h)', size=17)
    if stat_type == 'gg':
        ax.set_ylabel(r'$\mathrm{\mathbb{Cov}}(\xi_{gg}(R),\xi_{gg}(R))$', size=17)
    if stat_type == 'gg_mm':
        ax.set_ylabel(r'$\mathrm{\mathbb{Cov}}(\xi_{gg/mm}(R),\xi_{gg/mm}(R))$', size=17)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.legend(fontsize=15, frameon=False)
    plt.tight_layout()
    fig.savefig(save_plot_dir + 'cov_comp_bin' + str(binval) + '_' + stat_type + '.png', dpi=240)
    plt.close()

    ax1.plot(r_array, np.diag(cov_data), color=colors[j], ls='-', label='Bin' + str(binval))
    ax1.plot(r_array, np.diag(cov_total), color=colors[j], ls='--')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'R  (Mpc/h)', size=17)
if stat_type == 'gg':
    ax1.set_ylabel(r'$\mathrm{\mathbb{Cov}}(\xi_{gg}(R),\xi_{gg}(R))$', size=17)
if stat_type == 'gg_mm':
    ax1.set_ylabel(r'$\mathrm{\mathbb{Cov}}(\xi_{gg/mm}(R),\xi_{gg/mm}(R))$', size=17)
ax1.set_xlim(3.9, 45.)
xticks = [4, 10, 20, 40]
ax1.set_xticks(xticks)
labels = [xticks[i] for i, t in enumerate(xticks)]
ax1.set_xticklabels(labels)
if stat_type == 'gg':
    ax1.set_ylim(9e-7, 1e-2)
    ax1.legend(fontsize=14, frameon=False, loc='upper right')
if stat_type == 'gg_mm':
    ax1.set_ylim(9e-5, 5e-1)
    ax1.legend(fontsize=13, frameon=False, loc='lower right')
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='both', which='minor', labelsize=15)

plt.tight_layout()
fig1.savefig(save_plot_dir + 'cov_comp_allbins' + '_' + stat_type + '.png', dpi=240)
plt.close()
