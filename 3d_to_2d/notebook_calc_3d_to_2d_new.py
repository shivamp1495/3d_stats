import sys, platform, os
import numpy as np
import scipy as sp
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
import pk_to_xi_new as ptx

likef = __import__('3d_like_toimp')
import copy
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import multiprocessing

pklin_file = 'pkz_lin_MICE_cosmo.npz'
pknl_file = 'pkz_nl_MICE_cosmo.npz'
nz_dir = '/home/shivam/Research/cosmosis/y3kp-bias-model/3d_stats/3d_to_2d/src/nz_data/'
data_file = '/media/shivam/shivam_backup/Dropbox/research/Cori_files/data_project/mice_sims/measurements/gg_mm__gm_mm_datavec_3dcorr_r_0.8_50.0_nr_20_zbin_1_2_3_4_5_jk_True_njk_180.pk'
twopt_file = fits.open('/home/shivam/Research/cosmosis/y3kp-bias-model/simulations/y1/mice/tpt_rot1_vY1_sn_wcov.fits')

data = pk.load(open(data_file, 'rb'))
r_obs, data_obs, cov_obs = data['sep'], data['mean'], data['cov']
bins_all = [1, 2, 3, 4, 5]
bins_to_fit = [2]
bin_source = 4

bins_to_rem = copy.deepcopy(bins_all)
for bins in bins_to_fit:
    bins_to_rem.remove(bins)

stat_type = 'gg_mm__gm_mm'
r_obs_new, data_obs_new, cov_obs_new = likef.import_data(r_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all,stat_type)

k_hres_min = 1e-4
k_hres_max = 500
n_k_hres_bin = 10000

k_hres = np.logspace(np.log10(k_hres_min), np.log10(k_hres_max), n_k_hres_bin)
reg_c = 10.0

output_nl_grid = True
pt_type = 'oneloop_eul_bk'


Pkz_lin_f = np.load(pklin_file)
Pkz_nl_f = np.load(pknl_file)
klin, zlin_orig, Pkzlin_orig = Pkz_lin_f['k'], Pkz_lin_f['z'], Pkz_lin_f['pkz']
knl, znl_orig, Pnl_kz_orig = Pkz_nl_f['k'], Pkz_nl_f['z'], Pkz_nl_f['pkz']

z_array = np.linspace(0.001,1.5,4000)
Pkzlin = np.zeros((len(z_array),Pkzlin_orig.shape[1]))
Pnl_kz = np.zeros((len(z_array),Pnl_kz_orig.shape[1]))
for j in range(len(klin)):
    Pkzlin_interp = interpolate.interp1d(np.log(zlin_orig + 1e-80),np.log(Pkzlin_orig[:,j]),fill_value='extrapolate')
    Pkzlin[:,j] = np.exp(Pkzlin_interp(np.log(z_array + 1e-80)))

for j in range(len(knl)):
    Pkznl_interp = interpolate.interp1d(np.log(znl_orig + 1e-80), np.log(Pnl_kz_orig[:, j]), fill_value='extrapolate')
    Pnl_kz[:, j] = np.exp(Pkznl_interp(np.log(z_array + 1e-80)))

znl = z_array
zlin = z_array
# Pkzlin = np.array([Pkzlin[0,:]])
# Pnl_kz = np.array([Pnl_kz[0,:]])

Pk_terms_names = ['Plin', 'Pmm', 'Pd1d2', 'Pd2d2', 'Pd1s2', 'Pd2s2', 'Ps2s2', 'Pd1d3nl', 'k2Pk', 'sig4']

Pkth_array, karray, xi_all, r_array = ptx.get_Pktharray(output_nl_grid, klin, knl, Pkzlin, Pnl_kz,pt_type=pt_type,Pk_terms_names = Pk_terms_names, z_array=znl, output_xi=True, use_fftlog=False)

Pkth_array_khres = np.zeros((len(Pk_terms_names), len(znl), len(k_hres)))

do_regularize_pk = True
do_reg_all = False
for j1 in range(len(Pk_terms_names)):

    print 'processing Pk ' + str(Pk_terms_names[j1])

    P_gg_khres = np.zeros((len(znl), len(k_hres)))
    for i in range(len(znl)):
        P_gg_j1_i = Pkth_array[j1][i, :]

        Pgg_temp = intspline(karray, P_gg_j1_i)
        Pgg_term_interp = Pgg_temp(k_hres)
        P_gg_khres[i, :] = Pgg_term_interp

    if Pk_terms_names[j1] == 'Plin':
        P_gg_khres_reg = P_gg_khres
    else:
        if do_regularize_pk:
            if do_reg_all:
                # P_gg_khres_reg = reg_Pk(P_gg_khres, P_lin_khres, k_hres, reg_k, c_val=reg_c)
                P_gg_khres_reg = ptx.reg_Pk_mat_expmult(P_gg_khres, k_hres, reg_c)
            else:
                if Pk_terms_names[j1] == 'k2Pk':
                    # P_gg_khres_reg = reg_Pk(P_gg_khres, P_lin_khres, k_hres, reg_k, c_val=reg_c)
                    # P_gg_khres_reg = reg_Pk_gaussian(P_gg_khres, k_hres, reg_k, c_val=reg_c)
                    P_gg_khres_reg = ptx.reg_Pk_mat_expmult(P_gg_khres, k_hres, reg_c)
                else:
                    P_gg_khres_reg = P_gg_khres
        else:
            P_gg_khres_reg = P_gg_khres

    Pkth_array_khres[j1, :, :] = P_gg_khres_reg

Pk_mm = Pkth_array_khres[1]
z_pk = znl

param_name = ['b1E', 'b2E', 'bsE', 'b3nlE', 'bkE']
# param_array_bin1 = [2.0,1.0,0.8,0.5,-1.0]

if bins_to_fit[0] == 1:
    b1E_def = 1.3848850293899444
    b2E_def = 0.0554863414599977

if bins_to_fit[0] == 2:
    b1E_def = 1.4325550216315344
    b2E_def = 0.15655541801620562

if bins_to_fit[0] == 3:
    b1E_def = 1.557122472799171
    b2E_def = 0.21403674350409543

if bins_to_fit[0] == 4:
    b1E_def = 1.8177591949456122
    b2E_def = 0.5070993543581124

if bins_to_fit[0] == 5:
    b1E_def = 2.4197471830035955
    b2E_def = 1.1396069563358795

param_array_bin1 = [b1E_def,b2E_def,(-4./7.)*(b1E_def-1),(b1E_def - 1),0.0]
param_array_bin2 = param_array_bin1

Pk_gg, _ = ptx.get_PXX_terms_bins(param_array_bin1, param_array_bin2, Pkth_array_khres,pt_type=pt_type)
Pk_gm, _ = ptx.get_PXm_terms(param_array_bin1, Pkth_array_khres,pt_type=pt_type)

xi_gg, xi_gg_terms = ptx.get_xiXX_terms_bins(param_array_bin1, param_array_bin2, xi_all,pt_type=pt_type)
xi_gm, xi_gm_terms = ptx.get_xiXm_terms(param_array_bin1, xi_all,pt_type=pt_type)
xi_mm = xi_all[1]

def get_nz_lens():
    filename_nzlens = nz_dir + 'nz_g_m_' + '_zbin_' + str(bins_to_fit[0]) + '_dsg_' + str(1) + '_dsm_' + str(1) + '.pk'
    nz_data = pk.load(open(filename_nzlens, 'rb'))
    nz_g, nz_m, nz_z = nz_data['nz_g'], nz_data['nz_m'], nz_data['nz_z']
    return nz_g, nz_m, nz_z

def get_nz_lens_2pt_pz():
    z_mid = twopt_file['nz_pos_zrm'].data['Z_MID']
    bin4 = twopt_file['nz_pos_zrm'].data['BIN' + str(bins_to_fit[0])]
    int_b4 = sp.integrate.simps(bin4,z_mid)
    nz_b4 = bin4/int_b4
    return z_mid, nz_b4

def get_nz_lens_2pt_specz():
    z_mid = twopt_file['nz_pos_zspec'].data['Z_MID']
    bin4 = twopt_file['nz_pos_zspec'].data['BIN' + str(bins_to_fit[0])]
    int_b4 = sp.integrate.simps(bin4,z_mid)
    nz_b4 = bin4/int_b4
    return z_mid, nz_b4


def get_nz_source():
    z_mid = twopt_file['nz_shear_true'].data['Z_MID']
    bin4 = twopt_file['nz_shear_true'].data['BIN' + str(bin_source)]
    int_b4 = sp.integrate.simps(bin4,z_mid)
    nz_b4 = bin4/int_b4
    return z_mid, nz_b4


