import sys, os
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import treecorr
import healpy as hp
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample
from numpy.random import rand
import pickle as pk
import matplotlib.cm as cm
import scipy.interpolate as interpolate
from numpy.linalg import inv
import pdb
import time
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.optimize as op
import scipy as sp
from astropy import constants as const
import process_cats_class as pcc
import colossus
from colossus.cosmology import cosmology
from colossus.lss import bias
from colossus.lss import mass_function
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import concentration

do_m = 0
do_rm = 0
do_g = 1
do_rg = 1
do_plot = False

massbin_min = [12.0, 12.5, 13.0, 13.5, 14.0]
massbin_max = [12.5, 13.0, 13.5, 14.0, 14.5]
# nrand_fac_array = [15,20,30,50,80]
nrand_fac_array = [10,10,10,10,10]
ds_m = 1

ds_g_inp = 1
ds_g_process = 1

ds_g = ds_g_inp * ds_g_process

cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

other_params_dict = {}
other_params_dict['zmin_bins'] = [0.15,0.3,0.45,0.6,0.75]
other_params_dict['zmax_bins'] = [0.3,0.45,0.6,0.75,0.9]

# other_params_dict['zmin_bins'] = [0.25,0.325,0.4,0.475,0.55]
# other_params_dict['zmax_bins'] = [0.325,0.4,0.475,0.55,0.625]

other_params_dict['bin_n_array'] = [1,2,3,4,5]
other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5']

njk_radec = 180
njk_z = 1
other_params_dict['njk_radec'] = njk_radec
other_params_dict['njk_z'] = njk_z

gnf = pcc.general_funcs(cosmo_params_dict)
z_array = np.linspace(0, 1.5, 10000)
chi_array = np.zeros(len(z_array))
for j in range(len(z_array)):
    chi_array[j] = gnf.get_Dcom(z_array[j])
other_params_dict['chi_interp'] = interpolate.interp1d(z_array, chi_array)

chi_array = np.linspace(0, 4000, 50000)
z_array = np.zeros(len(chi_array))
for j in range(len(z_array)):
    z_array[j] = gnf.get_z_from_chi(chi_array[j])
other_params_dict['z_interp'] = interpolate.interp1d(chi_array, z_array)


print('loading galaxies and matter catalogs')
file_matter_mice = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/matter_ra_dec_r_z_L3072N4096-LC129-1in700.fits')[1].data
ra_m, dec_m, z_m = file_matter_mice['RA'],file_matter_mice['DEC'],file_matter_mice['Z']

halo_inp = fits.open('/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/MICEv2_halos_Mlow_1e12_ds_' + str(ds_g_inp) + '.fits')

if ds_m > 1:
    ind_ds = np.random.randint(0,len(ra_m),int(len(ra_m)/ds_m))
    ra_m, dec_m, z_m  = ra_m[ind_ds], dec_m[ind_ds], z_m[ind_ds]

z_min, z_max = np.min(z_m), np.max(z_m)
nzbins_total = 5000
zarray_all = np.linspace(z_min, z_max, nzbins_total)
zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
zarray = zarray_all[1:-1]
chi_array_r = gnf.get_Dcom_array(zarray)
dchi_dz_array_r = (const.c.to(u.km / u.s)).value / (gnf.get_Hz(zarray))
chi_max = gnf.get_Dcom_array([z_max])[0]
chi_min = gnf.get_Dcom_array([z_min])[0]
VT = (4 * np.pi / 3) * (chi_max ** 3 - chi_min ** 3)
dndz = (4 * np.pi) * (chi_array_r ** 2) * dchi_dz_array_r / VT



dndm_model = 'crocce10'
bias_model = 'bhattacharya11'
mdef = 'fof'
cosmo_params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.0448, 'sigma8': 0.8, 'ns': 0.95}


cosmology.addCosmology('mock_cosmo', cosmo_params)
cosmo_colossus = cosmology.setCosmology('mock_cosmo')
h = cosmo_params['H0'] / 100.


# get the halo mass function and halo bias using the colossus module
def get_dndm_bias(M_mat,z_array, mdef):

    dndm_array_Mz, bm_array_Mz = np.zeros(M_mat.shape), np.zeros(M_mat.shape)

    for j in range(len(z_array)):
        M_array = M_mat[j, :]
        dndm_array_Mz[j, :] = (1. / M_array) * mass_function.massFunction(M_array, z_array[j],mdef=mdef, model=dndm_model,q_out='dndlnM')

        bm_array_Mz[j, :] = bias.haloBias(M_array, z_array[j], model=bias_model, mdef=mdef)

    return dndm_array_Mz, bm_array_Mz


M_array = np.logspace(11,16,2000)
nm = len(M_array)
nz = len(zarray)
M_mat = np.tile(M_array.reshape(1, nm), (nz, 1))


dndm_array, bm_array = get_dndm_bias(M_mat,zarray, mdef)

for jm in range(len(massbin_min)):

    lmhalo_min = massbin_min[jm]
    lmhalo_max = massbin_max[jm]

    print('processing ' + str(lmhalo_min) + ', ' + str(lmhalo_max))

    save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/'
    if jm == 0:
        save_filename_matter = 'matter_ra_dec_r_z_bin_jk_maglim_L3072N4096-LC129-1in700_njkradec_' + str(
            njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m) + '_v2.fits'
        save_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_maglim_L3072N4096-LC129-1in700_njkradec_' + str(
            njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m) + '_th_nz_v2.fits'

    save_filename_galaxy = 'halos_ra_dec_r_z_bin_jk_mice_lmhalo_' + str(lmhalo_min) + '_' + str(
        lmhalo_max) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_g) + '_v2.fits'
    save_filename_galaxy_randoms = 'randoms_halos_ra_dec_r_z_bin_jk_mice_lmhalo_' + str(lmhalo_min) + '_' + str(
        lmhalo_max) + 'njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_g) + '_th_nz_v2.fits'
    save_filename_jk_obj = 'jkobj_mice_lmhalo_' + str(lmhalo_min) + '_' + str(
        lmhalo_max) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + 'v2.pk'


    # ra_all, dec_all, z_all, lmhalo_all = halo_inp[1].data['ra_gal'],halo_inp[1].data['dec_gal'],halo_inp[1].data['z_cgal'], halo_inp[1].data['lmhalo']
    ra_all, dec_all, z_all, lmhalo_all = halo_inp[1].data['ra'], halo_inp[1].data['dec'], halo_inp[1].data['z'], halo_inp[1].data['log_m']
    ind = np.where((lmhalo_all >= lmhalo_min)  & (lmhalo_all <= lmhalo_max) )[0]

    ra_g, dec_g, z_g = ra_all[ind], dec_all[ind], z_all[ind]

    if ds_g_process > 1:
        ind_ds = np.unique(np.random.randint(0,len(ra_g),int(len(ra_g)/ds_g_process)))
        ra_g, dec_g, z_g  = ra_g[ind_ds], dec_g[ind_ds], z_g[ind_ds]

    if jm == 0:
        print('getting jk obj map from galaxies')
        if os.path.isfile(save_dir + save_filename_jk_obj):
            jkobj_map_radec_centers = pk.load(open(save_dir + save_filename_jk_obj,'rb'))['jkobj_map_radec_centers']
            jkobj_map_radec = KMeans(jkobj_map_radec_centers)
        else:
            # ind_jk_g = np.where((z_g > other_params_dict['zmin_bins'][0]) & (z_g < other_params_dict['zmax_bins'][0]))[0]
            ind_jk_g = np.where((z_g > other_params_dict['zmin_bins'][0]) & (z_g < (other_params_dict['zmin_bins'][0] + 0.05) ))[0]
            jkobj_map_radec = pcc.get_jkobj(np.transpose([ra_g[ind_jk_g], dec_g[ind_jk_g]]),njk_radec)
            jk_dict = {'jkobj_map_radec_centers':jkobj_map_radec.centers}
            pk.dump(jk_dict, open(save_dir + save_filename_jk_obj, 'wb'))

        other_params_dict['jkobj_map_radec'] = jkobj_map_radec

    ind_lt_90 = np.where( (ra_g < 90) & (ra_g > 0) )[0]
    ra_g, dec_g,z_g = ra_g[ind_lt_90], dec_g[ind_lt_90],z_g[ind_lt_90]

    if jm == 0:
        if do_m:
            CF_m = pcc.Catalog_funcs(ra_m, dec_m, z_m ,cosmo_params_dict,other_params_dict)

            print('getting matter randoms')
            # nz_bins_total = np.min(np.array([20000,len(ra_m)/10]))
            # ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform(other_params_dict['zmax_bins'][0], other_params_dict['zmax_bins'][-1], nzbins_total = nz_bins_total)

            ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=dndz, nrand_fac=nrand_fac_array[jm])
            print('getting matter jk')
            bin_n_all_m,jk_all_m = CF_m.get_jk_stats()
            CF_m.save_cat(ra_m, dec_m, z_m,bin_n_all_m,jk_all_m,save_dir,save_filename_matter)

            del CF_m

        if do_rm:

            CF_rand_m = pcc.Catalog_funcs(ra_rand_m, dec_rand_m, z_rand_m ,cosmo_params_dict,other_params_dict)
            print('getting matter randoms jk')
            bin_n_all_rand_m,jk_all_rand_m = CF_rand_m.get_jk_stats()
            CF_rand_m.save_cat(ra_rand_m, dec_rand_m, z_rand_m,bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)

            del CF_rand_m

    if do_g:
        CF_g = pcc.Catalog_funcs(ra_g, dec_g, z_g ,cosmo_params_dict,other_params_dict)

        print('getting galaxy randoms')
        # nz_bins_total = np.min(np.array([20000,len(ra_g)/10]))
        # ra_rand_g, dec_rand_g, z_rand_g = CF_g.create_random_cat_uniform(other_params_dict['zmax_bins'][0], other_params_dict['zmax_bins'][-1], nzbins_total = nz_bins_total)

        ind_sel = np.where((M_mat < 10 ** lmhalo_min) | (M_mat > 10 ** lmhalo_max))
        dndm_Mmat_bin = np.copy(dndm_array)
        dndm_Mmat_bin[ind_sel] = 0.0
        nbar_zarray = sp.integrate.simps(dndm_Mmat_bin, M_array)
        dNdz_array = nbar_zarray * (4 * np.pi) * (chi_array_r ** 2) * dchi_dz_array_r
        N_T = sp.integrate.simps(dNdz_array, zarray)
        dndz_g = dNdz_array / N_T

        ra_rand_g, dec_rand_g, z_rand_g = CF_g.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=dndz_g, nrand_fac=nrand_fac_array[jm])
        print('getting galaxy jk')
        bin_n_all_g,jk_all_g = CF_g.get_jk_stats()
        CF_g.save_cat(ra_g, dec_g, z_g,bin_n_all_g,jk_all_g,save_dir,save_filename_galaxy)

        del CF_g

    if do_rg:

        CF_rand_g = pcc.Catalog_funcs(ra_rand_g, dec_rand_g, z_rand_g ,cosmo_params_dict,other_params_dict)
        print('getting galaxy randoms jk')
        bin_n_all_rand_g,jk_all_rand_g = CF_rand_g.get_jk_stats()
        CF_rand_g.save_cat(ra_rand_g, dec_rand_g, z_rand_g,bin_n_all_rand_g,jk_all_rand_g,save_dir,save_filename_galaxy_randoms)

        del CF_rand_g

    if do_plot:
        nz_unnorm_g, _ = np.histogram(z_g, zarray_edges)
        nz_normed_g = nz_unnorm_g / (integrate.simps(nz_unnorm_g, zarray))
        nz_unnorm, _ = np.histogram(z_m, zarray_edges)
        nz_normed = nz_unnorm / (integrate.simps(nz_unnorm, zarray))

        nz_unnorm_g_r, _ = np.histogram(z_rand_g, zarray_edges)
        nz_normed_g_r = nz_unnorm_g_r / (integrate.simps(nz_unnorm_g_r, zarray))
        nz_unnorm_r, _ = np.histogram(z_rand_m, zarray_edges)
        nz_normed_r = nz_unnorm_r / (integrate.simps(nz_unnorm_r, zarray))


        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        # ax.set_xlim(0.1, 1.5)
        # ax.set_ylim(1e-2, 2.5)
        ax.plot(zarray, nz_normed, 'r-', label='Matter Data', linewidth=0.5)
        ax.plot(zarray, nz_normed_r, 'k-', label='Matter Randoms')
        ax.plot(zarray, nz_normed_g, 'orange', label='Halo Data', linewidth=0.3)
        ax.plot(zarray, nz_normed_g_r, 'k--', label='Halo Randoms')

        ax.legend(fontsize=18)
        plt.xlabel(r'$z$', fontsize=22)
        plt.ylabel(r'$n(z)$', fontsize=26)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='minor', labelsize=15)
        plt.tight_layout()
        fig.savefig(save_dir + 'halos_nz_M_' + str(lmhalo_min) + '_' + str(lmhalo_max) + '_hrandv2.png')















