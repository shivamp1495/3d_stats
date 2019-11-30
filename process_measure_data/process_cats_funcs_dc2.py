import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import treecorr
import healpy as hp
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import kmeans_radec
from numpy.random import rand
import pickle as pk
import matplotlib.cm as cm
import scipy.interpolate as interpolate
from numpy.linalg import inv
import pdb
import time
import GCRCatalogs
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.optimize as op
import scipy as sp
import process_cats_class as pcc

do_m = 0
do_rm = 0
do_g = 1
do_rg = 1

ds_m = 2

cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

zmin_dc2 = 0.15
zmax_dc2 = 0.6


other_params_dict = {}
other_params_dict['zmin_bins'] = [0.15,0.3,0.45]
other_params_dict['zmax_bins'] = [0.3,0.45,0.6]
other_params_dict['bin_n_array'] = [1,2,3]
other_params_dict['bin_array'] = ['bin1','bin2','bin3']

njk_radec = 60
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


# def get_int(chi_max, chi_min):
#     chi_array = np.linspace(chi_min, chi_max, 5000)
#     int_total = sp.integrate.simps(chi_array ** 2, chi_array)
#     return int_total


save_dir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/dc2_v1.0/process_cats/'
# save_filename_matter = 'matter_ra_dec_r_z_bin_jk_cosmoDC2_v1.0_njkradec_' + str(njk_radec) + '_njkz_' + str(
#     njk_z) + '_ds_' + str(ds_m) + '.fits'
# save_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_cosmoDC2_v1.0_njkradec_' + str(
#     njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m) + '.fits'
save_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_cosmoDC2_v1.0_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '.fits'
save_filename_galaxy_randoms = 'randoms_galaxy_ra_dec_r_z_bin_jk_cosmoDC2_v1.0_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '.fits'


print 'loading galaxies'
# file_matter_mice = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/matter_ra_dec_r_z_L3072N4096-LC129-1in700.fits')[1].data
# ra_m, dec_m, z_m = file_matter_mice['RA'],file_matter_mice['DEC'],file_matter_mice['Z']
#
# if ds_m > 1:
#     ind_ds = np.random.randint(0,len(ra_m),int(len(ra_m)/ds_m))
#     ra_m, dec_m, z_m  = ra_m[ind_ds], dec_m[ind_ds], z_m[ind_ds]

gal_dc2 = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_higdc2_0.5-10.fit')[1].data
ra_g_dc2_all, dec_g_dc2_all, z_g_dc2_all = gal_dc2['RA'],gal_dc2['DEC'],gal_dc2['ZSPEC']

ind_dc2 = np.where((z_g_dc2_all > zmin_dc2) & (z_g_dc2_all < zmax_dc2))[0]
ra_g_dc2, dec_g_dc2, z_g_dc2 = ra_g_dc2_all[ind_dc2], dec_g_dc2_all[ind_dc2], z_g_dc2_all[ind_dc2]

print 'getting jk obj map from galaxies'
ind_jk_g = np.where((z_g_dc2 > other_params_dict['zmin_bins'][0]) & (z_g_dc2 < other_params_dict['zmax_bins'][0]))[0]
jkobj_map_radec = pcc.get_jkobj(np.transpose([ra_g_dc2[ind_jk_g], dec_g_dc2[ind_jk_g]]),njk_radec)
other_params_dict['jkobj_map_radec'] = jkobj_map_radec
np.savetxt(save_dir + 'jk_centers_mice2_des_run_redmapper_v6.4.16_redmagic_njkradec_' + str(njk_radec) + '.txt',jkobj_map_radec.centers)

gal_hlum = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_highlum_1.0-04.fit')[1].data
ra_g_hlum_all, dec_g_hlum_all, z_g_hlum_all = gal_hlum['RA'],gal_hlum['DEC'],gal_hlum['ZSPEC']

ind_hlum = np.where((z_g_hlum_all > zmin_hlum) & (z_g_hlum_all < zmax_hlum))[0]
ra_g_hlum, dec_g_hlum, z_g_hlum  = ra_g_hlum_all[ind_hlum], dec_g_hlum_all[ind_hlum], z_g_hlum_all[ind_hlum]

gal_hrlum = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_higherlum_1.5-01.fit')[1].data
ra_g_hrlum_all, dec_g_hrlum_all, z_g_hrlum_all = gal_hrlum['RA'],gal_hrlum['DEC'],gal_hrlum['ZSPEC']

ind_hrlum = np.where((z_g_hrlum_all > zmin_hrlum) & (z_g_hrlum_all < zmax_hrlum))[0]
ra_g_hrlum, dec_g_hrlum, z_g_hrlum  = ra_g_hrlum_all[ind_hrlum], dec_g_hrlum_all[ind_hrlum], z_g_hrlum_all[ind_hrlum]

ra_g = np.hstack((ra_g_dc2,ra_g_hlum,ra_g_hrlum))
dec_g = np.hstack((dec_g_dc2,dec_g_hlum,dec_g_hrlum))
z_g = np.hstack((z_g_dc2,z_g_hlum,z_g_hrlum))

ind_lt_90 = np.where(ra_g < 90)[0]
ra_g, dec_g,z_g = ra_g[ind_lt_90], dec_g[ind_lt_90],z_g[ind_lt_90]

if do_m:
    CF_m = pcc.Catalog_funcs(ra_m, dec_m, z_m ,cosmo_params_dict,other_params_dict)

    print 'getting matter randoms'
    ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform(0.0, zmax_hrlum)

    print 'getting matter jk'
    bin_n_all_m,jk_all_m = CF_m.get_jk_stats()
    CF_m.save_cat(ra_m, dec_m, z_m,bin_n_all_m,jk_all_m,save_dir,save_filename_matter)

    del CF_m

if do_rm:

    CF_rand_m = pcc.Catalog_funcs(ra_rand_m, dec_rand_m, z_rand_m ,cosmo_params_dict,other_params_dict)
    print 'getting matter randoms jk'
    bin_n_all_rand_m,jk_all_rand_m = CF_rand_m.get_jk_stats()
    CF_rand_m.save_cat(ra_rand_m, dec_rand_m, z_rand_m,bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)

    del CF_rand_m

if do_g:
    CF_g = pcc.Catalog_funcs(ra_g, dec_g, z_g ,cosmo_params_dict,other_params_dict)

    print 'getting galaxy randoms'
    ra_rand_g, dec_rand_g, z_rand_g = CF_g.create_random_cat_uniform(0.0, zmax_hrlum)

    print 'getting galaxy jk'
    bin_n_all_g,jk_all_g = CF_g.get_jk_stats()
    CF_g.save_cat(ra_g, dec_g, z_g,bin_n_all_g,jk_all_g,save_dir,save_filename_galaxy)

    del CF_g

if do_rg:

    CF_rand_g = pcc.Catalog_funcs(ra_rand_g, dec_rand_g, z_rand_g ,cosmo_params_dict,other_params_dict)
    print 'getting galaxy randoms jk'
    bin_n_all_rand_g,jk_all_rand_g = CF_rand_g.get_jk_stats()
    CF_rand_g.save_cat(ra_rand_g, dec_rand_g, z_rand_g,bin_n_all_rand_g,jk_all_rand_g,save_dir,save_filename_galaxy_randoms)

    del CF_rand_g










