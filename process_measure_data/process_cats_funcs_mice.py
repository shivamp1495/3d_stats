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
import scipy.signal as spsg
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


do_m = 1
do_rm = 1
do_g = 1
do_rg = 1

# ds_m = 1.1
ds_m = 1

cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}

zmin_hdens = 0.15
zmax_hdens = 0.6
zmin_hlum = 0.6
zmax_hlum = 0.75
zmin_hrlum = 0.75
zmax_hrlum = 0.9

other_params_dict = {}
other_params_dict['zmin_bins'] = [0.15,0.3,0.45,0.6,0.75]
other_params_dict['zmax_bins'] = [0.3,0.45,0.6,0.75,0.9]
other_params_dict['bin_n_array'] = [1,2,3,4,5]
other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5']

# njk_radec = 180
# njk_z = 1
# njk_radec = 300
# njk_z = 1
njk_radec = 100
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


save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/'

# save_filename_matter = 'matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(njk_radec) + '_njkz_' + str(
#     njk_z) + '_ds_' + str(ds_m) + '.fits'
# save_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
#     njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m) + '.fits'



save_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_mice_des_run_redmapper_v6.4.16_redmagic_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '_v2.fits'
save_filename_galaxy_randoms = 'randoms_galaxy_ra_dec_r_z_bin_jk_mice_des_run_redmapper_v6.4.16_redmagic_redmagic_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '_v2.fits'
# save_filename_jk_obj = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/jkobj_mice_lmhalo_12.0_12.5_njkradec_180_njkz_1v2.pk'
save_filename_jk_obj = 'jkobj_mice' + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_v2.pk'

print('loading galaxies and matter cat')
file_matter_mice = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/matter_ra_dec_r_z_L3072N4096-LC129-1in700.fits')[1].data
ra_m, dec_m, z_m = file_matter_mice['RA'],file_matter_mice['DEC'],file_matter_mice['Z']

if ds_m > 1:
    ind_ds = np.unique(np.random.randint(0,len(ra_m),int(len(ra_m)/ds_m)))
    ds_m_save = np.around(len(z_m)/(1.0*len(ind_ds)),2)
    print('matter downsampled/original = ' + str(len(z_m)/(1.0*len(ind_ds))))
    ra_m, dec_m, z_m  = ra_m[ind_ds], dec_m[ind_ds], z_m[ind_ds]
else:
    ds_m_save = ds_m

save_filename_matter = 'matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(njk_radec) + '_njkz_' + str(
    njk_z) + '_ds_' + str(ds_m_save) + '_v2.fits'
save_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_save) + '_v2.fits'


z_min, z_max = np.min(z_m), np.max(z_m)
nzbins_total = 1000
zarray_all = np.linspace(z_min, z_max, nzbins_total)
zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
zarray = zarray_all[1:-1]
chi_array_r = gnf.get_Dcom_array(zarray)
dchi_dz_array_r = (const.c.to(u.km / u.s)).value / (gnf.get_Hz(zarray))
chi_max = gnf.get_Dcom_array([z_max])[0]
chi_min = gnf.get_Dcom_array([z_min])[0]
VT = (4 * np.pi / 3) * (chi_max ** 3 - chi_min ** 3)
dndz = (4 * np.pi) * (chi_array_r ** 2) * dchi_dz_array_r / VT

# dndm_model = 'crocce10'
# bias_model = 'bhattacharya11'
# mdef = 'fof'
# cosmo_params = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.0448, 'sigma8': 0.8, 'ns': 0.95}

# cosmology.addCosmology('mock_cosmo', cosmo_params)
# cosmo_colossus = cosmology.setCosmology('mock_cosmo')
# h = cosmo_params['H0'] / 100.

print('getting jk obj map from galaxies')
# ind_jk_g = np.where((z_g_hdens > other_params_dict['zmin_bins'][0]) & (z_g_hdens < other_params_dict['zmax_bins'][0]))[0]
# jkobj_map_radec = pcc.get_jkobj(np.transpose([ra_g_hdens[ind_jk_g], dec_g_hdens[ind_jk_g]]),njk_radec)
# other_params_dict['jkobj_map_radec'] = jkobj_map_radec
# np.savetxt(save_dir + 'jk_centers_mice2_des_run_redmapper_v6.4.16_redmagic_njkradec_' + str(njk_radec) + '.txt',jkobj_map_radec.centers)


gal_hdens = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_highdens_0.5-10.fit')[1].data
ra_g_hdens_all, dec_g_hdens_all, z_g_hdens_all = gal_hdens['RA'],gal_hdens['DEC'],gal_hdens['ZSPEC']
ind_hdens = np.where((z_g_hdens_all > zmin_hdens) & (z_g_hdens_all < zmax_hdens))[0]
ra_g_hdens, dec_g_hdens, z_g_hdens = ra_g_hdens_all[ind_hdens], dec_g_hdens_all[ind_hdens], z_g_hdens_all[ind_hdens]

if os.path.isfile(save_dir + save_filename_jk_obj):
    jkobj_map_radec_centers = pk.load(open(save_dir + save_filename_jk_obj,'rb'))['jkobj_map_radec_centers']
    jkobj_map_radec = KMeans(jkobj_map_radec_centers)
else:
    ind_jk_g = np.where((z_g_hdens > other_params_dict['zmin_bins'][0]) & (z_g_hdens < (other_params_dict['zmin_bins'][0] + 0.1) ))[0]
    jkobj_map_radec = pcc.get_jkobj(np.transpose([ra_g_hdens[ind_jk_g], dec_g_hdens[ind_jk_g]]),njk_radec)
    jk_dict = {'jkobj_map_radec_centers':jkobj_map_radec.centers}
    pk.dump(jk_dict, open(save_dir + save_filename_jk_obj, 'wb'),protocol=2)
other_params_dict['jkobj_map_radec'] = jkobj_map_radec


CF_hdens_all = pcc.Catalog_funcs(ra_g_hdens_all, dec_g_hdens_all, z_g_hdens_all ,cosmo_params_dict,other_params_dict)
nz_unnorm, z_edge = np.histogram(z_g_hdens_all, zarray_edges)
nz_unnorm_smooth =  spsg.savgol_filter(nz_unnorm, 21, 5)
nz_normed = nz_unnorm/(integrate.simps(nz_unnorm,zarray))
nz_normed_smooth = nz_unnorm_smooth/(integrate.simps(nz_unnorm_smooth,zarray))
ra_rand_g_hdens_all, dec_rand_g_hdens_all, z_rand_g_hdens_all = CF_hdens_all.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=nz_normed_smooth, nrand_fac=10, ra_min=0, ra_max=90, dec_min=0, dec_max=90)
ind_hdens = np.where((z_rand_g_hdens_all > zmin_hdens) & (z_rand_g_hdens_all < zmax_hdens))[0]
ra_rand_g_hdens, dec_rand_g_hdens, z_rand_g_hdens = ra_rand_g_hdens_all[ind_hdens], dec_rand_g_hdens_all[ind_hdens], z_rand_g_hdens_all[ind_hdens]

do_plot = True
if do_plot:
    fig, ax = plt.subplots(1,1, figsize = (10,8))
    ax.set_xlim(0.1,0.9)
    ax.plot(zarray, nz_normed_smooth, 'k-', label='Smoothed',linewidth=1.5)
    ax.plot(zarray, nz_normed, color='red', label='Original',linewidth=1.8)
    ax.legend(fontsize=18, loc='upper left')
    plt.xlabel(r'z', fontsize=22)
    plt.ylabel(r'n(z)', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()
    plt.savefig('nz_mice_v2_redmagic_hdens_comp.png',dpi=360)


gal_hlum = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_highlum_1.0-04.fit')[1].data
ra_g_hlum_all, dec_g_hlum_all, z_g_hlum_all = gal_hlum['RA'],gal_hlum['DEC'],gal_hlum['ZSPEC']
ind_hlum = np.where((z_g_hlum_all > zmin_hlum) & (z_g_hlum_all < zmax_hlum))[0]
ra_g_hlum, dec_g_hlum, z_g_hlum  = ra_g_hlum_all[ind_hlum], dec_g_hlum_all[ind_hlum], z_g_hlum_all[ind_hlum]

CF_hlum_all = pcc.Catalog_funcs(ra_g_hlum_all, dec_g_hlum_all, z_g_hlum_all ,cosmo_params_dict,other_params_dict)
nz_unnorm, z_edge = np.histogram(z_g_hlum_all, zarray_edges)
nz_unnorm_smooth =  spsg.savgol_filter(nz_unnorm, 21, 5)
nz_normed = nz_unnorm/(integrate.simps(nz_unnorm,zarray))
nz_normed_smooth = nz_unnorm_smooth/(integrate.simps(nz_unnorm_smooth,zarray))
ra_rand_g_hlum_all, dec_rand_g_hlum_all, z_rand_g_hlum_all = CF_hlum_all.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=nz_normed_smooth, nrand_fac=10, ra_min=0, ra_max=90, dec_min=0, dec_max=90)
ind_hlum = np.where((z_rand_g_hlum_all > zmin_hlum) & (z_rand_g_hlum_all < zmax_hlum))[0]
ra_rand_g_hlum, dec_rand_g_hlum, z_rand_g_hlum = ra_rand_g_hlum_all[ind_hlum], dec_rand_g_hlum_all[ind_hlum], z_rand_g_hlum_all[ind_hlum]

do_plot = True
if do_plot:
    fig, ax = plt.subplots(1,1, figsize = (10,8))
    ax.set_xlim(0.1,0.9)
    ax.plot(zarray, nz_normed_smooth, 'k-', label='Smoothed',linewidth=1.5)
    ax.plot(zarray, nz_normed, color='red', label='Original',linewidth=1.8)
    ax.legend(fontsize=18, loc='upper left')
    plt.xlabel(r'z', fontsize=22)
    plt.ylabel(r'n(z)', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()
    plt.savefig('nz_mice_v2_redmagic_hlum_comp.png',dpi=360)

gal_hrlum = fits.open('/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/mice2_des_run_redmapper_v6.4.16_redmagic_higherlum_1.5-01.fit')[1].data
ra_g_hrlum_all, dec_g_hrlum_all, z_g_hrlum_all = gal_hrlum['RA'],gal_hrlum['DEC'],gal_hrlum['ZSPEC']
ind_hrlum = np.where((z_g_hrlum_all > zmin_hrlum) & (z_g_hrlum_all < zmax_hrlum))[0]
ra_g_hrlum, dec_g_hrlum, z_g_hrlum  = ra_g_hrlum_all[ind_hrlum], dec_g_hrlum_all[ind_hrlum], z_g_hrlum_all[ind_hrlum]

CF_hrlum_all = pcc.Catalog_funcs(ra_g_hrlum_all, dec_g_hrlum_all, z_g_hrlum_all ,cosmo_params_dict,other_params_dict)
nz_unnorm, z_edge = np.histogram(z_g_hrlum_all, zarray_edges)
nz_unnorm_smooth =  spsg.savgol_filter(nz_unnorm, 21, 5)
nz_normed = nz_unnorm/(integrate.simps(nz_unnorm,zarray))
nz_normed_smooth = nz_unnorm_smooth/(integrate.simps(nz_unnorm_smooth,zarray))
ra_rand_g_hrlum_all, dec_rand_g_hrlum_all, z_rand_g_hrlum_all = CF_hrlum_all.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=nz_normed_smooth, nrand_fac=10, ra_min=0, ra_max=90, dec_min=0, dec_max=90)
ind_hrlum = np.where((z_rand_g_hrlum_all > zmin_hrlum) & (z_rand_g_hrlum_all < zmax_hrlum))[0]
ra_rand_g_hrlum, dec_rand_g_hrlum, z_rand_g_hrlum = ra_rand_g_hrlum_all[ind_hrlum], dec_rand_g_hrlum_all[ind_hrlum], z_rand_g_hrlum_all[ind_hrlum]

do_plot = True
if do_plot:
    fig, ax = plt.subplots(1,1, figsize = (10,8))
    ax.set_xlim(0.1,0.9)
    ax.plot(zarray, nz_normed_smooth, 'k-', label='Smoothed',linewidth=1.5)
    ax.plot(zarray, nz_normed, color='red', label='Original',linewidth=1.8)
    ax.legend(fontsize=18, loc='upper left')
    plt.xlabel(r'z', fontsize=22)
    plt.ylabel(r'n(z)', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()
    plt.savefig('nz_mice_v2_redmagic_hrlum_comp.png',dpi=360)

ra_g = np.hstack((ra_g_hdens,ra_g_hlum,ra_g_hrlum))
dec_g = np.hstack((dec_g_hdens,dec_g_hlum,dec_g_hrlum))
z_g = np.hstack((z_g_hdens,z_g_hlum,z_g_hrlum))

ra_rand_g = np.hstack((ra_rand_g_hdens,ra_rand_g_hlum,ra_rand_g_hrlum))
dec_rand_g = np.hstack((dec_rand_g_hdens,dec_rand_g_hlum,dec_rand_g_hrlum))
z_rand_g = np.hstack((z_rand_g_hdens,z_rand_g_hlum,z_rand_g_hrlum))


ind_lt_90 = np.where(ra_g < 90)[0]
ra_g, dec_g,z_g = ra_g[ind_lt_90], dec_g[ind_lt_90],z_g[ind_lt_90]

ind_lt_90 = np.where(ra_rand_g < 90)[0]
ra_rand_g, dec_rand_g,z_rand_g = ra_rand_g[ind_lt_90], dec_rand_g[ind_lt_90],z_rand_g[ind_lt_90]

print(len(ra_g), len(ra_rand_g))


# if do_m:
#     CF_m = pcc.Catalog_funcs(ra_m, dec_m, z_m ,cosmo_params_dict,other_params_dict)

#     print('getting matter randoms')
#     ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform(0.0, zmax_hrlum)

#     print('getting matter jk')
#     bin_n_all_m,jk_all_m = CF_m.get_jk_stats()
#     CF_m.save_cat(ra_m, dec_m, z_m,bin_n_all_m,jk_all_m,save_dir,save_filename_matter)

#     del CF_m

# if do_rm:

#     CF_rand_m = pcc.Catalog_funcs(ra_rand_m, dec_rand_m, z_rand_m ,cosmo_params_dict,other_params_dict)
#     print('getting matter randoms jk')
#     bin_n_all_rand_m,jk_all_rand_m = CF_rand_m.get_jk_stats()
#     CF_rand_m.save_cat(ra_rand_m, dec_rand_m, z_rand_m,bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)

#     del CF_rand_m


if do_m:
    CF_m = pcc.Catalog_funcs(ra_m, dec_m, z_m ,cosmo_params_dict,other_params_dict)

    print('getting matter randoms')
    # nz_bins_total = np.min(np.array([20000,len(ra_m)/10]))
    # ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform(other_params_dict['zmax_bins'][0], other_params_dict['zmax_bins'][-1], nzbins_total = nz_bins_total)

    ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_uniform_esutil(zarray=zarray, nz_normed=dndz, nrand_fac=10)
    print('getting matter jk')
    del ra_m, dec_m, z_m
    bin_n_all_m,jk_all_m = CF_m.get_jk_stats()

    # CF_m.save_cat(ra_m, dec_m, z_m,bin_n_all_m,jk_all_m,save_dir,save_filename_matter)
    CF_m.save_cat(bin_n_all_m,jk_all_m,save_dir,save_filename_matter)

    del CF_m


if do_rm:

    CF_rand_m = pcc.Catalog_funcs(ra_rand_m, dec_rand_m, z_rand_m ,cosmo_params_dict,other_params_dict)
    del ra_rand_m, dec_rand_m, z_rand_m
    print('getting matter randoms jk')
    bin_n_all_rand_m,jk_all_rand_m = CF_rand_m.get_jk_stats()
    # CF_rand_m.save_cat(ra_rand_m, dec_rand_m, z_rand_m,bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)
    CF_rand_m.save_cat(bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)

    del CF_rand_m

if do_g:
    CF_g = pcc.Catalog_funcs(ra_g, dec_g, z_g ,cosmo_params_dict,other_params_dict)

    print('getting galaxy jk')
    bin_n_all_g,jk_all_g = CF_g.get_jk_stats()
    # CF_g.save_cat(ra_g, dec_g, z_g,bin_n_all_g,jk_all_g,save_dir,save_filename_galaxy)
    CF_g.save_cat(bin_n_all_g,jk_all_g,save_dir,save_filename_galaxy)

    del CF_g

if do_rg:

    CF_rand_g = pcc.Catalog_funcs(ra_rand_g, dec_rand_g, z_rand_g ,cosmo_params_dict,other_params_dict)
    print('getting galaxy randoms jk')
    bin_n_all_rand_g,jk_all_rand_g = CF_rand_g.get_jk_stats()
    # CF_rand_g.save_cat(ra_rand_g, dec_rand_g, z_rand_g,bin_n_all_rand_g,jk_all_rand_g,save_dir,save_filename_galaxy_randoms)
    CF_rand_g.save_cat(bin_n_all_rand_g,jk_all_rand_g,save_dir,save_filename_galaxy_randoms)

    del CF_rand_g



do_plot = False
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
    ax.plot(zarray, nz_normed_g, 'orange', label='Redmagic Data', linewidth=0.3)
    ax.plot(zarray, nz_normed_g_r, 'k--', label='Redmagic Randoms')

    ax.legend(fontsize=18)
    plt.xlabel(r'$z$', fontsize=22)
    plt.ylabel(r'$n(z)$', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tight_layout()
    fig.savefig(save_dir + 'redmagic_matter_nz_v2.png')







