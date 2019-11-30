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
import healpy as hp
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.optimize as op
import scipy as sp
import process_cats_class as pcc

do_m = 1
do_rm = 1
do_g = 0
do_rg = 0

cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.286, 'Ob0': 0.047, 'sigma8': 0.82, 'ns': 0.96}

other_params_dict = {}
other_params_dict['zmin_bins'] = [0.15,0.35,0.5,0.65,0.85]
other_params_dict['zmax_bins'] = [0.35,0.5,0.65,0.85,0.95]
other_params_dict['bin_n_array'] = [1,2,3,4,5]
other_params_dict['bin_array'] = ['bin1','bin2','bin3','bin4','bin5']

njk_radec = 180
njk_z = 1
njk_total = njk_radec * njk_z
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


zmin_hdens = 0.15
zmax_hdens = 0.65
zmin_hlum = 0.65
zmax_hlum = 0.95
zmin_hrlum = 0.95
zmax_hrlum = 1.05

save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/process_cats/'
save_filename_matter = 'matter_ra_dec_r_z_bin_jk_downsampled_particles.fits.downsample_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '.fits'
save_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_downsampled_particles.fits.downsample_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '.fits'
save_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_buzzard_1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '.fits'
save_filename_galaxy_randoms = 'randoms_galaxy_ra_dec_r_z_bin_jk_buzzard_1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '.fits'


print 'loading galaxies'
file_matter_buzzard = fits.open('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/downsampled_particles.fits.downsample')
ra_m, dec_m, z_m = file_matter_buzzard[1].data['azim_ang'],file_matter_buzzard[1].data['polar_ang'],file_matter_buzzard[1].data['redshift']

gal_hdens = fits.open('/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/redstuff/buzzard-1.9.8/des-y3a1-sc/redmapper_v6.4.22/redmagic_3a/buzzard-1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_highdens_0.5-10.fit')[1].data
ra_g_hdens_all, dec_g_hdens_all, z_g_hdens_all = gal_hdens['RA'],gal_hdens['DEC'],gal_hdens['ZSPEC']

ind_hdens = np.where((z_g_hdens_all > zmin_hdens) & (z_g_hdens_all < zmax_hdens))[0]
ra_g_hdens, dec_g_hdens, z_g_hdens = ra_g_hdens_all[ind_hdens], dec_g_hdens_all[ind_hdens], z_g_hdens_all[ind_hdens]

print 'getting jk obj map from galaxies'
ind_jk_g = np.where((z_g_hdens > 0.15) & (z_g_hdens < 0.3))[0]
jkobj_map_radec = pcc.get_jkobj(np.transpose([ra_g_hdens[ind_jk_g], dec_g_hdens[ind_jk_g]]),njk_radec)
other_params_dict['jkobj_map_radec'] = jkobj_map_radec
np.savetxt(save_dir + 'jk_centers_buzzard_1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_njkradec_' + str(njk_radec) + '.txt',jkobj_map_radec.centers)

gal_hlum = fits.open('/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/redstuff/buzzard-1.9.8/des-y3a1-sc/redmapper_v6.4.22/redmagic_3a/buzzard-1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_highlum_1.0-04.fit')[1].data
ra_g_hlum_all, dec_g_hlum_all, z_g_hlum_all = gal_hlum['RA'],gal_hlum['DEC'],gal_hlum['ZSPEC']

ind_hlum = np.where((z_g_hlum_all > zmin_hlum) & (z_g_hlum_all < zmax_hlum))[0]
ra_g_hlum, dec_g_hlum, z_g_hlum  = ra_g_hlum_all[ind_hlum], dec_g_hlum_all[ind_hlum], z_g_hlum_all[ind_hlum]

gal_hrlum = fits.open('/global/cscratch1/sd/jderose/BCC/Chinchilla/Herd/Chinchilla-3/redstuff/buzzard-1.9.8/des-y3a1-sc/redmapper_v6.4.22/redmagic_3a/buzzard-1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_higherlum_1.5-01.fit')[1].data
ra_g_hrlum_all, dec_g_hrlum_all, z_g_hrlum_all = gal_hrlum['RA'],gal_hrlum['DEC'],gal_hrlum['ZSPEC']

ind_hrlum = np.where((z_g_hrlum_all > zmin_hrlum) & (z_g_hrlum_all < zmax_hrlum))[0]
ra_g_hrlum, dec_g_hrlum, z_g_hrlum  = ra_g_hrlum_all[ind_hrlum], dec_g_hrlum_all[ind_hrlum], z_g_hrlum_all[ind_hrlum]

ra_g = np.hstack((ra_g_hdens,ra_g_hlum,ra_g_hrlum))
dec_g = np.hstack((dec_g_hdens,dec_g_hlum,dec_g_hrlum))
z_g = np.hstack((z_g_hdens,z_g_hlum,z_g_hrlum))


print 'loading galaxy randoms'
rand_hdens = fits.open('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/redmagic_3y3/buzzard_1.9.2+1_3y3_run_redmapper_v6.4.22_redmagic_highdens_0.5-10_randoms.fit')[1].data
ra_rand_g_hdens_all, dec_rand_g_hdens_all, z_rand_g_hdens_all = rand_hdens['RA'],rand_hdens['DEC'],rand_hdens['Z']

ind_hdens = np.where((z_rand_g_hdens_all > zmin_hdens) & (z_rand_g_hdens_all < zmax_hdens))[0]
ra_rand_g_hdens, dec_rand_g_hdens, z_rand_g_hdens = ra_rand_g_hdens_all[ind_hdens], dec_rand_g_hdens_all[ind_hdens], z_rand_g_hdens_all[ind_hdens]

rand_hlum = fits.open('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/redmagic_3y3/buzzard_1.9.2+1_3y3_run_redmapper_v6.4.22_redmagic_highlum_1.0-04_randoms.fit')[1].data
ra_rand_g_hlum_all, dec_rand_g_hlum_all, z_rand_g_hlum_all = rand_hlum['RA'],rand_hlum['DEC'],rand_hlum['Z']

ind_hlum = np.where((z_rand_g_hlum_all > zmin_hlum) & (z_rand_g_hlum_all < zmax_hlum))[0]
ra_rand_g_hlum, dec_rand_g_hlum, z_rand_g_hlum  = ra_rand_g_hlum_all[ind_hlum], dec_rand_g_hlum_all[ind_hlum], z_rand_g_hlum_all[ind_hlum]

rand_hrlum = fits.open('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/redmagic_3y3/buzzard_1.9.2+1_3y3_run_redmapper_v6.4.22_redmagic_higherlum_1.5-01_randoms.fit')[1].data
ra_rand_g_hrlum_all, dec_rand_g_hrlum_all, z_rand_g_hrlum_all = rand_hrlum['RA'],rand_hrlum['DEC'],rand_hrlum['Z']

ind_hrlum = np.where((z_rand_g_hrlum_all > zmin_hrlum) & (z_rand_g_hrlum_all < zmax_hrlum))[0]
ra_rand_g_hrlum, dec_rand_g_hrlum, z_rand_g_hrlum  = ra_rand_g_hrlum_all[ind_hrlum], dec_rand_g_hrlum_all[ind_hrlum], z_rand_g_hrlum_all[ind_hrlum]

ra_rand_g = np.hstack((ra_rand_g_hdens,ra_rand_g_hlum,ra_rand_g_hrlum))
dec_rand_g = np.hstack((dec_rand_g_hdens,dec_rand_g_hlum,dec_rand_g_hrlum))
z_rand_g = np.hstack((z_rand_g_hdens,z_rand_g_hlum,z_rand_g_hrlum))

# print 'setting up the classes'
# print 'getting matter randoms'
# theta_g, phi_g = pcc.eq2ang(ra_g, dec_g)
# nside_mask = 128
# ind_g_f = hp.ang2pix(nside_mask, theta_g, phi_g)
# mask_d = np.zeros(hp.nside2npix(nside_mask))
# mask_d[ind_g_f] = 1
# plt.figure()
# hp.mollview(mask_d)
# plt.savefig('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/measurements/' + 'data_m_sky.png')
# pdb.set_trace()
# ra_rand_m, dec_rand_m, z_rand_m = CF_m.create_random_cat_masked(0.0, zmax_hrlum,ind_g_f,nside_mask)

if do_m:
    CF_m = pcc.Catalog_funcs(ra_m, dec_m, z_m ,cosmo_params_dict,other_params_dict)
    print 'getting matter jk'
    bin_n_all_m,jk_all_m = CF_m.get_jk_stats()
    CF_m.save_cat(ra_m, dec_m, z_m,bin_n_all_m,jk_all_m,save_dir,save_filename_matter)

if do_rm:
    ra_rand_m, dec_rand_m, z_rand_m = ra_rand_g, dec_rand_g, z_rand_g
    print ' nrand_g: ',len(ra_rand_g), ', ng:',len(ra_g),' nrand_m: ',len(ra_rand_m), ', nm:',len(ra_m)
    CF_rand_m = pcc.Catalog_funcs(ra_rand_m, dec_rand_m, z_rand_m ,cosmo_params_dict,other_params_dict)
    print 'getting matter randoms jk'
    bin_n_all_rand_m,jk_all_rand_m = CF_rand_m.get_jk_stats()
    CF_rand_m.save_cat(ra_rand_m, dec_rand_m, z_rand_m,bin_n_all_rand_m,jk_all_rand_m,save_dir,save_filename_matter_randoms)

if do_g:
    CF_g = pcc.Catalog_funcs(ra_g, dec_g, z_g ,cosmo_params_dict,other_params_dict)
    print 'getting galaxy jk'
    bin_n_all_g,jk_all_g = CF_g.get_jk_stats()
    CF_g.save_cat(ra_g, dec_g, z_g,bin_n_all_g,jk_all_g,save_dir,save_filename_galaxy)

if do_rg:
    CF_rand_g = pcc.Catalog_funcs(ra_rand_g, dec_rand_g, z_rand_g ,cosmo_params_dict,other_params_dict)
    print 'getting galaxy randoms jk'
    bin_n_all_rand_g,jk_all_rand_g = CF_rand_g.get_jk_stats()
    CF_rand_g.save_cat(ra_rand_g, dec_rand_g, z_rand_g,bin_n_all_rand_g,jk_all_rand_g,save_dir,save_filename_galaxy_randoms)










