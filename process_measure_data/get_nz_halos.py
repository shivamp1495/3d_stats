import sys, os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from numpy.random import rand
import pickle as pk
import matplotlib.cm as cm
import scipy.interpolate as interpolate
import pdb
import time
import multiprocessing as mp
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--make_plots', default=1, type=int, help='Do make the plots')
    parser.add_argument('--nz', default=250, type=int, help='Number of radial bins to do the correlation')
    parser.add_argument('--zmin', default=0.1, type=float, help='Minimum of the radial bin')
    parser.add_argument('--zmax', default=1.1, type=float, help='Maximum of the radial bin')
    parser.add_argument('--ds_m_inp', default=1, type=int,
                        help='Ratio by which matter catalog was initially downsampled')
    parser.add_argument('--ds_g_inp', default=1, type=int,
                        help='Ratio by which galaxy catalog was initially downsampled')
    parser.add_argument('--njk_radec', default=180, type=int, help='Number of jack-knife patches in ra-dec dimensions')
    parser.add_argument('--njk_z', default=1, type=int,
                        help='Number of jack-knife patches in redshift dimension. Put as 1 to only get patches in ra and dec')
    parser.add_argument('--bins', default=[1, 2, 3, 4, 5], type=list, help='Bins in the catalogs')
    parser.add_argument('--massbins_min', default=[12.0, 12.5, 13.0, 13.5, 14.0], type=list, help='Mass Bins in the catalogs')
    parser.add_argument('--massbins_max', default=[12.5, 13.0, 13.5, 14.0, 14.5], type=list, help='Mass Bins in the catalogs')

    args_all = parser.parse_args()
    return args_all


if __name__ == "__main__":

    args = parse_arguments()

    make_plots = args.make_plots
    bins_all = args.bins

    # radial bins
    nz = args.nz
    zmin = args.zmin
    zmax = args.zmax
    delta_z = (zmax - zmin) / nz
    zarray_all = np.linspace(zmin, zmax, nz)
    zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
    zarray = zarray_all[1:-1]

    # load the catalaogs of redamagic, matter and randoms

    ds_m_inp = args.ds_m_inp
    ds_g_inp = args.ds_g_inp
    njk_radec = args.njk_radec
    njk_z = args.njk_z
    njk = njk_radec * njk_z

    load_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/'

    massbin_min = args.massbins_min
    massbin_max = args.massbins_max
    
    load_filename_matter = 'matter_ra_dec_r_z_bin_jk_maglim_L3072N4096-LC129-1in700_njkradec_' + str(
        njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_inp) + '_v2.fits'

    print 'loading m'
    load_cat_m = fits.open(load_dir + load_filename_matter)

    ra_m_all, dec_m_all, r_m_all, z_m_all, bin_m_all, jk_m_all = load_cat_m[1].data['RA'], load_cat_m[1].data['DEC'], load_cat_m[1].data['R'], \
                                     load_cat_m[1].data['Z'], load_cat_m[1].data['BIN'], load_cat_m[1].data['JK']

    for j in range(len(massbin_min)):
        print(j)
        mass_min = massbin_min[j]
        mass_max = massbin_max[j]
        save_dir = '/global/project/projectdirs/des/shivamp/cosmosis/y3kp-bias-model/3d_stats/data_dir/nz_halos/halos_' + str(mass_min) + '_' + str(mass_max) + '/'
        

        load_filename_galaxy = 'halos_ra_dec_r_z_bin_jk_mice_lmhalo_' + str(mass_min) + '_' + str(mass_max) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_g_inp) + '_v2.fits'

        print 'loading g'
        load_cat_g = fits.open(load_dir + load_filename_galaxy)

        ra_g_all, dec_g_all, r_g_all, z_g_all, bin_g_all, jk_g_all = load_cat_g[1].data['RA'], load_cat_g[1].data['DEC'], \
                                                                     load_cat_g[1].data['R'], \
                                                                     load_cat_g[1].data['Z'], load_cat_g[1].data['BIN'], \
                                                                     load_cat_g[1].data['JK']

        for binval in bins_all:
            ind_bin_g = np.where(bin_g_all == binval)[0]
            ind_bin_m = np.where(bin_m_all == binval)[0]

            ra_g, dec_g, r_g, z_g, bin_g, jk_g = ra_g_all[ind_bin_g], dec_g_all[ind_bin_g], r_g_all[ind_bin_g], z_g_all[
                ind_bin_g], bin_g_all[ind_bin_g], jk_g_all[ind_bin_g]

            ra_m, dec_m, r_m, z_m, bin_m, jk_m = ra_m_all[ind_bin_m], dec_m_all[ind_bin_m], r_m_all[ind_bin_m], z_m_all[ind_bin_m], bin_m_all[ind_bin_m], jk_m_all[ind_bin_m]

            hist_g, bin_edges = np.histogram(z_g, bins=zarray_edges)
            hist_g_norm = hist_g / (np.sum(hist_g) * delta_z)
            
            hist_m, bin_edges = np.histogram(z_m, bins=zarray_edges)
            hist_m_norm = hist_m / (np.sum(hist_m) * delta_z)

            output_dict = {'nz_g':hist_g_norm,'nz_m':hist_m_norm, 'nz_z':zarray}
            save_filename = 'nz_g_m__zbin_' + str(binval) + '_dsg_1_dsm_1.pk'
            pk.dump(output_dict, open(save_dir + save_filename, 'wb'))


            
