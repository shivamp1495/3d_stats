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
import correlate_gg_gm_3d_class as corr_class
import argparse

nthreads = mp.cpu_count()
print 'nthreads are ', nthreads


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', required=True, type=int, help='bin to run the correlation for')
    parser.add_argument('--do_gg', default=1, type=int, help='Do run gg correlation')
    parser.add_argument('--do_gm', default=1, type=int, help='Do run gm correlation')
    parser.add_argument('--do_mm', default=1, type=int, help='Do run mm correlation')
    parser.add_argument('--get_ratio', default=1, type=int, help='Do get ratio of gg and mm')
    parser.add_argument('--make_plots', default=1, type=int, help='Do make the plots')
    parser.add_argument('--nrad', default=20, type=int, help='Number of radial bins to do the correlation')
    parser.add_argument('--minrad', default=0.8, type=float, help='Minimum of the radial bin')
    parser.add_argument('--maxrad', default=50.0, type=float, help='Maximum of the radial bin')
    parser.add_argument('--do_jk', default=True, type=bool, help='Do jack-knife covariance estimation')
    parser.add_argument('--njk_radec', default=180, type=int, help='Number of jack-knife patches in ra-dec dimensions')
    parser.add_argument('--njk_z', default=1, type=int,
                        help='Number of jack-knife patches in redshift dimension. Put as 1 to only get patches in ra and dec')
    parser.add_argument('--ds_g', default=1, type=int, help='Ratio by which to downsample the galaxy catalog')
    parser.add_argument('--ds_m', default=1, type=int, help='Ratio by which to downsample the matter catalog')
    parser.add_argument('--ds_m_inp', default=1, type=int,
                        help='Ratio by which matter catalog was initially downsampled')
    parser.add_argument('--ds_g_inp', default=1, type=int,
                        help='Ratio by which galaxy catalog was initially downsampled')
    args_all = parser.parse_args()
    return args_all

if __name__ == "__main__":

    args = parse_arguments()
    # which tomographic bin to evaluate the correlation function of
    binval = args.bin

    get_gg = args.do_gg
    get_gm = args.do_gm
    get_mm = args.do_mm
    get_ratio = args.get_ratio
    make_plots = args.make_plots

    # radial bins
    nrad = args.nrad
    minrad = args.minrad
    maxrad = args.maxrad

    do_jk = args.do_jk
    njk_radec = args.njk_radec
    njk_z = args.njk_z
    njk = njk_radec * njk_z

    # downsample the galaxy and matter catalogs to make the calculations fast
    ds_g = args.ds_g
    ds_m = args.ds_m

    # load the catalaogs of redamagic, matter and randoms

    ds_m_inp = args.ds_m_inp
    ds_g_inp = args.ds_g_inp

    # load the catalaogs of redamagic, matter and randoms

    load_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/process_cats/'
    load_filename_matter = 'matter_ra_dec_r_z_bin_jk_downsampled_particles.fits.downsample_njkradec_' + str(
        njk_radec) + '_njkz_' + str(njk_z) + '.fits'
    load_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_downsampled_particles.fits.downsample_njkradec_' + str(
        njk_radec) + '_njkz_' + str(njk_z) + '.fits'
    load_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_buzzard_1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_njkradec_' + str(
        njk_radec) + '_njkz_' + str(njk_z) + '.fits'
    load_filename_galaxy_randoms = 'randoms_galaxy_ra_dec_r_z_bin_jk_buzzard_1.9.8_3y3a_run_redmapper_v6.4.22_redmagic_njkradec_' + str(
        njk_radec) + '_njkz_' + str(njk_z) + '.fits'

    save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/measurements/v1.9.8/'
    save_filename_gg = 'gg_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pk'
    save_filename_mm = 'mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pk'
    save_filename_gm = 'gm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pk'
    save_filename_gg_mm = 'gg_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pk'
    save_filename_gm_mm = 'gm_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pk'

    plot_save_name_gg_gm_mm = 'gg_gm_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
        binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
        ds_g) + '_dsm_' + str(ds_m) + '.pdf'
    plot_save_name_gg_gm_mm_ratio = 'gg_gm_mm_ratio_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(
        nrad) + '_zbin_' + str(binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(
        njk_z) + '_dsg_' + str(ds_g) + '_dsm_' + str(ds_m) + '.pdf'

    print 'loading g'
    load_cat_g = fits.open(load_dir + load_filename_galaxy)

    print 'loading m'
    load_cat_m = fits.open(load_dir + load_filename_matter)

    print 'loading rg'
    load_cat_rand_g = fits.open(load_dir + load_filename_galaxy_randoms)

    print 'loading rm'
    load_cat_rand_m = fits.open(load_dir + load_filename_matter_randoms)

    ra_g, dec_g, r_g, z_g, bin_g, jk_g = load_cat_g[1].data['RA'], load_cat_g[1].data['DEC'], load_cat_g[1].data['R'], \
                                         load_cat_g[1].data['Z'], load_cat_g[1].data['BIN'], load_cat_g[1].data['JK']

    ra_m, dec_m, r_m, z_m, bin_m, jk_m = load_cat_m[1].data['RA'], load_cat_m[1].data['DEC'], load_cat_m[1].data['R'], \
                                         load_cat_m[1].data['Z'], load_cat_m[1].data['BIN'], load_cat_m[1].data['JK']

    ra_rand_g, dec_rand_g, r_rand_g, z_rand_g, bin_rand_g, jk_rand_g = load_cat_rand_g[1].data['RA'], \
                                                                       load_cat_rand_g[1].data['DEC'], \
                                                                       load_cat_rand_g[1].data['R'], \
                                                                       load_cat_rand_g[1].data['Z'], \
                                                                       load_cat_rand_g[1].data['BIN'], \
                                                                       load_cat_rand_g[1].data['JK']

    ra_rand_m, dec_rand_m, r_rand_m, z_rand_m, bin_rand_m, jk_rand_m = load_cat_rand_m[1].data['RA'], \
                                                                       load_cat_rand_m[1].data['DEC'], \
                                                                       load_cat_rand_m[1].data['R'], \
                                                                       load_cat_rand_m[1].data['Z'], \
                                                                       load_cat_rand_m[1].data['BIN'], \
                                                                       load_cat_rand_m[1].data['JK']

    # use only the data corresponding to the given tomographic bin

    ind_bin_g = np.where(bin_g == binval)[0]
    ind_bin_m = np.where(bin_m == binval)[0]
    ind_bin_rand_g = np.where(bin_rand_g == binval)[0]
    ind_bin_rand_m = np.where(bin_rand_m == binval)[0]

    ra_g, dec_g, r_g, z_g, bin_g, jk_g = ra_g[ind_bin_g], dec_g[ind_bin_g], r_g[ind_bin_g], z_g[ind_bin_g], bin_g[
        ind_bin_g], \
                                         jk_g[ind_bin_g]

    ra_m, dec_m, r_m, z_m, bin_m, jk_m = ra_m[ind_bin_m], dec_m[ind_bin_m], r_m[ind_bin_m], z_m[ind_bin_m], bin_m[
        ind_bin_m], \
                                         jk_m[ind_bin_m]

    ra_rand_g, dec_rand_g, r_rand_g, z_rand_g, bin_rand_g, jk_rand_g = ra_rand_g[ind_bin_rand_g], dec_rand_g[
        ind_bin_rand_g], \
                                                                       r_rand_g[ind_bin_rand_g], z_rand_g[
                                                                           ind_bin_rand_g], \
                                                                       bin_rand_g[ind_bin_rand_g], \
                                                                       jk_rand_g[ind_bin_rand_g]

    ra_rand_m, dec_rand_m, r_rand_m, z_rand_m, bin_rand_m, jk_rand_m = ra_rand_m[ind_bin_rand_m], dec_rand_m[
        ind_bin_rand_m], \
                                                                       r_rand_m[ind_bin_rand_m], z_rand_m[
                                                                           ind_bin_rand_m], \
                                                                       bin_rand_m[ind_bin_rand_m], \
                                                                       jk_rand_m[ind_bin_rand_m]

    # pdb.set_trace()
    # if downsampled then truncate the data further
    if ds_g > 1:
        ind_ds_g = np.unique(np.random.randint(0, len(ra_g), len(ra_g) / ds_g))
        ind_ds_rand_g = np.unique(np.random.randint(0, len(ra_rand_g), len(ra_rand_g) / ds_g))

        ra_g, dec_g, r_g, z_g, bin_g, jk_g = ra_g[ind_ds_g], dec_g[ind_ds_g], r_g[ind_ds_g], z_g[ind_ds_g], bin_g[
            ind_ds_g], \
                                             jk_g[ind_ds_g]

        ra_rand_g, dec_rand_g, r_rand_g, z_rand_g, bin_rand_g, jk_rand_g = ra_rand_g[ind_ds_rand_g], dec_rand_g[
            ind_ds_rand_g], \
                                                                           r_rand_g[ind_ds_rand_g], z_rand_g[
                                                                               ind_ds_rand_g], \
                                                                           bin_rand_g[ind_ds_rand_g], \
                                                                           jk_rand_g[ind_ds_rand_g]

    if ds_m > 1:
        ind_ds_m = np.unique(np.random.randint(0, len(ra_m), len(ra_m) / ds_m))
        ind_ds_rand_m = np.unique(np.random.randint(0, len(ra_rand_m), len(ra_rand_m) / ds_m))

        ra_m, dec_m, r_m, z_m, bin_m, jk_m = ra_m[ind_ds_m], dec_m[ind_ds_m], r_m[ind_ds_m], z_m[ind_ds_m], bin_m[
            ind_ds_m], \
                                             jk_m[ind_ds_m]

        ra_rand_m, dec_rand_m, r_rand_m, z_rand_m, bin_rand_m, jk_rand_m = ra_rand_m[ind_ds_rand_m], dec_rand_m[
            ind_ds_rand_m], \
                                                                           r_rand_m[ind_ds_rand_m], z_rand_m[
                                                                               ind_ds_rand_m], \
                                                                           bin_rand_m[ind_ds_rand_m], \
                                                                           jk_rand_m[ind_ds_rand_m]

    print 'number of galaxies : ', len(ra_g)
    print 'number of matter particles : ', len(ra_m)
    print 'number of galaxies randoms: ', len(ra_rand_g)
    print 'number of matter randoms: ', len(ra_rand_m)

    if make_plots:
        def eq2ang(ra, dec):
            phi = ra * np.pi / 180.
            theta = (np.pi / 2.) - dec * (np.pi / 180.)
            return theta, phi


        theta_m, phi_m = eq2ang(ra_rand_m, dec_rand_m)
        ind_m_f = hp.ang2pix(128, theta_m, phi_m)
        mask_m = np.zeros(hp.nside2npix(128))
        mask_m[ind_m_f] = 1
        plt.figure()
        hp.mollview(mask_m)
        plt.savefig(save_dir + 'rand_m_sky.png')

    # pdb.set_trace()

    galaxy_param_dict = {'RA': ra_g, 'DEC': dec_g, 'R': r_g, 'JK': jk_g}
    galaxy_random_param_dict = {'RA': ra_rand_g, 'DEC': dec_rand_g, 'R': r_rand_g, 'JK': jk_rand_g}
    matter_param_dict = {'RA': ra_m, 'DEC': dec_m, 'R': r_m, 'JK': jk_m}
    matter_random_param_dict = {'RA': ra_rand_m, 'DEC': dec_rand_m, 'R': r_rand_m, 'JK': jk_rand_m}
    other_params_dict = {'do_jk': do_jk, 'njk': njk, 'nrad': nrad, 'minrad': minrad, 'maxrad': maxrad,
                         'nthreads': nthreads}

    print 'setting up the class'
    corr3d = corr_class.correlate_3d(galaxy_param_dict, galaxy_random_param_dict, matter_param_dict,
                                     matter_random_param_dict, other_params_dict)
    if get_gg:
        print 'correlating gg'
        output_data_gg = corr3d.corr_gg()
        pk.dump(output_data_gg, open(save_dir + save_filename_gg, 'wb'))
    else:
        output_data_gg = pk.load(open(save_dir + save_filename_gg, "rb"))
    if get_gm:
        print 'correlating gm'
        output_data_gm = corr3d.corr_gm()
        pk.dump(output_data_gm, open(save_dir + save_filename_gm, 'wb'))
    else:
        output_data_gm = pk.load(open(save_dir + save_filename_gm, "rb"))
    if get_mm:
        print 'correlating mm'
        output_data_mm = corr3d.corr_mm()
        pk.dump(output_data_mm, open(save_dir + save_filename_mm, 'wb'))
    else:
        output_data_mm = pk.load(open(save_dir + save_filename_mm, "rb"))
    if get_ratio:
        print 'getting ratios of gg and gm with mm'
        output_data_gg_mm, output_data_gm_mm = corr3d.get_corr_gg_mm__gm_mm(output_data_gg, output_data_gm,
                                                                            output_data_mm)
        pk.dump(output_data_gg_mm, open(save_dir + save_filename_gg_mm, 'wb'))
        pk.dump(output_data_gm_mm, open(save_dir + save_filename_gm_mm, 'wb'))
    else:
        output_data_gg_mm = pk.load(open(save_dir + save_filename_gg_mm, "rb"))
        output_data_gm_mm = pk.load(open(save_dir + save_filename_gm_mm, "rb"))

    if make_plots:
        print 'making plots'
        corr_class.plot(save_dir + plot_save_name_gg_gm_mm, output_data_gg=output_data_gg,
                        output_data_gm=output_data_gm,
                        output_data_mm=output_data_mm)
        corr_class.plot(save_dir + plot_save_name_gg_gm_mm_ratio, output_data_gg_mm=output_data_gg_mm,
                        output_data_gm_mm=output_data_gm_mm)
