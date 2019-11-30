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
print('nthreads are ', nthreads)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--bin', required=True, type=int, help='bin to run the correlation for')
    parser.add_argument('--do_gg', default=1, type=int, help='Do run gg correlation')
    parser.add_argument('--do_gm', default=0, type=int, help='Do run gm correlation')
    parser.add_argument('--do_mm', default=0, type=int, help='Do run mm correlation')
    parser.add_argument('--get_ratio', default=0, type=int, help='Do get ratio of gg and mm')
    parser.add_argument('--make_plots', default=1, type=int, help='Do make the plots')
    parser.add_argument('--nrad', default=20, type=int, help='Number of radial bins to do the correlation')
    parser.add_argument('--minrad', default=0.8, type=float, help='Minimum of the radial bin')
    parser.add_argument('--maxrad', default=60.0, type=float, help='Maximum of the radial bin')
    parser.add_argument('--do_jk', default=False, type=bool, help='Do jack-knife covariance estimation')
    parser.add_argument('--njk_radec', default=60, type=int, help='Number of jack-knife patches in ra-dec dimensions')
    parser.add_argument('--njk_z', default=1, type=int, help='Number of jack-knife patches in redshift dimension. Put as 1 to only get patches in ra and dec')
    parser.add_argument('--ds_g', default=1, type=int, help='Ratio by which to downsample the galaxy catalog')
    parser.add_argument('--ds_m', default=1, type=int, help='Ratio by which to downsample the matter catalog')
    parser.add_argument('--ds_m_inp', default=2, type=int, help='Ratio by which matter catalog was initially downsampled')
    args_all = parser.parse_args()
    return args_all

if __name__ == "__main__":

    args = parse_arguments()
    # which tomographic bin to evaluate the correlation function of
    # binval = args.bin

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

    load_dir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/baseDC2_snapshot_z0.15_v0.1/process_cats/'
    # load_filename_matter = 'matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
    #     njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_inp) + '.fits'
    # load_filename_matter_randoms = 'randoms_matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
    #     njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_inp) + '.fits'
    ngal_sel = 1000000
    nrand_sel = 10 * ngal_sel
    load_filename_galaxy = 'galaxy_cat_small_xyz_ng' + str(ngal_sel) + '.fits'
    load_filename_galaxy_randoms = 'galaxy_rand_small_xyz_nr' + str(nrand_sel) + '.fits'


    save_dir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/baseDC2_snapshot_z0.15_v0.1/measurements/'
    save_filename_gg = 'gg_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_jk_' + str(do_jk)+ '_dsg_' + str(ds_g) + '.pk'
    # save_filename_mm = 'mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
    #     binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
    #     ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '.pk'
    # save_filename_gm = 'gm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
    #     binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
    #     ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '.pk'
    # save_filename_gg_mm = 'gg_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
    #     binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
    #     ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '.pk'
    # save_filename_gm_mm = 'gm_mm_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad) + '_zbin_' + str(
    #     binval) + '_jk_' + str(do_jk) + '_njkradec_' + str(njk_radec) + '_njkz_' + str(njk_z) + '_dsg_' + str(
    #     ds_g) + '_dsm_' + str(ds_m_inp*ds_m) + '.pk'

    plot_save_name_gg_gm_mm = 'gg_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad)  + '_jk_' + str(do_jk)+ '_dsg_' + str(ds_g) + '.pdf'
    plot_save_name_gg_gm_mm_ratio = 'gg_gm_mm_ratio_3dcorr_r_' + str(minrad) + '_' + str(maxrad) + '_nr_' + str(nrad)  + '_jk_' + str(do_jk)  + '_dsg_' + str(ds_g) + '.pdf'

    print('loading g')
    load_cat_g = fits.open(load_dir + load_filename_galaxy)

    # print 'loading m'
    # load_cat_m = fits.open(load_dir + load_filename_matter)

    print('loading rg')
    load_cat_rand_g = fits.open(load_dir + load_filename_galaxy_randoms)

    # print 'loading rm'
    # load_cat_rand_m = fits.open(load_dir + load_filename_matter_randoms)

    x_g, y_g, z_g = load_cat_g[1].data['X'], load_cat_g[1].data['Y'], load_cat_g[1].data['Z']
    # ra_m, dec_m, r_m, z_m, bin_m, jk_m = load_cat_m[1].data['RA'], load_cat_m[1].data['DEC'], load_cat_m[1].data['R'], \
    #                                      load_cat_m[1].data['Z'], load_cat_m[1].data['BIN'], load_cat_m[1].data['JK']

    x_rand_g, y_rand_g, z_rand_g = load_cat_rand_g[1].data['X'], load_cat_rand_g[1].data['Y'], load_cat_rand_g[1].data['Z']


    # ra_rand_m, dec_rand_m, r_rand_m, z_rand_m, bin_rand_m, jk_rand_m = load_cat_rand_m[1].data['RA'], \
    #                                                                    load_cat_rand_m[1].data['DEC'], \
    #                                                                    load_cat_rand_m[1].data['R'], \
    #                                                                    load_cat_rand_m[1].data['Z'], \
    #                                                                    load_cat_rand_m[1].data['BIN'], \
    #                                                                    load_cat_rand_m[1].data['JK']

    # use only the data corresponding to the given tomographic bin


    # pdb.set_trace()
    # if downsampled then truncate the data further
    if ds_g > 1:
        ind_ds_g = np.unique(np.random.randint(0, len(x_g), int(len(x_g) / ds_g)))
        ind_ds_rand_g = np.unique(np.random.randint(0, len(x_rand_g), int(len(x_rand_g) / ds_g)))

        x_g, y_g, z_g = x_g[ind_ds_g], y_g[ind_ds_g], z_g[ind_ds_g]

        x_rand_g, y_rand_g, z_rand_g = x_rand_g[ind_ds_rand_g], y_rand_g[ind_ds_rand_g], z_rand_g[ind_ds_rand_g]

    # if ds_m > 1:
    #     ind_ds_m = np.unique(np.random.randint(0, len(ra_m), len(ra_m) / ds_m))
    #     ind_ds_rand_m = np.unique(np.random.randint(0, len(ra_rand_m), len(ra_rand_m) / ds_m))
    #
    #     ra_m, dec_m, r_m, z_m, bin_m, jk_m = ra_m[ind_ds_m], dec_m[ind_ds_m], r_m[ind_ds_m], z_m[ind_ds_m], bin_m[ind_ds_m], \
    #                                          jk_m[ind_ds_m]
    #
    #     ra_rand_m, dec_rand_m, r_rand_m, z_rand_m, bin_rand_m, jk_rand_m = ra_rand_m[ind_ds_rand_m], dec_rand_m[
    #         ind_ds_rand_m], \
    #                                                                        r_rand_m[ind_ds_rand_m], z_rand_m[ind_ds_rand_m], \
    #                                                                        bin_rand_m[ind_ds_rand_m], \
    #                                                                        jk_rand_m[ind_ds_rand_m]

    print('number of galaxies : ', len(x_g))
    # print('number of matter particles : ', len(ra_m))
    print('number of galaxies randoms: ', len(x_rand_g))
    # print('number of matter randoms: ', len(ra_rand_m))

    # if make_plots:
    #     def eq2ang(ra, dec):
    #         phi = ra * np.pi / 180.
    #         theta = (np.pi / 2.) - dec * (np.pi / 180.)
    #         return theta, phi
    #
    #
    #     theta_m, phi_m = eq2ang(ra_rand_g, dec_rand_g)
    #     ind_m_f = hp.ang2pix(128, theta_m, phi_m)
    #     mask_m = np.zeros(hp.nside2npix(128))
    #     mask_m[ind_m_f] = 1
    #     plt.figure()
    #     hp.mollview(mask_m)
    #     plt.savefig(save_dir + 'rand_g_sky.png')
    #     plt.close()
    #
    #     theta_m, phi_m = eq2ang(ra_g, dec_g)
    #     ind_m_f = hp.ang2pix(128, theta_m, phi_m)
    #     mask_m = np.zeros(hp.nside2npix(128))
    #     mask_m[ind_m_f] = 1
    #     plt.figure()
    #     hp.mollview(mask_m)
    #     plt.savefig(save_dir + 'g_sky.png')
    #     plt.close()

    # pdb.set_trace()

    galaxy_param_dict = {'x': x_g, 'y': y_g, 'z': z_g, 'JK':None}
    galaxy_random_param_dict = {'x': x_rand_g, 'y': y_rand_g, 'z': z_rand_g, 'JK':None}
    # matter_param_dict = {'RA': ra_m, 'DEC': dec_m, 'R': r_m, 'JK': jk_m}
    # matter_random_param_dict = {'RA': ra_rand_m, 'DEC': dec_rand_m, 'R': r_rand_m, 'JK': jk_rand_m}
    matter_param_dict = {'RA': None, 'DEC': None, 'R': None, 'JK': None}
    matter_random_param_dict = {'RA': None, 'DEC': None, 'R': None, 'JK': None}
    other_params_dict = {'do_jk': do_jk, 'njk': njk, 'nrad': nrad, 'minrad': minrad, 'maxrad': maxrad, 'nthreads': nthreads}

    print('setting up the class')
    corr3d = corr_class.correlate_3d(galaxy_param_dict, galaxy_random_param_dict, matter_param_dict,
                                     matter_random_param_dict, other_params_dict)
    if get_gg:
        print('correlating gg')
        output_data_gg = corr3d.corr_gg()
        pk.dump(output_data_gg, open(save_dir + save_filename_gg, 'wb'))
    else:
        output_data_gg = pk.load(open(save_dir + save_filename_gg, "rb"))
    if get_gm:
        print('correlating gm')
        output_data_gm = corr3d.corr_gm()
        pk.dump(output_data_gm, open(save_dir + save_filename_gm, 'wb'))
    # else:
    #     output_data_gm = pk.load(open(save_dir + save_filename_gm, "rb"))
    if get_mm:
        print('correlating mm')
        output_data_mm = corr3d.corr_mm()
        pk.dump(output_data_mm, open(save_dir + save_filename_mm, 'wb'))
    # else:
    #     output_data_mm = pk.load(open(save_dir + save_filename_mm, "rb"))
    if get_ratio:
        print('getting ratios of gg and gm with mm')
        output_data_gg_mm, output_data_gm_mm = corr3d.get_corr_gg_mm__gm_mm(output_data_gg, output_data_gm, output_data_mm)
        pk.dump(output_data_gg_mm, open(save_dir + save_filename_gg_mm, 'wb'))
        pk.dump(output_data_gm_mm, open(save_dir + save_filename_gm_mm, 'wb'))
    # else:
    #     output_data_gg_mm = pk.load(open(save_dir + save_filename_gg_mm, "rb"))
    #     output_data_gm_mm = pk.load(open(save_dir + save_filename_gm_mm, "rb"))

    import camb
    from camb import model
    import scipy as sp

    sys.path.insert(0, '/global/u1/s/spandey/actxdes/sz_forecasts/helper/')
    import mycosmo as cosmodef
    import LSS_funcs as hmf

    cosmo_params = {'flat': True, 'H0': 71.0, 'Om0': 0.265, 'Ob0': 0.0448, 'sigma8': 0.8, 'ns': 0.963}

    h = cosmo_params['H0'] / 100.
    cosmo_func = cosmodef.mynew_cosmo(h, cosmo_params['Om0'], cosmo_params['Ob0'], cosmo_params['ns'],
                                      cosmo_params['sigma8'])

    k_array = np.logspace(-4, 2, 30000)
    Pklinz_z0_test = hmf.get_Pklinz(0.0, k_array, current_cosmo=cosmo_func)
    sig8h = hmf.sigRz0(8., k_array, Pklinz_z0_test, window='tophat')
    sig8_ratio = ((0.8 / sig8h) ** 2)
    Pklinz0 = sig8_ratio * Pklinz_z0_test
    Pklin = sig8_ratio * hmf.get_Pklinz(0.15, k_array, current_cosmo=cosmo_func)
    Pk_nl = hmf.Pkhalofit(k_array, Pklinz0, Pklin, 0.15, current_cosmo=cosmo_func)


    def get_corrfunc_realspace(r, karr, Pkarr):
        toint = (karr ** 2) * Pkarr * (np.sin(karr * r) / (karr * r))
        val = sp.integrate.simps(toint, karr)
        valf = (1 / (2 * np.pi ** 2)) * val
        return valf


    r_array = np.logspace(np.log10(0.8), np.log10(50), 15)
    k_full = np.logspace(-5, 3, 100000)

    xi_lin = np.zeros(len(r_array))

    for j in range(len(r_array)):
        Pk_interp = sp.interpolate.interp1d(np.log(k_array), np.log(Pklin), fill_value='extrapolate')
        Pk_full = np.exp(Pk_interp(np.log(k_full)))
        xi_lin[j] = get_corrfunc_realspace(r_array[j], k_full, Pk_full)

    xi_nl = np.zeros(len(r_array))

    for j in range(len(r_array)):
        Pk_interp = sp.interpolate.interp1d(np.log(k_array), np.log(Pk_nl), fill_value='extrapolate')
        Pk_full = np.exp(Pk_interp(np.log(k_full)))
        xi_nl[j] = get_corrfunc_realspace(r_array[j], k_full, Pk_full)

    xi_nl_interp = interpolate.interp1d(np.log(r_array), np.log(xi_nl),fill_value='extrapolate')
    output_data_gg_xinl = output_data_gg['xi_gg_full']/np.exp(xi_nl_interp(np.log(output_data_gg['r_gg'])))

    if make_plots:
        print('making plots')
        corr_class.plot(save_dir + plot_save_name_gg_gm_mm, output_data_gg=output_data_gg, xi_lin=xi_lin, xi_nl=xi_nl, r_array=r_array,zeval=0.15)
        corr_class.plot(save_dir + plot_save_name_gg_gm_mm_ratio, output_data_gg_xinl=[output_data_gg['r_gg'],output_data_gg_xinl])
