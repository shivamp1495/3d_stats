import sys, os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import treecorr
print(treecorr.__version__)
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import kmeans_radec
from numpy.random import rand
import pickle as pk
import matplotlib.cm as cm
import scipy.interpolate as interpolate
import pdb
import time
import multiprocessing as mp

nthreads = mp.cpu_count()


class correlate_3d:

    def __init__(self, galaxy_param, galaxy_random_param, matter_param, matter_random_param,
                 other_params):

        if 'RA' in galaxy_param.keys():
            if galaxy_param['RA'] is not None:
                self.ra_g, self.dec_g, self.r_g, self.jk_g = galaxy_param['RA'], galaxy_param['DEC'], galaxy_param['R'], galaxy_param['JK']
                self.ra_rand_g, self.dec_rand_g, self.r_rand_g, self.jk_rand_g = galaxy_random_param['RA'], galaxy_random_param[
                    'DEC'], galaxy_random_param['R'],galaxy_random_param['JK']
                self.n_g = len(self.ra_g)
                self.n_rg = len(self.ra_rand_g)
                print('number of galaxies ', len(self.ra_g))
                print('number of galaxy randoms ', len(self.ra_rand_g))
                self.cat_g = treecorr.Catalog(ra=self.ra_g, dec=self.dec_g, r=self.r_g, ra_units='degrees', dec_units='degrees')
                self.cat_rand_g = treecorr.Catalog(ra=self.ra_rand_g, dec=self.dec_rand_g, r=self.r_rand_g, ra_units='degrees',
                                                   dec_units='degrees')
                self.cat_type = 'radec'

        if 'x' in galaxy_param.keys():
            if galaxy_param['x'] is not None:
                self.x_g, self.y_g, self.z_g, self.jk_g = galaxy_param['x'], galaxy_param['y'], galaxy_param['z'], galaxy_param['JK']
                self.x_rand_g, self.y_rand_g, self.z_rand_g, self.jk_rand_g = galaxy_random_param['x'], galaxy_random_param[
                    'y'], galaxy_random_param['z'],galaxy_random_param['JK']
                self.n_g = len(self.x_g)
                self.n_rg = len(self.x_rand_g)
                print('number of galaxies ', len(self.x_g))
                print('number of galaxy randoms ', len(self.x_rand_g))
                self.cat_g = treecorr.Catalog(x=self.x_g, y=self.y_g, z=self.z_g)
                self.cat_rand_g = treecorr.Catalog(x=self.x_rand_g, y=self.y_rand_g, z=self.z_rand_g)
                self.cat_type = 'xyz'

        if 'RA' in matter_param.keys():
            if matter_param['RA'] is not None:
                self.ra_m, self.dec_m, self.r_m, self.jk_m = matter_param['RA'], matter_param['DEC'], matter_param['R'], matter_param['JK']

                self.ra_rand_m, self.dec_rand_m, self.r_rand_m, self.jk_rand_m = matter_random_param['RA'], matter_random_param[
                    'DEC'], matter_random_param['R'], matter_random_param['JK']
                self.n_m = len(self.ra_m)
                self.n_rm = len(self.ra_rand_m)
                print('number of matter ', len(self.ra_m))
                print('number of matter randoms ', len(self.ra_rand_m))
                self.cat_m = treecorr.Catalog(ra=self.ra_m, dec=self.dec_m, r=self.r_m, ra_units='degrees', dec_units='degrees')
                self.cat_rand_m = treecorr.Catalog(ra=self.ra_rand_m, dec=self.dec_rand_m, r=self.r_rand_m, ra_units='degrees',
                                                   dec_units='degrees')
                self.cat_type = 'radec'

        if 'x' in matter_param.keys():
            if matter_param['x'] is not None:
                self.x_m, self.y_m, self.z_m, self.jk_m = matter_param['x'], matter_param['y'], matter_param['z'], matter_param['JK']
                self.x_rand_m, self.y_rand_m, self.z_rand_m, self.jk_rand_m = matter_random_param['x'], matter_random_param[
                    'y'], matter_random_param['z'],matter_random_param['JK']
                self.n_m = len(self.x_m)
                self.n_rm = len(self.x_rand_m)
                print('number of galaxies ', len(self.x_m))
                print('number of matter randoms ', len(self.x_rand_m))
                self.cat_m = treecorr.Catalog(x=self.x_m, y=self.y_m, z=self.z_m)
                self.cat_rand_m = treecorr.Catalog(x=self.x_rand_m, y=self.y_rand_m, z=self.z_rand_m)
                self.cat_type = 'xyz'


        self.nrad, self.minrad, self.maxrad, self.nthreads = other_params['nrad'], other_params['minrad'], \
                                                             other_params['maxrad'], other_params['nthreads']
        self.do_jk,self.njk = other_params['do_jk'],other_params['njk']
        self.bin_slop = other_params['bin_slop']

    # Calculate galaxy galaxy correlation function
    def corr_gg(self):

        g_g = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                     num_threads=self.nthreads, bin_slop=self.bin_slop)
        g_rg = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                      num_threads=self.nthreads, bin_slop=self.bin_slop)
        rg_rg = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                       num_threads=self.nthreads, bin_slop=self.bin_slop)

        t_g_g_i = time.time()
        print('correlating g g ')
        g_g.process(self.cat_g, self.cat_g)
        t_g_g_f = time.time()
        print('time for correlating g g ', t_g_g_f - t_g_g_i)

        t_g_rg_i = time.time()
        print('correlating g rg ')
        g_rg.process(self.cat_g, self.cat_rand_g)
        t_g_rg_f = time.time()
        print('time for correlating g rg ', t_g_rg_f - t_g_rg_i)

        t_rg_rg_i = time.time()
        print('correlating rg rg ')
        rg_rg.process(self.cat_rand_g, self.cat_rand_g)
        t_rg_rg_f = time.time()
        print('time for correlating rg rg ', t_rg_rg_f - t_rg_rg_i)

        g_g_np_norm = g_g.npairs * 1. / (1. * self.n_g * self.n_g)

        g_rg_np_norm = g_rg.npairs * 1. / (1. * self.n_g * self.n_rg)

        rg_rg_np_norm = rg_rg.npairs * 1. / (1. * self.n_rg * self.n_rg)

        xi_gg_full = (g_g_np_norm - 2. * g_rg_np_norm + rg_rg_np_norm) / (1. * rg_rg_np_norm)
        r_gg = np.exp(g_g.meanlogr)

        print('r_gg', r_gg)
        print('xi_gg', xi_gg_full)

        xigg_big_all = np.zeros((self.njk, len(r_gg)))
        rnom_big_all = np.zeros((self.njk, len(r_gg)))

        if self.do_jk:
            for j in range(self.njk):

                if np.mod(j, 20) == 0:
                    print('processing jk', j)

                ind_g_jk = np.where(self.jk_g == j)[0]
                ind_rg_jk = np.where(self.jk_rand_g == j)[0]

                n_g_s, n_rg_s = len(ind_g_jk), len(ind_rg_jk)
                n_g_b, n_rg_b = self.n_g - n_g_s, self.n_rg - n_rg_s

                if self.cat_type == 'radec':
                    cat_g_s = treecorr.Catalog(ra=self.ra_g[ind_g_jk], dec=self.dec_g[ind_g_jk], r=self.r_g[ind_g_jk],
                                               ra_units='degrees', dec_units='degrees')

                    cat_rand_g_s = treecorr.Catalog(ra=self.ra_rand_g[ind_rg_jk], dec=self.dec_rand_g[ind_rg_jk],
                                                    r=self.r_rand_g[ind_rg_jk], ra_units='degrees', dec_units='degrees')

                if self.cat_type == 'xyz':
                    cat_g_s = treecorr.Catalog(x=self.x_g[ind_g_jk], y=self.y_g[ind_g_jk], z=self.z_g[ind_g_jk])

                    cat_rand_g_s = treecorr.Catalog(x=self.x_rand_g[ind_rg_jk], y=self.y_rand_g[ind_rg_jk],
                                                    z=self.z_rand_g[ind_rg_jk])

                g_g_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)
                g_g_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_rg_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)
                rg_rg_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rg_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rg_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rg_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_g_s_s.process(cat_g_s, cat_g_s)
                g_g_f_s.process(self.cat_g, cat_g_s)
                rg_rg_s_s.process(cat_rand_g_s, cat_rand_g_s)
                rg_rg_f_s.process(self.cat_rand_g, cat_rand_g_s)
                g_rg_s_s.process(cat_g_s, cat_rand_g_s)
                g_rg_f_s.process(self.cat_g, cat_rand_g_s)
                g_rg_s_f.process(cat_g_s, self.cat_rand_g)

                g_g_s_s_np, g_g_f_s_np = g_g_s_s.npairs, g_g_f_s.npairs
                rg_rg_s_s_np, rg_rg_f_s_np = rg_rg_s_s.npairs, rg_rg_f_s.npairs
                g_rg_s_s_np, g_rg_f_s_np, g_rg_s_f_np = g_rg_s_s.npairs, g_rg_f_s.npairs, g_rg_s_f.npairs

                g_g_b_b_np_norm = (g_g.npairs - 2. * g_g_f_s_np + g_g_s_s_np) / (1. * n_g_b * n_g_b)
                rg_rg_b_b_np_norm = (rg_rg.npairs - 2. * rg_rg_f_s_np + rg_rg_s_s_np) / (1. * n_rg_b * n_rg_b)
                g_rg_b_b_np_norm = (g_rg.npairs - g_rg_f_s_np - g_rg_s_f_np + g_rg_s_s_np) / (1. * n_g_b * n_rg_b)

                xi_gg_big = (g_g_b_b_np_norm - 2. * g_rg_b_b_np_norm + rg_rg_b_b_np_norm) / (1. * rg_rg_b_b_np_norm)

                xigg_big_all[j, :] = xi_gg_big
                rnom_big_all[j, :] = np.exp(g_g_s_s.meanlogr)

        if self.do_jk:
            xi_gg_mean = np.tile(xi_gg_full.transpose(), (self.njk, 1))
            xi_gg_sub = xigg_big_all - xi_gg_mean
            xi_gg_cov = (1.0 * (self.njk - 1.) / self.njk) * np.matmul(xi_gg_sub.T, xi_gg_sub)
            xi_gg_sig = np.sqrt(np.diag(xi_gg_cov))
            output_data = {'g_g': g_g, 'g_rg': g_rg, 'rg_rg': rg_rg, 'n_g': self.n_g, 'n_rg': self.n_rg,
                           'xi_gg_full': xi_gg_full, 'r_gg': r_gg, 'xigg_big_all': xigg_big_all,
                           'self.r_gg_all': rnom_big_all,
                           'cov': xi_gg_cov, 'sig': xi_gg_sig}
        else:
            output_data = {'g_g': g_g, 'g_rg': g_rg, 'rg_rg': rg_rg, 'n_g': self.n_g, 'n_rg': self.n_rg,
                           'xi_gg_full': xi_gg_full, 'r_gg': r_gg}

        return output_data

    # Calculate galaxy matter correlation function

    def corr_gm(self):
        g_m = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                     num_threads=self.nthreads, bin_slop=self.bin_slop)
        g_rm = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                      num_threads=self.nthreads, bin_slop=self.bin_slop)
        m_rg = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                      num_threads=self.nthreads, bin_slop=self.bin_slop)
        rm_rg = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                       num_threads=self.nthreads, bin_slop=self.bin_slop)

        t_g_m_i = time.time()
        print('correlating g m ')
        g_m.process(self.cat_g, self.cat_m)
        t_g_m_f = time.time()
        print('time for correlating g m ', t_g_m_f - t_g_m_i)

        t_g_rm_i = time.time()
        print('correlating g rm ')
        g_rm.process(self.cat_g, self.cat_rand_m)
        t_g_rm_f = time.time()
        print('time for correlating g rm ', t_g_rm_f - t_g_rm_i)

        t_m_rg_i = time.time()
        print('correlating m rg ')
        m_rg.process(self.cat_m, self.cat_rand_g)
        t_m_rg_f = time.time()
        print('time for correlating m rg ', t_m_rg_f - t_m_rg_i)

        t_rm_rg_i = time.time()
        print('correlating rm rg ')
        rm_rg.process(self.cat_rand_m, self.cat_rand_g)
        t_rm_rg_f = time.time()
        print('time for correlating rm rg ', t_rm_rg_f - t_rm_rg_i)

        g_m_npairs_norm = g_m.npairs * 1. / (1. * self.n_g * self.n_m)

        g_rm_npairs_norm = g_rm.npairs * 1. / (1. * self.n_g * self.n_rm)

        m_rg_npairs_norm = m_rg.npairs * 1. / (1. * self.n_rg * self.n_m)

        rm_rg_npairs_norm = rm_rg.npairs * 1. / (1. * self.n_rg * self.n_rm)

        xi_gm_full = (g_m_npairs_norm - g_rm_npairs_norm - m_rg_npairs_norm + rm_rg_npairs_norm) / (
                1. * rm_rg_npairs_norm)
        r_gm = np.exp(g_m.meanlogr)

        print('r_gm', r_gm)
        print('xi_gm', xi_gm_full)

        xigm_big_all = np.zeros((self.njk, len(r_gm)))
        rnom_big_all = np.zeros((self.njk, len(r_gm)))

        if self.do_jk:
            for j in range(self.njk):

                if np.mod(j, 20) == 0:
                    print('processing jk', j)

                ind_g_jk = np.where(self.jk_g == j)[0]
                ind_rg_jk = np.where(self.jk_rand_g == j)[0]

                n_g_s, n_rg_s = len(ind_g_jk), len(ind_rg_jk)
                n_g_b, n_rg_b = self.n_g - n_g_s, self.n_rg - n_rg_s

                ind_m_jk = np.where(self.jk_m == j)[0]
                ind_rm_jk = np.where(self.jk_rand_m == j)[0]

                n_m_s, n_rm_s = len(ind_m_jk), len(ind_rm_jk)
                n_m_b, n_rm_b = self.n_m - n_m_s, self.n_rm - n_rm_s
                
                if self.cat_type == 'radec':
                    cat_g_s = treecorr.Catalog(ra=self.ra_g[ind_g_jk], dec=self.dec_g[ind_g_jk], r=self.r_g[ind_g_jk],
                                               ra_units='degrees',
                                               dec_units='degrees')
    
                    cat_rand_g_s = treecorr.Catalog(ra=self.ra_rand_g[ind_rg_jk], dec=self.dec_rand_g[ind_rg_jk],
                                                    r=self.r_rand_g[ind_rg_jk],
                                                    ra_units='degrees', dec_units='degrees')
    
                    cat_m_s = treecorr.Catalog(ra=self.ra_m[ind_m_jk], dec=self.dec_m[ind_m_jk], r=self.r_m[ind_m_jk],
                                               ra_units='degrees',
                                               dec_units='degrees')
    
                    cat_rand_m_s = treecorr.Catalog(ra=self.ra_rand_m[ind_rm_jk], dec=self.dec_rand_m[ind_rm_jk],
                                                    r=self.r_rand_m[ind_rm_jk],
                                                    ra_units='degrees', dec_units='degrees')
                
                if self.cat_type == 'xyz':
                    cat_g_s = treecorr.Catalog(x=self.x_g[ind_g_jk], y=self.y_g[ind_g_jk], z=self.z_g[ind_g_jk])

                    cat_rand_g_s = treecorr.Catalog(x=self.x_rand_g[ind_rg_jk], y=self.y_rand_g[ind_rg_jk],
                                                    z=self.z_rand_g[ind_rg_jk])

                    cat_m_s = treecorr.Catalog(x=self.x_m[ind_m_jk], y=self.y_m[ind_m_jk], z=self.z_m[ind_m_jk])

                    cat_rand_m_s = treecorr.Catalog(x=self.x_rand_m[ind_rm_jk], y=self.y_rand_m[ind_rm_jk],
                                                    z=self.z_rand_m[ind_rm_jk])

                g_m_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_m_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_m_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rm_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rm_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_rm_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_m_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_m_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_m_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_rm_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_rm_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)

                rg_rm_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)

                g_m_s_s.process(cat_g_s, cat_m_s)
                g_m_f_s.process(self.cat_g, cat_m_s)
                g_m_s_f.process(cat_g_s, self.cat_m)
                g_m_s_s_np, g_m_f_s_np, g_m_s_f_np = g_m_s_s.npairs, g_m_f_s.npairs, g_m_s_f.npairs

                g_rm_s_s.process(cat_g_s, cat_rand_m_s)
                g_rm_f_s.process(self.cat_g, cat_rand_m_s)
                g_rm_s_f.process(cat_g_s, self.cat_rand_m)
                g_rm_s_s_np, g_rm_f_s_np, g_rm_s_f_np = g_rm_s_s.npairs, g_rm_f_s.npairs, g_rm_s_f.npairs

                rg_m_s_s.process(cat_rand_g_s, cat_m_s)
                rg_m_f_s.process(self.cat_rand_g, cat_m_s)
                rg_m_s_f.process(cat_rand_g_s, self.cat_m)
                rg_m_s_s_np, rg_m_f_s_np, rg_m_s_f_np = rg_m_s_s.npairs, rg_m_f_s.npairs, rg_m_s_f.npairs

                rg_rm_s_s.process(cat_rand_g_s, cat_rand_m_s)
                rg_rm_f_s.process(self.cat_rand_g, cat_rand_m_s)
                rg_rm_s_f.process(cat_rand_g_s, self.cat_rand_m)
                rg_rm_s_s_np, rg_rm_f_s_np, rg_rm_s_f_np = rg_rm_s_s.npairs, rg_rm_f_s.npairs, rg_rm_s_f.npairs

                g_m_b_b_np_norm = (g_m.npairs - g_m_f_s_np - g_m_s_f_np + g_m_s_s_np) / (1. * n_g_b * n_m_b)

                g_rm_b_b_np_norm = (g_rm.npairs - g_rm_f_s_np - g_rm_s_f_np + g_rm_s_s_np) / (1. * n_g_b * n_rm_b)

                rg_m_b_b_np_norm = (m_rg.npairs - rg_m_f_s_np - rg_m_s_f_np + rg_m_s_s_np) / (1. * n_rg_b * n_m_b)

                rg_rm_b_b_np_norm = (rm_rg.npairs - rg_rm_f_s_np - rg_rm_s_f_np + rg_rm_s_s_np) / (1. * n_rg_b * n_rm_b)

                xi_gm_big = (g_m_b_b_np_norm - g_rm_b_b_np_norm - rg_m_b_b_np_norm + rg_rm_b_b_np_norm) / (
                        1. * rg_rm_b_b_np_norm)

                xigm_big_all[j, :] = xi_gm_big
                rnom_big_all[j, :] = np.exp(g_m_s_s.meanlogr)

        if self.do_jk:
            xi_gm_mean = np.tile(xi_gm_full.transpose(), (self.njk, 1))
            xi_gm_sub = xigm_big_all - xi_gm_mean
            xi_gm_cov = (1.0 * (self.njk - 1.) / self.njk) * np.matmul(xi_gm_sub.T, xi_gm_sub)
            xi_gm_sig = np.sqrt(np.diag(xi_gm_cov))
            output_data = {'g_m': g_m, 'g_rm': g_rm, 'm_rg': m_rg, 'rm_rg': rm_rg, 'n_g': self.n_g,
                           'n_rg': self.n_rg,'n_m': self.n_m, 'n_rm': self.n_rm, 'xi_gm_full': xi_gm_full, 'r_gm': r_gm,
                           'xigm_big_all': xigm_big_all, 'self.r_gm_all': rnom_big_all, 'cov': xi_gm_cov,
                           'sig': xi_gm_sig}
        else:
            output_data = {'g_m': g_m, 'g_rm': g_rm, 'm_rg': m_rg, 'rm_rg': rm_rg, 'n_g': self.n_g,
                           'n_rg': self.n_rg,'n_m': self.n_m, 'n_rm': self.n_rm, 'xi_gm_full': xi_gm_full, 'r_gm': r_gm}

        return output_data

    # Calculate matter matter correlation function
    def corr_mm(self):
        m_m = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                     num_threads=self.nthreads, bin_slop=self.bin_slop)
        m_rm = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                      num_threads=self.nthreads, bin_slop=self.bin_slop)
        rm_rm = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                       num_threads=self.nthreads, bin_slop=self.bin_slop)

        t_m_m_i = time.time()
        print('correlatinm m m ')
        m_m.process(self.cat_m, self.cat_m)
        t_m_m_f = time.time()
        print('time for correlatinm m m ', t_m_m_f - t_m_m_i)

        t_m_rm_i = time.time()
        print('correlatinm m rm ')
        m_rm.process(self.cat_m, self.cat_rand_m)
        t_m_rm_f = time.time()
        print('time for correlatinm m rm ', t_m_rm_f - t_m_rm_i)

        t_rm_rm_i = time.time()
        print('correlatinm rm rm ')
        rm_rm.process(self.cat_rand_m, self.cat_rand_m)
        t_rm_rm_f = time.time()
        print('time for correlatinm rm rm ', t_rm_rm_f - t_rm_rm_i)

        m_m_npairs_norm = m_m.npairs * 1. / (1. * self.n_m * self.n_m)

        m_rm_npairs_norm = m_rm.npairs * 1. / (1. * self.n_m * self.n_rm)

        rm_rm_npairs_norm = rm_rm.npairs * 1. / (1. * self.n_rm * self.n_rm)

        xi_mm_full = (m_m_npairs_norm - 2. * m_rm_npairs_norm + rm_rm_npairs_norm) / (1. * rm_rm_npairs_norm)
        r_mm = np.exp(m_m.meanlogr)

        print('r_mm', r_mm)
        print('xi_mm', xi_mm_full)

        ximm_big_all = np.zeros((self.njk, len(r_mm)))
        rnom_big_all = np.zeros((self.njk, len(r_mm)))

        if self.do_jk:
            for j in range(self.njk):

                if np.mod(j, 20) == 0:
                    print('processing jk', j)

                ind_m_jk = np.where(self.jk_m == j)[0]
                ind_rm_jk = np.where(self.jk_rand_m == j)[0]

                n_m_s, n_rm_s = len(ind_m_jk), len(ind_rm_jk)
                n_m_b, n_rm_b = self.n_m - n_m_s, self.n_rm - n_rm_s

                if self.cat_type == 'radec':
                    cat_m_s = treecorr.Catalog(ra=self.ra_m[ind_m_jk], dec=self.dec_m[ind_m_jk], r=self.r_m[ind_m_jk],
                                               ra_units='degrees',
                                               dec_units='degrees')

                    cat_rand_m_s = treecorr.Catalog(ra=self.ra_rand_m[ind_rm_jk], dec=self.dec_rand_m[ind_rm_jk],
                                                    r=self.r_rand_m[ind_rm_jk],
                                                    ra_units='degrees', dec_units='degrees')

                if self.cat_type == 'xyz':
                    cat_m_s = treecorr.Catalog(x=self.x_m[ind_m_jk], y=self.y_m[ind_m_jk], z=self.z_m[ind_m_jk])

                    cat_rand_m_s = treecorr.Catalog(x=self.x_rand_m[ind_rm_jk], y=self.y_rand_m[ind_rm_jk],
                                                    z=self.z_rand_m[ind_rm_jk])

                m_m_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)
                m_m_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                 num_threads=self.nthreads, bin_slop=self.bin_slop)

                rm_rm_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)
                rm_rm_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                   num_threads=self.nthreads, bin_slop=self.bin_slop)

                m_rm_s_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                m_rm_f_s = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                m_rm_s_f = treecorr.NNCorrelation(nbins=self.nrad, min_sep=self.minrad, max_sep=self.maxrad, verbose=0,
                                                  num_threads=self.nthreads, bin_slop=self.bin_slop)

                m_m_s_s.process(cat_m_s, cat_m_s)
                m_m_f_s.process(self.cat_m, cat_m_s)
                rm_rm_s_s.process(cat_rand_m_s, cat_rand_m_s)
                rm_rm_f_s.process(self.cat_rand_m, cat_rand_m_s)
                m_rm_s_s.process(cat_m_s, cat_rand_m_s)
                m_rm_f_s.process(self.cat_m, cat_rand_m_s)
                m_rm_s_f.process(cat_m_s, self.cat_rand_m)

                m_m_s_s_np, m_m_f_s_np = m_m_s_s.npairs, m_m_f_s.npairs
                rm_rm_s_s_np, rm_rm_f_s_np = rm_rm_s_s.npairs, rm_rm_f_s.npairs
                m_rm_s_s_np, m_rm_f_s_np, m_rm_s_f_np = m_rm_s_s.npairs, m_rm_f_s.npairs, m_rm_s_f.npairs

                m_m_b_b_np_norm = (m_m.npairs - 2. * m_m_f_s_np + m_m_s_s_np) / (1. * n_m_b * n_m_b)
                rm_rm_b_b_np_norm = (rm_rm.npairs - 2. * rm_rm_f_s_np + rm_rm_s_s_np) / (1. * n_rm_b * n_rm_b)
                m_rm_b_b_np_norm = (m_rm.npairs - m_rm_f_s_np - m_rm_s_f_np + m_rm_s_s_np) / (1. * n_m_b * n_rm_b)

                xi_mm_big = (m_m_b_b_np_norm - 2. * m_rm_b_b_np_norm + rm_rm_b_b_np_norm) / (1. * rm_rm_b_b_np_norm)

                ximm_big_all[j, :] = xi_mm_big
                rnom_big_all[j, :] = np.exp(m_m_s_s.meanlogr)

        if self.do_jk:
            xi_mm_mean = np.tile(xi_mm_full.transpose(), (self.njk, 1))
            xi_mm_sub = ximm_big_all - xi_mm_mean
            xi_mm_cov = (1.0 * (self.njk - 1.) / self.njk) * np.matmul(xi_mm_sub.T, xi_mm_sub)
            xi_mm_sig = np.sqrt(np.diag(xi_mm_cov))
            output_data = {'m_m': m_m, 'm_rm': m_rm, 'rm_rm': rm_rm, 'n_m': self.n_m, 'n_rm': self.n_rm,
                           'xi_mm_full': xi_mm_full, 'r_mm': r_mm, 'ximm_big_all': ximm_big_all,
                           'self.r_mm_all': rnom_big_all,
                           'cov': xi_mm_cov, 'sig': xi_mm_sig}
        else:
            output_data = {'m_m': m_m, 'm_rm': m_rm, 'rm_rm': rm_rm, 'n_m': self.n_m, 'n_rm': self.n_rm,
                           'xi_mm_full': xi_mm_full, 'r_mm': r_mm}

        return output_data


    def get_corr_gg_mm__gm_mm(self,output_data_gg,output_data_gm,output_data_mm):

        xi_gg_full,xigg_big_all = output_data_gg['xi_gg_full'],output_data_gg['xigg_big_all']
        xi_gm_full,xigm_big_all = output_data_gm['xi_gm_full'],output_data_gm['xigm_big_all']
        xi_mm_full,ximm_big_all = output_data_mm['xi_mm_full'],output_data_mm['ximm_big_all']

        xi_gg_mm_full = xi_gg_full/xi_mm_full
        xi_gg_mm_mean = np.tile(xi_gg_mm_full.transpose(), (self.njk, 1))
        xi_gg_mm_big_all = xigg_big_all/ximm_big_all
        xi_gg_mm_sub = xi_gg_mm_big_all - xi_gg_mm_mean
        xi_gg_mm_cov = (1.0 * (self.njk - 1.) / self.njk) * np.matmul(xi_gg_mm_sub.T, xi_gg_mm_sub)
        xi_gg_mm_sig = np.sqrt(np.diag(xi_gg_mm_cov))

        xi_gm_mm_full = xi_gm_full/xi_mm_full
        xi_gm_mm_mean = np.tile(xi_gm_mm_full.transpose(), (self.njk, 1))
        xi_gm_mm_big_all = xigm_big_all/ximm_big_all
        xi_gm_mm_sub = xi_gm_mm_big_all - xi_gm_mm_mean
        xi_gm_mm_cov = (1.0 * (self.njk - 1.) / self.njk) * np.matmul(xi_gm_mm_sub.T, xi_gm_mm_sub)
        xi_gm_mm_sig = np.sqrt(np.diag(xi_gm_mm_cov))

        output_data_gg_mm = dict(output_data_gg,**output_data_mm)
        output_data_gg_mm['xi_gg_mm_full'] = xi_gg_mm_full
        output_data_gg_mm['xi_gg_mm_big_all'] = xi_gg_mm_big_all
        output_data_gg_mm['cov'] = xi_gg_mm_cov
        output_data_gg_mm['sig'] = xi_gg_mm_sig

        output_data_gm_mm = dict(output_data_gm,**output_data_mm)
        output_data_gm_mm['xi_gm_mm_full'] = xi_gm_mm_full
        output_data_gm_mm['xi_gm_mm_big_all'] = xi_gm_mm_big_all
        output_data_gm_mm['cov'] = xi_gm_mm_cov
        output_data_gm_mm['sig'] = xi_gm_mm_sig

        return output_data_gg_mm, output_data_gm_mm


def plot(plot_save_name,output_data_gg=None,output_data_gm=None,output_data_mm=None,output_data_gg_mm=None, output_data_gm_mm=None, xi_lin=None, xi_nl=None, r_array=None, zeval=None, output_data_gg_xinl=None):

    # Plot the correlation function with the errorbars

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    j = 0

    if output_data_gm is not None:
        if 'sig' in output_data_gm.keys():
            ax.errorbar((1.01**j)*output_data_gm['r_gm'], output_data_gm['xi_gm_full'], output_data_gm['sig'], color='blue', label=r'$\xi_{gm}$', marker='*', linestyle='')
        j += 1
        ax.set_yscale('log')
        
    if output_data_gg is not None:
        if 'sig' in output_data_gg.keys():
            ax.errorbar((1.01**j)*output_data_gg['r_gg'], output_data_gg['xi_gg_full'], output_data_gg['sig'], color='red', label=r'$\xi_{gg}$', marker='*', linestyle='')
        else:
            ax.plot((1.01 ** j) * output_data_gg['r_gg'], output_data_gg['xi_gg_full'],
                        color='red', label=r'$\xi_{gg}$', marker='*', linestyle='')
        j += 1
        ax.set_yscale('log')
        
    if output_data_mm is not None:
        ax.errorbar((1.01**j)*output_data_mm['r_mm'], output_data_mm['xi_mm_full'], output_data_mm['sig'], color='black', label=r'$\xi_{mm}$', marker='*', linestyle='')
        j += 1
        ax.set_yscale('log')

    if output_data_gm_mm is not None:
        ax.errorbar((1.01**j)*output_data_gm_mm['r_gm'], output_data_gm_mm['xi_gm_mm_full'], output_data_gm_mm['sig'], color='magenta', label=r'$\xi_{gm}/\xi_{mm}$', marker='*', linestyle='')
        j += 1

    if output_data_gg_mm is not None:
        ax.errorbar((1.01**j)*output_data_gg_mm['r_gg'], output_data_gg_mm['xi_gg_mm_full'], output_data_gg_mm['sig'], color='green', label=r'$\xi_{gg}/\xi_{mm}$', marker='*', linestyle='')
        j += 1

    if output_data_gg_xinl is not None:
        ax.errorbar((1.01**j)*output_data_gg_xinl[0], output_data_gg_xinl[1], color='green', label=r'$\xi_{gg}/\xi_{nl}$', marker='*', linestyle='')
        j += 1


    if xi_lin is not None:
        ax.plot(r_array, xi_lin, color='black', label=r'$\xi_{lin}(z=' + str(zeval) + ')$')

    if xi_nl is not None:
        ax.plot(r_array, xi_nl, color='magenta', label=r'$\xi_{nl}(z=' + str(zeval) + ')$')


    ax.legend(fontsize=15, frameon=False)
    ax.set_xlabel(r'r(Mpc/h)', size=20)
    ax.set_ylabel(r'$\xi$', size=20)
    ax.set_xscale('log')


    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)

    plt.savefig(plot_save_name)











