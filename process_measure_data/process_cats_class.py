import sys, os
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import numpy as np
import matplotlib
import esutil
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
import mycosmo as cosmodef
import multiprocessing


def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec


def eq2ang(ra, dec):
    phi = ra * np.pi / 180.
    theta = (np.pi / 2.) - dec * (np.pi / 180.)
    return theta, phi


def get_jkobj(radec_mat, njk):
    jkobj_map = kmeans_radec.kmeans_sample(radec_mat, njk, maxiter=200)
    return jkobj_map


def get_hist_arrays(bin_min, bin_max, nbins_hist):
    delta_bin = (bin_max - bin_min) / nbins_hist
    bin_centers_all = np.linspace(bin_min, bin_max, nbins_hist)
    bin_edges = (bin_centers_all[1:] + bin_centers_all[:-1]) / 2.
    bin_centers = bin_centers_all[1:-1]
    return bin_centers, bin_edges, delta_bin


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

    def get_Dcom_array(self,zarray):
        Omega_m = self.cosmo.Om0
        Omega_L = 1. - Omega_m
        c = 3 * 10 ** 5
        Dcom_array = np.zeros(len(zarray))
        for j in range(len(zarray)):
            zf = zarray[j]
            res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
            Dcom = res1[0]
            Dcom_array[j] = Dcom
        return Dcom_array

    def get_Hz(self,zarray):
        Omega_m = self.cosmo.Om0
        Omega_L = 1 - Omega_m
        Ez = np.sqrt(Omega_m * (1 + zarray) ** 3 + Omega_L)
        Hz = 100. * Ez
        return Hz

    def get_diff(self, zf, chi):
        return chi - self.get_Dcom(zf)

    def root_find(self, init_x, chi):
        nll = lambda *args: self.get_diff(*args)
        result = op.root(nll, np.array([init_x]), args=chi, options={'maxfev': 50}, tol=0.01)
        return result.x[0]

    def get_z_from_chi(self, chi):
        valf = self.root_find(0., chi)
        return valf


class Catalog_funcs:
    def __init__(self, data_ra_array, data_dec_array, data_z_array, cosmo_params, other_params, data_w_array = None):
        self.datapoint_ra, self.datapoint_dec, self.datapoint_z = data_ra_array, data_dec_array, data_z_array
        if data_w_array is None:
            self.datapoint_w = np.ones(len(self.datapoint_ra))
        else:
            self.datapoint_w = data_w_array

        self.gnf = general_funcs(cosmo_params)
        self.chi_interp = other_params['chi_interp']
        self.z_interp = other_params['z_interp']

        self.zmin_bins = other_params['zmin_bins']
        self.zmax_bins = other_params['zmax_bins']
        self.bin_n_array = other_params['bin_n_array']
        self.bin_array = other_params['bin_array']
        self.jkobj_map_2d = other_params['jkobj_map_radec']
        self.njk_radec = other_params['njk_radec']
        self.njk_z = other_params['njk_z']

    def get_chi(self, z_all):
        datapoint_chi = self.chi_interp(z_all)
        return datapoint_chi

    def create_random_cat_uniform_esutil(self, nrand_fac=10, zarray=None, nz_normed=None, ra_min=None, ra_max=None, dec_min=None, dec_max=None, z_min=None, z_max=None):
        if ra_min is None:
            ra_min = np.min(self.datapoint_ra)

        if dec_min is None:
            dec_min = np.min(self.datapoint_dec)

        if ra_max is None:
            ra_max = np.max(self.datapoint_ra)

        if dec_max is None:
            dec_max = np.max(self.datapoint_dec)

        if z_min is None:
            z_min = np.min(self.datapoint_z)

        if z_max is None:
            z_max = np.max(self.datapoint_z)

        n_rand = int(len(self.datapoint_ra) * nrand_fac)
        rand_ra, rand_dec = esutil.coords.randsphere(n_rand, ra_range=[ra_min, ra_max], dec_range=[dec_min, dec_max])
        if (zarray is None) and (nz_normed is None):
            nzbins_total = 100
            zarray_all = np.linspace(z_min, z_max, nzbins_total)
            zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
            zarray = zarray_all[1:-1]
            nz_unnorm, z_edge = np.histogram(self.datapoint_z, zarray_edges)
            nz_normed = nz_unnorm / (integrate.simps(nz_unnorm, zarray))
        gen = esutil.random.Generator(nz_normed, zarray)
        rand_z = gen.sample(n_rand)
        return rand_ra, rand_dec, rand_z



    def create_random_cat_uniform(self, zmin, zmax, nzbins_total = 20000):

        theta_datapoint, phi_datapoint = eq2ang(self.datapoint_ra, self.datapoint_dec)
        costheta_datapoint = np.cos(theta_datapoint)

        phi_min = np.amin(phi_datapoint)
        phi_max = np.amax(phi_datapoint)
        costheta_min = np.amin(costheta_datapoint)
        costheta_max = np.amax(costheta_datapoint)

        zarray_all = np.linspace(zmin, zmax, nzbins_total)
        zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
        zarray = zarray_all[1:-1]

        rand_ra, rand_dec, rand_z = np.array([]), np.array([]), np.array([])
        for j in range(len(zarray)):
            zmin_jh = zarray_edges[j]
            zmax_jh = zarray_edges[j + 1]
            ngal_in_zh = max(len(np.where((self.datapoint_z > zmin_jh) & (self.datapoint_z < zmax_jh))[0]), 5)

            nrand = 10 * ngal_in_zh
            rand_phi_j = phi_min + (np.random.rand(nrand)) * (phi_max - phi_min)
            rand_theta_j = np.arccos(costheta_min + (np.random.rand(nrand)) * (costheta_max - costheta_min))
            rand_ra_j, rand_dec_j = ang2eq(rand_theta_j, rand_phi_j)

            if len(rand_ra) == 0:
                rand_ra = rand_ra_j
                rand_dec = rand_dec_j
                rand_z = zmin_jh + (zmax_jh - zmin_jh) * np.random.random(size=len(rand_ra_j))
                # rand_z = (zarray[j])*(np.ones(len(rand_ra_j)))
            else:
                rand_ra = np.hstack((rand_ra, rand_ra_j))
                rand_dec = np.hstack((rand_dec, rand_dec_j))
                rand_z = np.hstack((rand_z, zmin_jh + (zmax_jh - zmin_jh) * np.random.random(size=len(rand_ra_j))))
                # rand_z = np.hstack((rand_z,(zarray[j])*(np.ones(len(rand_ra_j)))))

        return rand_ra, rand_dec, rand_z

    def create_random_cat_masked(self, zmin, zmax, ind_datapoint_mask, nside_mask, nzbins_total = 20000):

        theta_datapoint, phi_datapoint = eq2ang(self.datapoint_ra, self.datapoint_dec)
        costheta_datapoint = np.cos(theta_datapoint)

        phi_min = np.amin(phi_datapoint)
        phi_max = np.amax(phi_datapoint)
        costheta_min = np.amin(costheta_datapoint)
        costheta_max = np.amax(costheta_datapoint)

        

        zarray_all = np.linspace(zmin, zmax, nzbins_total)
        zarray_edges = (zarray_all[1:] + zarray_all[:-1]) / 2.
        zarray = zarray_all[1:-1]

        rand_ra, rand_dec, rand_z = np.array([]), np.array([]), np.array([])
        for j in range(len(zarray)):
            zmin_jh = zarray_edges[j]
            zmax_jh = zarray_edges[j + 1]
            ngal_in_zh = max(len(np.where((self.datapoint_z > zmin_jh) & (self.datapoint_z < zmax_jh))[0]), 5)

            nrand = 20 * ngal_in_zh
            rand_phi_j = phi_min + (np.random.rand(nrand)) * (phi_max - phi_min)
            rand_theta_j = np.arccos(costheta_min + (np.random.rand(nrand)) * (costheta_max - costheta_min))
            rand_ra_j, rand_dec_j = ang2eq(rand_theta_j, rand_phi_j)

            if len(rand_ra) == 0:
                rand_ra = rand_ra_j
                rand_dec = rand_dec_j
                rand_z = zmin_jh + (zmax_jh - zmin_jh) * np.random.random(size=len(rand_ra_j))
                # rand_z = (zarray[j])*(np.ones(len(rand_ra_j)))
            else:
                rand_ra = np.hstack((rand_ra, rand_ra_j))
                rand_dec = np.hstack((rand_dec, rand_dec_j))
                rand_z = np.hstack((rand_z, zmin_jh + (zmax_jh - zmin_jh) * np.random.random(size=len(rand_ra_j))))
                # rand_z = np.hstack((rand_z,(zarray[j])*(np.ones(len(rand_ra_j)))))
        print('getting indices in the masked area')
        theta_rand, phi_rand = eq2ang(rand_ra, rand_dec)
        ind_rand_all = hp.ang2pix(nside_mask, theta_rand, phi_rand)
        ind_isin_mask = np.in1d(ind_rand_all, ind_datapoint_mask)

        rand_ra_masked, rand_dec_masked, rand_z_masked = rand_ra[ind_isin_mask], rand_dec[ind_isin_mask], rand_z[
            ind_isin_mask]

        make_plots = True
        if make_plots:
            theta_m, phi_m = eq2ang(rand_ra_masked, rand_dec_masked)
            ind_m_f = hp.ang2pix(nside_mask, theta_m, phi_m)
            mask_m = np.zeros(hp.nside2npix(nside_mask))
            mask_m[ind_m_f] = 1
            plt.figure()
            hp.mollview(mask_m)
            plt.savefig(
                '/global/project/projectdirs/m1727/shivamp_lsst/data_set/dc2_v1.0/measurements/' + 'rand_m_sky.png')
            plt.close()

            mask_d = np.zeros(hp.nside2npix(nside_mask))
            mask_d[ind_datapoint_mask] = 1
            plt.figure()
            hp.mollview(mask_d)
            plt.savefig(
                '/global/project/projectdirs/m1727/shivamp_lsst/data_set/dc2_v1.0/measurements/' + 'data_m_sky.png')
            plt.close()
            # pdb.set_trace()
        return rand_ra_masked, rand_dec_masked, rand_z_masked

    def get_int(self, chi_max, chi_min):
        chi_array = np.linspace(chi_min, chi_max, 5000)
        int_total = sp.integrate.simps(chi_array ** 2, chi_array)
        return int_total

    def get_diff(self, chi_max, chi_min, int_val):
        return int_val - self.get_int(chi_max, chi_min)

    def root_find(self, init_x, chi_min, int_val):
        nll = lambda *args: self.get_diff(*args)
        args = (chi_min, int_val)
        result = op.root(nll, np.array([init_x]), args=args, options={'maxfev': 50}, tol=0.01)
        return result.x[0]

    def get_chimax_from_int(self, chi_min, int_val):
        valf = self.root_find(2. * chi_min, chi_min, int_val)
        return valf

    def get_jk_stats(self):

        bin_n_all = np.zeros(len(self.datapoint_ra))
        jk_all = np.zeros(len(self.datapoint_ra))

        print('getting jack knife')
        for j in range(len(self.bin_array)):

            print('getting jk for bin ' + str(j + 1))

            zminh = self.zmin_bins[j]
            zmaxh = self.zmax_bins[j]

            ind_binh = np.where((self.datapoint_z > zminh) & (self.datapoint_z < zmaxh))[0]
            bin_n_all[ind_binh] = self.bin_n_array[j]

            print(len(ind_binh))

            datapoint_radec = np.transpose([self.datapoint_ra[ind_binh], self.datapoint_dec[ind_binh]])
            datapoint_jk_2d = self.jkobj_map_2d.find_nearest(datapoint_radec)

            if self.njk_z == 1:
                jk_all[ind_binh] = np.copy(datapoint_jk_2d)

            else:

                # chi_min,chi_max = self.get_chi(zminh),self.get_chi(zmaxh)
                # chi_array = np.linspace(chi_min,chi_max,2000)
                # int_total = sp.integrate.simps(chi_array**2,chi_array)
                # int_bin = int_total/self.njk_z
                # jk_z_array = []
                dchi_jk_z_array = []
                # zmin_i1,chi_min_i1 = zminh,chi_min

                jk_z_array = np.linspace(zminh, zmaxh, self.njk_z + 1)

                for i1 in range(self.njk_z):
                    # chi_max_i1 = self.get_chimax_from_int(chi_min_i1,int_bin)
                    # zmax_i1 = self.z_interp(chi_max_i1)
                    # jk_z_array.append(zmax_i1)

                    zmin_i1, zmax_i1 = jk_z_array[i1], jk_z_array[i1 + 1]

                    dchi_jk_z_array.append(self.get_chi(zmax_i1) - self.get_chi(zmin_i1))

                    ind_binh_i1 = np.where((self.datapoint_z > zmin_i1) & (self.datapoint_z < zmax_i1))[0]
                    datapoint_radec_i1 = np.transpose([self.datapoint_ra[ind_binh_i1], self.datapoint_dec[ind_binh_i1]])
                    datapoint_jk_2d_i1 = self.jkobj_map_2d.find_nearest(datapoint_radec_i1)
                    jk_all[ind_binh_i1] = np.copy(datapoint_jk_2d_i1 + i1 * self.njk_radec)

                    # zmin_i1, chi_min_i1 = np.copy(zmax_i1), np.copy(chi_max_i1)

                print('for bin:', str(j + 1), ', delta_chi:', dchi_jk_z_array, 'Mpc/h')
                print('for bin:', str(j + 1), ', z edges:', jk_z_array)

        return bin_n_all, jk_all

    def save_cat(self, bin_n_all, jk_all, save_dir, save_filename, ra_all=None, dec_all=None, z_all=None, w_all=None):

        if z_all is None:
            r_all = self.get_chi(self.datapoint_z)
        else:
            r_all = self.get_chi(z_all)

        if ra_all is None:
            c1 = fits.Column(name='RA', array=self.datapoint_ra, format='E')
        else:
            c1 = fits.Column(name='RA', array=ra_all, format='E')

        if dec_all is None:
            c2 = fits.Column(name='DEC', array=self.datapoint_dec, format='E')
        else:
            c2 = fits.Column(name='DEC', array=dec_all, format='E')


        c3 = fits.Column(name='R', array=r_all, format='E')

        if z_all is None:
            c4 = fits.Column(name='Z', array=self.datapoint_z, format='E')
        else:
            c4 = fits.Column(name='Z', array=z_all, format='E')


        c5 = fits.Column(name='BIN', format='K', array=bin_n_all)
        c6 = fits.Column(name='JK', format='K', array=jk_all)

        if w_all is None:
            w_all = np.ones_like(r_all)
        c7 = fits.Column(name='W', array=w_all, format='E')

        t = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7])
        t.writeto(save_dir + save_filename, clobber=True)
