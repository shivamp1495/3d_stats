import sys, platform, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pdb
from astropy.io import fits
from scipy import interpolate
import astropy.units as u
import pickle as pk
from astropy import constants as const
import notebook_calc_3d_to_2d as nc
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.25, Tcmb0=2.725, Ob0=0.0448)
h = 0.7
oneMpc_h = (((10 ** 6) / h) * (u.pc).to(u.m))
import copy

# Cosmology functions

# get comoving distance
def get_Dcom_array(zarray, Omega_m):
    Omega_L = 1. - Omega_m
    c = 3 * 10 ** 5
    Dcom_array = np.zeros(len(zarray))
    for j in xrange(len(zarray)):
        zf = zarray[j]
        res1 = sp.integrate.quad(lambda z: (c / 100) * (1 / (np.sqrt(Omega_L + Omega_m * ((1 + z) ** 3)))), 0, zf)
        Dcom = res1[0]
        Dcom_array[j] = Dcom
    return Dcom_array

# get 100 times dimensionless hubble constant as a function of redshift
def get_Hz(zarray, Omega_m):
    Omega_L = 1 - Omega_m
    Ez = np.sqrt(Omega_m * (1 + zarray) ** 3 + Omega_L)
    Hz = 100. * Ez
    return Hz

# get mean redshift of a bin
def get_zmean(zcent, nz_bin):
    prob_zcent = nz_bin
    delz = zcent[1] - zcent[0]
    zmean = (np.sum(prob_zcent * zcent * delz)) / (np.sum(prob_zcent * delz))
    return zmean

# get critical surface density (eq.9)
def get_sigcrit_zl_zs(zl,zs, Om):
    coeff = (const.c ** 2)/(4 * np.pi * const.G)
    chi_l = get_Dcom_array(np.array([zl]),Om)[0]
    chi_s = get_Dcom_array(np.array([zs]), Om)[0]
    DA_l = (chi_l/(1. + zl))*u.Mpc
    DA_s = (chi_s / (1. + zs))*u.Mpc
    DA_ls = ( (chi_s- chi_l) / (1. + zs)) * u.Mpc
    sigcrit =  ((coeff * (DA_s)/(DA_l * DA_ls)).to(u.kg/u.m**2)).value
    return sigcrit

# get projected correlation function (eq.2)
def get_wprp_from_xi(rp,r_array,xi_array):
    num = r_array * xi_array
    denom = np.sqrt(r_array**2 - rp**2)
    toint = num/denom
    val = 2.*sp.integrate.simps(toint,r_array)
    return val

# get w(theta) eq 3
def get_wtheta_from_xi(theta, r_array, xi_mat, z_array, ng, chi_array, dchi_dz):
    rp_array = chi_array * theta
    r_mat = np.tile(r_array.reshape(1, len(r_array)), (len(z_array), 1))
    rp_mat = np.tile(rp_array.reshape(len(z_array), 1), (1, len(r_array)))
    denom1 = (r_mat ** 2) - (rp_mat ** 2)
    ind = np.where(denom1 <= 0)
    denom1[ind] = 1e180
    integrand = r_mat * xi_mat / (np.sqrt(denom1))
    wprp_array = 2. * sp.integrate.simps(integrand, r_array)
    toint = ng ** 2 * wprp_array / dchi_dz
    val = sp.integrate.simps(toint, z_array)
    return val

# get Delta wp for gammat calculations, eq. 12 and 13
def get_Delta_wp(rp_array, r_array, xi_mat, z_array):
    r_mat = np.tile(r_array.reshape(1, len(r_array)), (len(z_array), 1))
    rp_mat = np.tile(rp_array.reshape(len(z_array), 1), (1, len(r_array)))
    denom1 = r_mat ** 2 - rp_mat ** 2
    ind = np.where(denom1 <= 0)
    denom1[ind] = 1e180
    integrand = r_mat * xi_mat / (np.sqrt(denom1))
    wprp_array = 2. * sp.integrate.simps(integrand, r_array)

    wprp_interp = interpolate.interp1d(rp_array, np.log(wprp_array), fill_value='extrapolate')

    wprp_mean = np.zeros(len(rp_array))
    for j in range(len(rp_array)):
        rp_ti = np.logspace(-2, np.log10(rp_array[j]), 50)
        wprp_ti = np.exp(wprp_interp(rp_ti))
        wprp_mean[j] = sp.integrate.simps(rp_ti * wprp_ti, rp_ti) / sp.integrate.simps(rp_ti, rp_ti)

    Delta_wp = wprp_mean - wprp_array

    return Delta_wp


# setup cosmological calculations

z_array = nc.z_array
chi_array = get_Dcom_array(z_array, cosmo.Om0)
DA_array = chi_array / (1. + z_array)
dchi_dz_array = (const.c.to(u.km / u.s)).value / (get_Hz(z_array, cosmo.Om0))
rhom_z = cosmo.Om0 * ((1 + z_array)**3) * (cosmo.critical_density0.to(u.kg/u.m**3)).value

bin_lens = nc.bins_to_fit[0]
bin_source = nc.bin_source

# get n(z) of sources and lenses

ng_lensz, nm_lensz,z_lensz =  nc.get_nz_lens()
z_lensz_pz, ng_lensz_pz = nc.get_nz_lens_2pt_pz()
z_lensz_specz, ng_lensz_specz = nc.get_nz_lens_2pt_specz()

z_sourcez, ng_sourcez = nc.get_nz_source()

ng_interp = interpolate.interp1d(z_lensz, np.log(ng_lensz + 1e-40), fill_value='extrapolate')
ng_array_lens = np.exp(ng_interp(z_array))

ng_interp = interpolate.interp1d(z_lensz_pz, np.log(ng_lensz_pz + 1e-40), fill_value='extrapolate')
ng_array_lens_pz = np.exp(ng_interp(z_array))

ng_interp = interpolate.interp1d(z_lensz_pz, np.log(ng_lensz_specz + 1e-40), fill_value='extrapolate')
ng_array_lens_specz = np.exp(ng_interp(z_array))

nm_interp = interpolate.interp1d(z_lensz, np.log(nm_lensz + 1e-40), fill_value='extrapolate')
nm_array_lens = np.exp(nm_interp(z_array))


ng_interp = interpolate.interp1d(z_sourcez, np.log(ng_sourcez + 1e-40), fill_value='extrapolate')
ng_array_source = np.exp(ng_interp(z_array))

zmean_bin = get_zmean(z_array, ng_array_lens)
zmean_ind = np.where(z_array > zmean_bin)[0][0]

# Calculate w(theta)

theta_arcmin = np.logspace(np.log10(2.5), np.log10(100), 25)
theta_rad = theta_arcmin * (1. / 60.) * (np.pi / 180.)

r_array_hres = np.logspace(-2, 2, 10000)

xi_hres_th = np.zeros((len(z_array), len(r_array_hres)))
xi_hres_data = np.zeros((len(z_array), len(r_array_hres)))
xi_data = (np.tile((nc.data_obs_new[0:20]), (len(z_array), 1))) * nc.xi_mm

for j in range(len(z_array)):
    xi_interp = interpolate.interp1d(np.log10(nc.r_obs_new[0]), np.log10(nc.xi_gg[j, :]), fill_value='extrapolate')
    xi_hres_th[j, :] = 10 ** (xi_interp(np.log10(r_array_hres)))

    xi_interp = interpolate.interp1d(np.log10(nc.r_obs_new[0]), np.log10(xi_data[j, :]), fill_value='extrapolate')
    xi_hres_data[j, :] = 10 ** (xi_interp(np.log10(r_array_hres)))

wtheta_th = np.zeros(len(theta_rad))  # bestfit theory w(theta)
wtheta_th_pz = np.zeros(len(theta_rad))  # bestfit theory w(theta)
wtheta_data = np.zeros(len(theta_rad))  # data w(theta)

for j in range(len(theta_rad)):
    wtheta_th[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_th, z_array, ng_array_lens, chi_array,
                                      dchi_dz_array)
    wtheta_th_pz[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_th, z_array, ng_array_lens_pz, chi_array,
                                         dchi_dz_array)
    wtheta_data[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_data, z_array, ng_array_lens, chi_array,
                                        dchi_dz_array)

# Calculate Sigma_crit
# when lens redshift > source redshift, set sigma_crit to high value so that gamma_t is zero
coeff_sigcrit = ((const.c ** 2)/(4 * np.pi * const.G * (1.0 * u.Mpc/h))).to(u.kg/u.m**2).value
z_lmat = np.tile(z_array.reshape(len(z_array),1), (1,len(z_array)) )
z_smat = np.tile(z_array.reshape(1,len(z_array)), (len(z_array),1) )

chi_lmat = np.tile(chi_array.reshape(len(z_array),1), (1,len(z_array)) )
chi_smat = np.tile(chi_array.reshape(1,len(z_array)), (len(z_array),1) )
DA_l = (chi_lmat/(1. + z_lmat))
DA_s = (chi_smat / (1. + z_smat))
DA_ls = ( (chi_smat- chi_lmat) / (1. + z_smat))
sig_crit_mat =  (coeff_sigcrit * DA_s/(DA_l * DA_ls))
ind_lz = np.where(DA_ls <= 0)
sig_crit_mat[ind_lz] = 1e180


# Do the integral over the source redshift, last integral in Eq.16

ng_array_source_rep = np.tile(ng_array_source.reshape(1,len(z_array)), (len(z_array), 1))
int_sourcez = sp.integrate.simps(ng_array_source_rep / sig_crit_mat, z_array)

# Calculate gamma_t

r_array_hres = np.logspace(-3, 2, 1000)

xi_gm_hres_th = np.zeros((len(z_array), len(r_array_hres)))
xi_gm_hres_data = np.zeros((len(z_array), len(r_array_hres)))
xi_gm_data = (np.tile((nc.data_obs_new[20:40]), (len(z_array), 1))) * nc.xi_mm

for j in range(len(z_array)):
    xi_interp = interpolate.interp1d(np.log10(nc.r_obs_new[1]), np.log10(nc.xi_gm[j, :]), fill_value='extrapolate')
    xi_gm_hres_th[j, :] = 10 ** (xi_interp(np.log10(r_array_hres)))

    xi_interp = interpolate.interp1d(np.log10(nc.r_obs_new[1]), np.log10(xi_gm_data[j, :]), fill_value='extrapolate')
    xi_gm_hres_data[j, :] = 10 ** (xi_interp(np.log10(r_array_hres)))

gtheta_data = np.zeros(len(theta_rad))  # bestfit theory gamma_t
gtheta_th = np.zeros(len(theta_rad))  # data gamma_t
gtheta_th_pz = np.zeros(len(theta_rad))

for j1 in range(len(theta_rad)):
    rp_array = chi_array * theta_rad[j1]

    Deltawp_data = get_Delta_wp(rp_array, r_array_hres, xi_gm_hres_data, z_array)
    Deltawp_th = get_Delta_wp(rp_array, r_array_hres, xi_gm_hres_th, z_array)

    gtheta_data[j1] = sp.integrate.simps(rhom_z * ng_array_lens * int_sourcez * Deltawp_data * oneMpc_h, z_array)
    gtheta_th[j1] = sp.integrate.simps(rhom_z * ng_array_lens * int_sourcez * Deltawp_th * oneMpc_h, z_array)
    gtheta_th_pz[j1] = sp.integrate.simps(rhom_z * ng_array_lens_pz * int_sourcez * Deltawp_th * oneMpc_h, z_array)







