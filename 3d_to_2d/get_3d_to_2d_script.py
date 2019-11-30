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
import notebook_calc_3d_to_2d_new as nc
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.25, Tcmb0=2.725, Ob0=0.0448)
h = 0.7
oneMpc_h = (((10 ** 6) / h) * (u.pc).to(u.m))
import copy

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

def get_zmean(zcent, nz_bin):
    prob_zcent = nz_bin
    delz = zcent[1] - zcent[0]
    zmean = (np.sum(prob_zcent * zcent * delz)) / (np.sum(prob_zcent * delz))
    return zmean

def get_wprp_from_xi(rp,r_array,xi_array):
    num = r_array * xi_array
    denom = np.sqrt(r_array**2 - rp**2)
    toint = num/denom
    val = 2.*sp.integrate.simps(toint,r_array)
    return val


def get_wtheta_from_xi(theta, r_array, xi_mat, z_array, ng, chi_array, dchi_dz):
    rp_array = chi_array * theta

    r_mat = np.tile(r_array.reshape(1, len(r_array)), (len(z_array), 1))
    rp_mat = np.tile(rp_array.reshape(len(z_array), 1), (1, len(r_array)))
    invdenom1 = 1. / (r_mat ** 2 - rp_mat ** 2)
    ind = np.where(invdenom1 <= 0)
    invdenom1[ind] = 0.0
    integrand = r_mat * xi_mat * (np.sqrt(invdenom1))
    wprp_array = 2. * sp.integrate.simps(integrand, r_array)

    toint = ng ** 2 * wprp_array / dchi_dz
    val = sp.integrate.simps(toint, z_array)

    return val


def get_wtheta_from_Pk(theta, k, Pk, z_array, ng, chi_array, dchi_dz):
    rp_array = chi_array * theta
    k_mat = np.tile(k.reshape(1, len(k)), (len(z_array), 1))
    wprp_array = np.zeros(len(rp_array))

    rp_mat = np.tile(rp_array.reshape(len(z_array), 1), (1, len(k)))
    J0_mat = sp.special.jv(0, k_mat * rp_mat)
    wprp_array = (sp.integrate.simps(k_mat * Pk * J0_mat, k)) / (2 * np.pi)

    toint = ng ** 2 * wprp_array / dchi_dz
    val = sp.integrate.simps(toint, z_array)

    return val

z_array = nc.z_array
chi_array = get_Dcom_array(z_array, cosmo.Om0)
DA_array = chi_array / (1. + z_array)
dchi_dz_array = (const.c.to(u.km / u.s)).value / (get_Hz(z_array, cosmo.Om0))
# rhom_z = cosmo.Om0 * ((1 + z_array)**3) * (cosmo.critical_density0.to(u.kg/u.m**3)).value

bin_lens = nc.bins_to_fit[0]
bin_source = nc.bin_source

df = fits.open('twopt_3d_to_2d_MICE.fits')
df_zmid = df['nz_pos_zspec'].data['Z_MID']
df_bin = df['nz_pos_zspec'].data['BIN'+ str(bin_lens)]

# ng_lensz, nm_lensz,z_lensz =  nc.get_nz_lens()
ng_lensz,z_lensz =   df_bin, df_zmid
z_lensz_pz, ng_lensz_pz = nc.get_nz_lens_2pt_pz()
z_lensz_specz, ng_lensz_specz = nc.get_nz_lens_2pt_specz()

df_zmid_s = df['nz_shear_true'].data['Z_MID']
df_bin_s = df['nz_shear_true'].data['BIN'+ str(bin_source)]
z_sourcez, ng_sourcez = df_zmid_s, df_bin_s

ng_interp = interpolate.interp1d(z_lensz, np.log(ng_lensz + 1e-40), fill_value='extrapolate')
ng_array_lens = np.exp(ng_interp(z_array))

ng_interp = interpolate.interp1d(z_lensz_pz, np.log(ng_lensz_pz + 1e-40), fill_value='extrapolate')
ng_array_lens_pz = np.exp(ng_interp(z_array))

ng_interp = interpolate.interp1d(z_lensz_pz, np.log(ng_lensz_specz + 1e-40), fill_value='extrapolate')
ng_array_lens_specz = np.exp(ng_interp(z_array))

ng_interp = interpolate.interp1d(z_sourcez, np.log(ng_sourcez + 1e-40), fill_value='extrapolate')
ng_array_source = np.exp(ng_interp(z_array))

zmean_bin = get_zmean(z_array, ng_array_lens)
zmean_ind = np.where(z_array > zmean_bin)[0][0]
print zmean_bin, z_array[zmean_ind]


# Calculate w(theta)
theta_arcmin = np.logspace(np.log10(2.5), np.log10(250), 20)
theta_rad = theta_arcmin * (1. / 60.) * (np.pi / 180.)

# r_array_hres = np.logspace(np.log10(np.min(nc.r_array)),3,5000)
r_array_hres = np.logspace(-2.8, 3.0, 2000)

xi_hres_th = np.zeros((len(z_array), len(r_array_hres)))
xi_hres_data = np.zeros((len(z_array), len(r_array_hres)))
xi_data_interp = interpolate.interp1d(np.log10(nc.r_obs_new[0]), np.log10((nc.data_obs_new[0:20])),
                                      fill_value='extrapolate')
xi_data_nc = 10 ** (xi_data_interp(np.log10(nc.r_array)))
xi_data = (np.tile((xi_data_nc), (len(z_array), 1))) * nc.xi_mm

for j in range(len(z_array)):
    xi_interp = interpolate.interp1d(np.log10(nc.r_array), (nc.xi_gg[j, :]), fill_value='extrapolate')
    xi_hres_th[j, :] = (xi_interp(np.log10(r_array_hres)))

    xi_interp = interpolate.interp1d(np.log10(nc.r_array), (xi_data[j, :]), fill_value='extrapolate')
    xi_hres_data[j, :] = (xi_interp(np.log10(r_array_hres)))

wtheta_th = np.zeros(len(theta_rad))  # bestfit theory w(theta)
# wtheta_th_pz = np.zeros(len(theta_rad)) #bestfit theory w(theta)
wtheta_data = np.zeros(len(theta_rad))  # data w(theta)
wtheta_th_pk = np.zeros(len(theta_rad))

for j in range(len(theta_rad)):
    print j
    wtheta_th[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_th, z_array, ng_array_lens, chi_array,
                                      dchi_dz_array)
    #     wtheta_th_pz[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_th, z_array, ng_array_lens_pz, chi_array, dchi_dz_array)

    wtheta_data[j] = get_wtheta_from_xi(theta_rad[j], r_array_hres, xi_hres_data, z_array, ng_array_lens, chi_array,
                                        dchi_dz_array)
#     wtheta_th_pk[j] = get_wtheta_from_Pk(theta_rad[j], nc.k_hres, nc.Pk_gg, z_array, ng_array_lens, chi_array, dchi_dz_array)


