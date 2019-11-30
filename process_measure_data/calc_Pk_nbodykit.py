import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import healpy as hp
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
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
import nbodykit as nbk
from nbodykit.source.catalog import FITSCatalog


def eq2ang(ra, dec):
    phi = ra * np.pi / 180.
    theta = (np.pi / 2.) - dec * (np.pi / 180.)
    return theta, phi

load_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/process_cats/'
load_filename_matter = 'matter_ra_dec_r_z_bin_jk_L3072N4096-LC129-1in700_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '_ds_' + str(ds_m_inp) + '.fits'
load_filename_galaxy = 'galaxy_ra_dec_r_z_bin_jk_mice2_des_run_redmapper_v6.4.16_redmagic_njkradec_' + str(
    njk_radec) + '_njkz_' + str(njk_z) + '.fits'

fm = fits.open(load_dir + load_filename_matter)
fg = fits.open(load_dir + load_filename_galaxy)

fcat_m = FITSCatalog(load_dir + load_filename_matter)
fcat_g = FITSCatalog(load_dir + load_filename_galaxy)

theta_m, phi_m = eq2ang(fm[1].data['RA'], fm[1].data['DEC'])
theta_g, phi_g = eq2ang(fg[1].data['RA'], fg[1].data['DEC'])

R_m, R_g = fm[1].data['R'],fg[1].data['R']

X_m = R_m * np.cos(phi_m) * np.cos(theta_m)
Y_m = R_m * np.sin(phi_m) * np.cos(theta_m)
Z_m = R_m * np.sin(theta_m)

X_g = R_g * np.cos(phi_g) * np.cos(theta_g)
Y_g = R_g * np.sin(phi_g) * np.cos(theta_g)
Z_g = R_g * np.sin(theta_g)

fcat_m['Position'] = np.vstack((X_m,Y_m,Z_m)).T

fcat_g['Position'] = np.vstack((X_g,Y_g,Z_g)).T




