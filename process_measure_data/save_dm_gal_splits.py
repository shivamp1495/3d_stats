import sys, platform, os
sys.path.insert(0,'/global/u1/s/spandey/kmeans_radec/')
import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.signal as spsg
import matplotlib.pyplot as plt
import pdb
import healpy as hp
from astropy.io import fits
from kmeans_radec import KMeans, kmeans_sample
import time
import math
from scipy import interpolate
import treecorr
import pickle as pk
import configparser
import ast
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.constants as const
import kmeans_radec
import h5py as h5
import argparse
import gc
sys.path.insert(0,'/global/cfs/cdirs/des/shivamp/cosmosis/y3kp-bias-model/3d_stats/process_measure_data/')
import process_cats_class as pcc
cosmo_params_dict = {'flat': True, 'H0': 70.0, 'Om0': 0.25, 'Ob0': 0.044, 'sigma8': 0.8, 'ns': 0.95}
from nbodykit.lab import HDFCatalog
def load_mesh_h5(cat_str, N, box_size):
    f =  HDFCatalog(cat_str)
    f.attrs['BoxSize'] = box_size
    return  f.to_mesh(Nmesh=N, compensated=True)


def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec


def eq2ang(ra, dec):
    phi = ra * np.pi / 180.
    theta = (np.pi / 2.) - dec * (np.pi / 180.)
    return theta, phi

def get_zmean(zcent,delz,nz_bin):
    prob_zcent = nz_bin
    zmean = (np.sum(prob_zcent*zcent*delz))/(np.sum(prob_zcent*delz))
    return zmean


box_size = 4225.35211 
box_size_h = box_size*0.71
N = 3000

# dm_str = '/global/cscratch1/sd/samgolds/gal_cat_24_5.h5'   
dm_str = '/global/cscratch1/sd/samgolds/dm_cat.h5'   

mesh_dm = load_mesh_h5(dm_str, N, box_size)

mesh_dm_real = mesh_dm.to_real_field()

nsp = 10
nbox = int(N/nsp)
for j1 in range(nsp):
	for j2 in range(nsp):
		for j3 in range(nsp):
			pk.dump(mesh_dm_real[j1*nbox:(j1+1)*nbox,j2*nbox:(j2+1)*nbox,j3*nbox:(j3+1)*nbox],open('/global/project/projectdirs/m1727/shivamp_lsst/data_set/dm_mesh/mesh_dm_real_Ng3000_fullbox_nsp' + str(nsp) + '_' + str(j1) + '_' + str(j2) + '_' + str(j3) + '.pk', 'wb'))


