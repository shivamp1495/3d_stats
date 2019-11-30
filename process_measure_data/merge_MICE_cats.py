import sys, os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import treecorr
from astropy.io import fits
import pickle as pk
import pdb

basedir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/matter1in100/'

fext = np.linspace(0,2442,2443)
# fext = np.linspace(0,2442,10)

ra, dec, z_true = [], [], []
for j in range(len(fext)):
    if np.mod(j, 10) == 0:
        print j
    filename = basedir + 'lightcone_129_1in100.' + str(int(fext[j]))
    data = np.loadtxt(filename)

    if len(data) > 0:
        nobj = data.shape[0]/3
        ind = np.random.randint(0, nobj, data.shape[0])
        if len(ra) == 0:
            ra = data[ind, 0]
            dec = data[ind, 1]
            z_true = data[ind, 2]
        else:
            ra = np.hstack((ra,data[ind, 0]))
            dec = np.hstack((dec,data[ind, 1]))
            z_true = np.hstack((z_true,data[ind, 2]))

# pdb.set_trace()

c1 = fits.Column(name='RA', array=np.array(ra), format='E')
c2 = fits.Column(name='DEC', array=np.array(dec), format='E')
c3 = fits.Column(name='Z', array=np.array(z_true), format='E')

t = fits.BinTableHDU.from_columns([c1, c2, c3])
t.writeto( '/global/project/projectdirs/des/y3-bias/MICE_all_data/v2/matter_ra_dec_z_L3072N4096-LC129-1in300.fits', clobber=True)








