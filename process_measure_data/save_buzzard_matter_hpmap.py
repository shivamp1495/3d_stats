import h5py as h5
from astropy.io import fits
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec

fname = '/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.8/sampleselection/Y3a/mastercat/Buzzard-3_v1.9.8_Y3a_mastercat.h5'
cat = h5.File(fname, 'r')
mcat = cat['catalog/downsampled_dm']
px, py, pz = mcat['px'][()],mcat['py'][()],mcat['pz'][()]
theta, phi = hp.vec2ang(np.array([px,py,pz]))
ra, dec = ang2eq(theta, phi)

nside_mask = 512
ind_g_f = hp.ang2pix(nside_mask, theta, phi)
mask_d = np.zeros(hp.nside2npix(nside_mask))
mask_d[ind_g_f] = 1
plt.figure()
hp.mollview(mask_d)
plt.savefig('/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/buzzard_matter_dist.png')



