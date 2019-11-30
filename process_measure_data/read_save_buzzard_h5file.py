import h5py as h5
from astropy.io import fits
import numpy as np
import healpy as hp

def ang2eq(theta, phi):
    ra = phi * 180. / np.pi
    dec = 90. - theta * 180. / np.pi
    return ra, dec

fname = '/project/projectdirs/des/jderose/Chinchilla/Herd/Chinchilla-3/v1.9.8/sampleselection/Y3a/mastercat/Buzzard-3_v1.9.8_Y3a_mastercat.h5'
cat = h5.File(fname, 'r')
mcat = cat['catalog/downsampled_dm']

px = mcat['px'][()]
ntotal = len(px)
dm = 2.8

ind_sel = np.unique(np.random.randint(0,ntotal, int(ntotal/dm)))

px, py, pz = mcat['px'][()][ind_sel],mcat['py'][()][ind_sel],mcat['pz'][()][ind_sel]

z_red = mcat['z_cos'][()][ind_sel]
print('min z=' + str(np.min(z_red)) + ', max z=' + str(np.max(z_red)))

dm_tosave = np.around(len(px)*1.0/ntotal,3)
print('total downsampling:' + str(dm_tosave))

theta, phi = hp.vec2ang(np.array([px,py,pz]))
ra, dec = ang2eq(theta, phi)


c1 = fits.Column(name='RA', array=ra, format='E')
c2 = fits.Column(name='DEC', array=dec, format='E')
c3 = fits.Column(name='Z', array=z_red, format='E')


t = fits.BinTableHDU.from_columns([c1, c2, c3])
save_dir = '/global/project/projectdirs/des/shivamp/actxdes/data_set/buzzard_sims/'
save_filename = 'downsampled_matter_catalog_dm_from_orig_' + str(dm_tosave) + '.fits'
t.writeto(save_dir + save_filename, clobber=True)




