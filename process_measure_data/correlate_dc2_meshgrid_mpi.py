import sys, platform, os
import numpy as np
from mpi4py import MPI
import treecorr
import time


class NNRatio(treecorr.BinnedCorr2):
    """A Correlation object where "xi" is really the ration (1+dd1)/(1+dd2)
    """
    def __init__(self, dd1, dd2):
        self.dd1 = dd1
        self.dd2 = dd2
        self._nbins = dd1._nbins
        self.npatch1 = dd1.npatch1
        self.npatch2 = dd1.npatch2
        self.results = dd1.results 

    def _calculate_xi_from_pairs(self, pairs):
        """Calculate "xi" (which is really (1+dd1)/(1+dd2)) given a list of region pairs.
        """
        xi1, w1 = self.dd1._calculate_xi_from_pairs(pairs)
        xi2, w2 = self.dd2._calculate_xi_from_pairs(pairs)
        ratio = (xi1) / (xi2)
        w = w1+w2  # This isn't actually used by the jackknife method, so doesn't much matter what we use here.
        return ratio,w

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


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cat', default='all', type=str, help='Cat type')
#     parser.add_argument('--wver', default='new', type=str, help='Cat type')
#     parser.add_argument('--njk', type=int, default=100)
#     parser.add_argument('--lowmem', type=bool, default=False)
#     parser.add_argument('--putw', type=bool, default=True)
#     parser.add_argument('--nth', type=int, default=64)
#     args_all = parser.parse_args()
#     return args_all




def save_corrf(jx,jy,jz,nsp,verbose=False):
    import pickle as pk
    import os
    import numpy as np
    # from mpi4py import MPI
    import time
    import treecorr
    import pickle as pk
    import gc
    import time    



    ti = time.time()
    do_gg = True
    minrad = 1.5
    maxrad = 70.0
    nrad = 15
    nthreads = 64
    bin_slop = 0.2

    save_dir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/dc2_v1.0/measurements/mesh_grid/nsp' + str(nsp) + '/'
    do_mm, do_gm, do_gg = True, True, True
    filename = save_dir + 'corrf_DC2_meshgrid_subsample_nsp' + str(nsp) + '_' + str(jx) + '_' + str(jy) + '_' + str(jz) + '.pk'
    if os.path.isfile(filename):
        save_data = pk.load(open(filename,'rb'))
        all_keys = list(save_data.keys())
        if ('xi_mm_full' in all_keys):
            do_mm = False
            xi_mm_full = save_data['xi_mm_full']
        if ('xi_gm_full' in all_keys):
            do_gm = False
            xi_gm_full = save_data['xi_gm_full']            
        if ('xi_gg_full' in all_keys):
            do_gg = False
            xi_gg_full = save_data['xi_gg_full']            
        if do_mm + do_gm + do_gg == 0:
            return 0
    else:
        save_data = {}

    if verbose:
        print('opening DM cat')

    ldir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/dm_mesh/'
    df = pk.load(open(ldir + 'mesh_dm_real_Ng3000_fullbox_nsp' + str(nsp) + '_' + str(jx) + '_' + str(jy) + '_' + str(jz) + '.pk','rb'))
    dm_density = np.ravel(df) - 1.
    del df
    gc.collect()

    if verbose:
        print('getting x,y,z meshgrid')
    x1d = np.linspace(300*jx,300*(jx+1),300)
    y1d = np.linspace(300*jy,300*(jy+1),300)
    z1d = np.linspace(300*jz,300*(jz+1),300)
    xm, ym, zm = np.meshgrid(x1d,y1d,z1d)

    xarr = np.ravel(xm)
    del xm
    gc.collect()
    yarr = np.ravel(ym)
    del ym
    gc.collect()
    zarr = np.ravel(zm)
    del zm
    gc.collect()

    if do_gg or do_gm:
        matter_cat = treecorr.Catalog(x=xarr, y=yarr, z=zarr, k=dm_density)               
    del dm_density
    gc.collect()
    
    if do_mm:
        t1 = time.time()
        m_m = treecorr.KKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                        num_threads=nthreads, bin_slop=bin_slop)
        if verbose:
            print('doing matterxmatter calculation')
        m_m.process(matter_cat, matter_cat)
        if verbose:
            print('this took ' + str(np.around(time.time() - t1,2)) + 's')
        xi_mm_full = m_m.xi
        r_mm = np.exp(m_m.meanlogr)
        print(r_mm)
        print(xi_mm_full)
        save_data = {
                    'xi_mm_full': xi_mm_full, 'r_mm': r_mm
                    }
        import pickle as pk
        pk.dump(save_data, open(filename, "wb"), protocol = 2)

    if verbose:
        print('opening gal cat')
    ldir = '/global/project/projectdirs/m1727/shivamp_lsst/data_set/gal_mesh_24p5/'
    df = pk.load(open(ldir + 'mesh_gal_real_Ng3000_fullbox_nsp' + str(nsp) + '_' + str(jx) + '_' + str(jy) + '_' + str(jz) + '.pk','rb'))
    gal_density = np.ravel(df) - 1.
    del df
    gc.collect()

    if do_gg or do_gm:
        gal_cat = treecorr.Catalog(x=xarr, y=yarr, z=zarr, k=gal_density)
    del gal_density, xarr, yarr, zarr
    gc.collect()

    if do_gm:
        if verbose:
            print('doing galaxyxmatter calculation')
        t1 = time.time()
        g_m = treecorr.KKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                        num_threads=nthreads, bin_slop=bin_slop) 
        g_m.process(gal_cat, matter_cat)
        print('this took ' + str(np.around(time.time() - t1,2)) + 's')
        del matter_cat
        gc.collect()
        xi_gm_full = g_m.xi
        r_gm = np.exp(g_m.meanlogr)
        if verbose:
            print(r_gm)
            print(xi_gm_full)
        save_data['xi_gm_full'] = xi_gm_full
        save_data['r_gm'] = r_gm
        import pickle as pk
        pk.dump(save_data, open(filename, "wb"), protocol = 2)


    if do_gg:
        if verbose:
            print('doing galaxyxgalaxy calculation')
        t1 = time.time()
        g_g = treecorr.KKCorrelation(nbins=nrad, min_sep=minrad, max_sep=maxrad, verbose=0,
                                        num_threads=nthreads, bin_slop=bin_slop)
        g_g.process(gal_cat, gal_cat)
        if verbose:
            print('this took ' + str(np.around(time.time() - t1,2)) + 's')
        del gal_cat
        gc.collect()
        xi_gg_full = g_g.xi
        r_gg = np.exp(g_g.meanlogr)
        if verbose:
            print(r_gg)
            print(xi_gg_full)
        save_data['xi_gg_full'] = xi_gg_full
        save_data['r_gg'] = r_gg
        import pickle as pk
        pk.dump(save_data, open(filename, "wb"), protocol = 2)

        if verbose:
            print('getting the ratios gg/mm')
        xi_gg_mm = (xi_gg_full)/(xi_mm_full)
        save_data['xi_gg_mm'] = xi_gg_mm
        if verbose:
            print(xi_gg_mm)
        import pickle as pk
        pk.dump(save_data, open(filename, "wb"), protocol = 2)

        if verbose:
            print('getting the ratios gm/mm')
        xi_gm_mm = (xi_gm_full)/(xi_mm_full)
        if verbose:
            print(xi_gm_mm)
        save_data['xi_gm_mm'] = xi_gm_mm
        import pickle as pk
        pk.dump(save_data, open(filename, "wb"), protocol = 2)

    if verbose:
        tf = time.time()
        print('Total pipeline took ' + str(np.around((tf - ti),2)) + 's')
    return 0


def run_xi_final(i_rank,nsp):
    all_ind = []
    for j1 in range(nsp):
        for j2 in range(nsp):
            for j3 in range(nsp):
                all_ind.append(np.array([j1,j2,j3]))
    all_ind = np.array(all_ind)
    ind_arr = np.arange(100*i_rank,100*(i_rank+1))
    for ji in range(len(ind_arr)):
        all_jk = all_ind[ind_arr[ji]]
        print('doing ' + str(all_jk))
        save_corrf(all_jk[0],all_jk[1],all_jk[2],nsp,verbose=False)

if __name__ == '__main__':
    run_count = 0
    n_jobs = 10
    nsp = 10
    while run_count<n_jobs:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank) < n_jobs:
            run_xi_final(comm.rank,nsp)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 


# salloc -N 10 -C haswell -q interactive -t 04:00:00 -L SCRATCH --account=m1727
# srun --nodes=10 --tasks-per-node=1 --cpus-per-task=64 --cpu-bind=cores python correlate_dc2_meshgrid_mpi.py