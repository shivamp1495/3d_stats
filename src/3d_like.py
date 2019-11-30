import sys, os
from cosmosis.datablock import names, option_section
from numpy import random
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import scipy as sp
import ast
import pickle as pk
import copy
import pdb


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in xrange(0, cov.shape[0]):
        for jj in xrange(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr

def get_theory_terms(block, r_data, stat_type, bins_array):
    xi_theory_rdata = []
    r_array = block.get_double_array_1d("pk_to_xi", "r")
    if stat_type == 'gg':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gg =  block.get_double_array_1d("pk_to_xi","xi_gg_bin % s" % bin_j)
            xi_gg_temp = intspline(r_array, xi_gg)
            xi_gg_f = xi_gg_temp(r_data[j])

            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gg_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gg_f))

    elif stat_type == 'gm':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gm = block.get_double_array_1d("pk_to_xi","xi_gm_bin % s" % bin_j)
            xi_gm_temp = intspline(r_array, xi_gm)
            xi_gm_f = xi_gm_temp(r_data[j])
            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gm_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gm_f))

    elif stat_type == 'gg_gm':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gg = block.get_double_array_1d("pk_to_xi","xi_gg_bin % s" % bin_j)
            xi_gg_temp = intspline(r_array, xi_gg)
            xi_gg_f = xi_gg_temp(r_data[j])
            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gg_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gg_f))

        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gm = block.get_double_array_1d("pk_to_xi","xi_gm_bin % s" % bin_j)
            xi_gm_temp = intspline(r_array, xi_gm)
            xi_gm_f = xi_gm_temp(r_data[j + nbins])
            xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gm_f))

    elif stat_type == 'gg_mm__gm_mm':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gg_mm = block.get_double_array_1d("pk_to_xi","xi_gg_mm_bin % s" % bin_j)
            xi_gg_mm_temp = intspline(r_array, xi_gg_mm)
            xi_gg_mm_f = xi_gg_mm_temp(r_data[j])
            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gg_mm_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gg_mm_f))

        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gm_mm = block.get_double_array_1d("pk_to_xi","xi_gm_mm_bin % s" % bin_j)
            xi_gm_mm_temp = intspline(r_array, xi_gm_mm)
            xi_gm_mm_f = xi_gm_mm_temp(r_data[j + nbins])
            xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gm_mm_f))

    return xi_theory_rdata


def lnprob_func(block, r_data, xi_data_gtcut, incov_obs_comp, stat_type, bins_array):
    xi_theory_rdata = get_theory_terms(block, r_data, stat_type, bins_array)
    valf = -0.5 * np.dot(np.dot(np.transpose((xi_data_gtcut - xi_theory_rdata)), incov_obs_comp),
                         (xi_data_gtcut - xi_theory_rdata))
    return valf, xi_theory_rdata


def setuplnprob_func(scale_cut_min, scale_cut_max, r_data_array, xi_data_full, cov_obs, stat_type, bins_array, cov_diag=False,
                     no_cov_zbins_only_gg_gm=False, no_cov_zbins_all=False, no_cov_gg_gm=False):
    r_data_comp_ll = []

    selection = []
    countk = 0

    if stat_type == 'gg_gm' or stat_type == 'gg_mm__gm_mm':

        r_data_gg_all = np.array([])
        r_data_gm_all = np.array([])
        for j in range(len(bins_array)):
            if len(r_data_gg_all) == 0:
                r_data_gg_all = r_data_array[j]
                r_data_gm_all = r_data_array[j + len(bins_array)]
            else:
                r_data_gg_all = np.hstack((r_data_gg_all, r_data_array[j]))
                r_data_gm_all = np.hstack((r_data_gm_all, r_data_array[j + len(bins_array)]))

        r_data_all = np.hstack((r_data_gg_all, r_data_gm_all))

        for j in range(len(bins_array)):
            r_data_j = r_data_array[j]
            selection_j = np.where((r_data_j >= scale_cut_min[j]) & (r_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))
            r_data_comp_j = r_data_j[selection_j]
            r_data_comp_ll.append(r_data_comp_j)
            countk += len(r_data_j)

        for j in range(len(bins_array)):
            r_data_j = r_data_array[j + len(bins_array)]
            selection_j = \
                np.where((r_data_j >= scale_cut_min[j + len(bins_array)]) & (r_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            selection = np.hstack((selection, countk + selection_j))
            r_data_comp_j = r_data_j[selection_j]
            r_data_comp_ll.append(r_data_comp_j)
            countk += len(r_data_j)

    if stat_type == 'gg':

        r_data_gg_all = np.array([])
        for j in range(len(bins_array)):
            if len(r_data_gg_all) == 0:
                r_data_gg_all = r_data_array[j]
            else:
                r_data_gg_all = np.hstack((r_data_gg_all, r_data_array[j]))

        r_data_all = r_data_gg_all

        for j in range(len(bins_array)):
            r_data_j = r_data_array[j]
            selection_j = np.where((r_data_j >= scale_cut_min[j]) & (r_data_j <= scale_cut_max[j]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            r_data_comp_j = r_data_j[selection_j]
            r_data_comp_ll.append(r_data_comp_j)
            countk += len(r_data_j)

    if stat_type == 'gm':

        r_data_gm_all = np.array([])

        for j in range(len(bins_array)):
            if len(r_data_gm_all) == 0:
                r_data_gm_all = r_data_array[j + len(bins_array)]
            else:
                r_data_gm_all = np.hstack((r_data_gm_all, r_data_array[j + len(bins_array)]))

        r_data_all = r_data_gm_all

        for j in range(len(bins_array)):
            r_data_j = r_data_array[j + len(bins_array)]
            selection_j = \
                np.where((r_data_j >= scale_cut_min[j + len(bins_array)]) & (r_data_j <= scale_cut_max[j + len(bins_array)]))[0]
            if len(selection) == 0:
                selection = selection_j
            else:
                selection = np.hstack((selection, countk + selection_j))

            r_data_comp_j = r_data_j[selection_j]
            r_data_comp_ll.append(r_data_comp_j)
            countk += len(r_data_j)

    selection = np.array(selection)
    cov_obs_comp = (cov_obs[:, selection])[selection, :]

    if no_cov_zbins_only_gg_gm or no_cov_zbins_all:
        bins_n_array = np.arange(len(bins_array))
        cov_obs_comp_h = np.copy(cov_obs_comp)

        if len(bins_array) > 1:
            print 'zeroing the covariance between z bins'

            if stat_type == 'gg_gm' or stat_type == 'gg_mm__gm_mm':
                z1_0 = []
                for ji in range(len(bins_array)):
                    if len(z1_0) == 0:
                        z1_0 = bins_n_array[ji] * np.ones(len(r_data_comp_ll[ji]))
                    else:
                        z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(r_data_comp_ll[ji]))))

                z1_1 = []
                for ji in range(len(bins_array)):
                    if len(z1_1) == 0:
                        z1_1 = bins_n_array[ji] * np.ones(len(r_data_comp_ll[len(bins_array) + ji]))
                    else:
                        z1_1 = np.hstack((z1_1, bins_n_array[ji] * np.ones(len(r_data_comp_ll[len(bins_array) + ji]))))

                z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
                z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

                z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
                z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

                if no_cov_zbins_only_gg_gm:
                    z1_mat_0c = -1 * np.ones(z1_mat_0.shape)
                    z1_mat_1c = -1 * np.ones(z1_mat_1.shape)
                if no_cov_zbins_all:
                    z1_mat_0c = z1_mat_0
                    z1_mat_1c = z1_mat_1

                z1_mat2 = np.concatenate((z1_mat_0c, z1_mat_01), axis=1)
                z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1c), axis=1)
                z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
                z2_matf = np.transpose(z1_matf)
                offdiag = np.where(z1_matf != z2_matf)
                cov_obs_comp_h[offdiag] = 0.0

            if stat_type == 'gg' or stat_type == 'gm':
                z1 = np.repeat(np.arange(len(bins_array)), len(r_data_comp_ll[0]))
                z1_mat = np.tile(z1, (len(bins_array) * len(r_data_comp_ll[0]), 1)).transpose()
                z2_mat = np.transpose(z1_mat)
                offdiag = np.where(z1_mat != z2_mat)
                cov_obs_comp_h[offdiag] = 0.0

        cov_obs_comp = cov_obs_comp_h

    if no_cov_gg_gm:
        bins_n_array = np.arange(len(bins_array)) + 1
        cov_obs_comp_hf = np.copy(cov_obs_comp)
        print 'zeroing the covariance between gg and gm'
        if stat_type == 'gg_gm' or stat_type == 'gg_mm__gm_mm':
            z1_0 = []
            for ji in range(len(bins_array)):
                if len(z1_0) == 0:
                    z1_0 = bins_n_array[ji] * np.ones(len(r_data_comp_ll[ji]))
                else:
                    z1_0 = np.hstack((z1_0, bins_n_array[ji] * np.ones(len(r_data_comp_ll[ji]))))

            z1_1 = []
            for ji in range(len(bins_array)):
                if len(z1_1) == 0:
                    z1_1 = -1. * bins_n_array[ji] * np.ones(len(r_data_comp_ll[len(bins_array) + ji]))
                else:
                    z1_1 = np.hstack((z1_1, -1. * bins_n_array[ji] * np.ones(len(r_data_comp_ll[len(bins_array) + ji]))))

            z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
            z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

            z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
            z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

            z1_mat2 = np.concatenate((z1_mat_0, z1_mat_01), axis=1)
            z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1), axis=1)
            z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
            z2_matf = np.transpose(z1_matf)
            offdiag = np.where(z1_matf != z2_matf)
            cov_obs_comp_hf[offdiag] = 0.0


        cov_obs_comp = cov_obs_comp_hf

    if cov_diag:
        cov_obs_comp = np.diag(np.diag(cov_obs_comp))

    incov_obs_comp = np.linalg.inv(cov_obs_comp)
    r_data_comp = r_data_all[selection]
    xi_data_gtcut = xi_data_full[selection]

    return xi_data_gtcut, r_data_comp, r_data_comp_ll, incov_obs_comp, cov_obs_comp

def import_data(r_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all, stat_type):
    if len(bins_to_rem) > 0:
        cov_obs_rm = np.ones(cov_obs.shape)
        cov_obs_copy = np.copy(cov_obs)

        z1_0 = []
        for ji in range(len(bins_all)):
            if len(z1_0) == 0:
                z1_0 = bins_all[ji] * np.ones(len(r_obs[ji]))
            else:
                z1_0 = np.hstack((z1_0, bins_all[ji] * np.ones(len(r_obs[ji]))))

        z1_1 = []
        for ji in range(len(bins_all)):
            if len(z1_1) == 0:
                z1_1 = bins_all[ji] * np.ones(len(r_obs[len(bins_all) + ji]))
            else:
                z1_1 = np.hstack((z1_1, bins_all[ji] * np.ones(len(r_obs[len(bins_all) + ji]))))

        z1_mat_0 = np.tile(z1_0, (len(z1_0), 1)).transpose()
        z1_mat_1 = np.tile(z1_1, (len(z1_1), 1)).transpose()

        z1_mat_01 = np.tile(z1_0, (len(z1_1), 1)).transpose()
        z1_mat_10 = np.tile(z1_1, (len(z1_0), 1)).transpose()

        z1_mat2 = np.concatenate((z1_mat_0, z1_mat_01), axis=1)
        z1_mat22 = np.concatenate((z1_mat_10, z1_mat_1), axis=1)
        z1_matf = np.concatenate((z1_mat2, z1_mat22), axis=0)
        z2_matf = np.transpose(z1_matf)
        ind_to_select = np.ones(z1_matf.shape)

        z1f = np.concatenate((z1_0, z1_1))

        ind_to_select_robs = []

        for bins in bins_to_rem:
            ax1_ind = np.where(z1_matf == bins)
            ax2_ind = np.where(z2_matf == bins)
            ax1_ind_robs = np.where(z1f == bins)[0]
            ind_to_select_robs.append(ax1_ind_robs)
            ind_to_select[ax1_ind] = 0
            ind_to_select[ax2_ind] = 0

        del_indf = (np.array(ind_to_select_robs)).flatten()

        ind_rm_f = np.where(ind_to_select == 0)
        cov_obs_rm[ind_rm_f] = 0
        non_zero_ind = np.nonzero(cov_obs_rm)

        newcovd = np.count_nonzero(cov_obs_rm[non_zero_ind[0][0], :])
        cov_obs_new = np.zeros((newcovd, newcovd))
        k = 0

        for j in range(len(cov_obs_rm[0, :])):
            cov_rm_j = cov_obs_rm[j, :]
            cov_obs_j = cov_obs_copy[j, :]
            nnzero_cov_obs_j = np.nonzero(cov_rm_j)
            if len(nnzero_cov_obs_j[0]) > 0:
                cov_obs_new[k, :] = cov_obs_j[nnzero_cov_obs_j]
                k += 1

        data_obs_new = np.delete(data_obs, del_indf)

        r_obs_new = []
        for bins in bins_to_fit:
            r_obs_new.append(r_obs[bins - 1])

        for bins in bins_to_fit:
            r_obs_new.append(r_obs[len(bins_all) + bins - 1])

    else:
        cov_obs_new = np.copy(cov_obs)
        data_obs_new = np.copy(data_obs)
        r_obs_new = np.copy(r_obs)

    if stat_type == 'gg':
        data_obs_new, cov_obs_new = data_obs_new[0:len(bins_to_fit) * len(r_obs[0])], cov_obs_new[
                                                                                      0:len(bins_to_fit) * len(
                                                                                          r_obs[0]),
                                                                                      0:len(bins_to_fit) * len(
                                                                                          r_obs[0])]

    if stat_type == 'gm':
        data_obs_new, cov_obs_new = data_obs_new[len(bins_to_fit) * len(r_obs[0]):len(data_obs_new)], cov_obs_new[
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          r_obs[0]):len(
                                                                                                          data_obs_new),
                                                                                                      len(
                                                                                                          bins_to_fit) * len(
                                                                                                          r_obs[0]):len(
                                                                                                          data_obs_new)]

    return r_obs_new, data_obs_new, cov_obs_new


def setup(options):
    bins_all = ast.literal_eval(options.get_string(option_section, "bins_all", "[1, 2, 3, 4, 5]"))
    bins_to_fit = ast.literal_eval(options.get_string(option_section, "bins_to_fit", "[1, 2, 3, 4, 5]"))
    rcomp_min = ast.literal_eval(options.get_string(option_section, "rcomp_min", "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]"))
    rcomp_max = ast.literal_eval(
        options.get_string(option_section, "rcomp_max", "[100, 100, 100, 100, 100, 100, 100, 100, 100, 100]"))
    cov_diag = options.get_bool(option_section, "cov_diag", False)
    no_cov_zbins_only_gg_gm = options.get_bool(option_section, "no_cov_zbins_only_gg_gm", False)
    no_cov_zbins_all = options.get_bool(option_section, "no_cov_zbins_all", False)
    no_cov_gg_gm = options.get_bool(option_section, "no_cov_gg_gm", False)
    stat_type = options.get_string(option_section, "stat_type", 'gg_gm')

    bins_to_rem = copy.deepcopy(bins_all)
    for bins in bins_to_fit:
        bins_to_rem.remove(bins)

    filename = options.get_string(option_section, "2PT_FILE")
    data = pk.load(open(filename, 'rb'))

    r_obs, data_obs, cov_obs = data['sep'], data['mean'], data['cov']

    r_obs_new, data_obs_new, cov_obs_new = import_data(r_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all,
                                                       stat_type)

    data_obs_comp, r_obs_comp, r_obs_comp_ll, incov_obs_comp, cov_obs_comp = setuplnprob_func(rcomp_min, rcomp_max,
                                                                                              r_obs_new, data_obs_new,
                                                                                              cov_obs_new, stat_type,
                                                                                              bins_to_fit,
                                                                                              cov_diag=cov_diag,
                                                                                              no_cov_zbins_only_gg_gm=no_cov_zbins_only_gg_gm,
                                                                                              no_cov_zbins_all=no_cov_zbins_all,
                                                                                              no_cov_gg_gm=no_cov_gg_gm)

    return data_obs_comp, r_obs_comp, r_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit


def execute(block, config):
    data_obs_comp, r_obs_comp, r_obs_comp_ll, incov_obs_comp, cov_obs_comp, stat_type, bins_to_fit = config
    like3d, xi_theory_rdata = lnprob_func(block, r_obs_comp_ll, data_obs_comp, incov_obs_comp, stat_type, bins_to_fit)
    chi2 = -2. * like3d

    likes = names.likelihoods
    block[likes, '3D_LIKE'] = like3d
    block[likes, '3D_CHI2'] = chi2
    block[likes, 'cov_obs_comp'] = cov_obs_comp
    block[likes, 'incov_obs_comp'] = incov_obs_comp
    block[likes, 'xi_theory_rdata'] = xi_theory_rdata
    block[likes, 'xi_data_gtcut'] = data_obs_comp

    block["data_vector", '3d_inverse_covariance'] = incov_obs_comp
    block["data_vector", '3d_theory'] = xi_theory_rdata

    return 0


def cleanup(config):
    pass
