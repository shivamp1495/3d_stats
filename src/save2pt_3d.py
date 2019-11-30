import sys, os
from cosmosis.datablock import names, option_section
from numpy import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import scipy as sp
import ast
import pickle as pk
import copy
import pdb


def get_theory_terms(block, r_data, stat_type, bins_array):
    xi_theory_rdata = []
    r_array = block.get_double_array_1d("pk_to_xi", "r"),
    if stat_type == 'gg':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gg = block.get_double_array_1d("pk_to_xi", "xi_gg_bin % s" % bin_j)
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
            xi_gm = block.get_double_array_1d("pk_to_xi", "xi_gm_bin % s" % bin_j)
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
            xi_gg = block.get_double_array_1d("pk_to_xi", "xi_gg_bin % s" % bin_j)
            xi_gg_temp = intspline(r_array, xi_gg)
            xi_gg_f = xi_gg_temp(r_data[j])
            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gg_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gg_f))

        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gm = block.get_double_array_1d("pk_to_xi", "xi_gm_bin % s" % bin_j)
            xi_gm_temp = intspline(r_array, xi_gm)
            xi_gm_f = xi_gm_temp(r_data[j + nbins])
            xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gm_f))


    elif stat_type == 'gg_mm__gm_mm':
        nbins = len(bins_array)
        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gg_mm = block.get_double_array_1d("pk_to_xi", "xi_gg_mm_bin % s" % bin_j)
            xi_gg_mm_temp = intspline(r_array, xi_gg_mm)
            xi_gg_mm_f = xi_gg_mm_temp(r_data[j])
            if len(xi_theory_rdata) == 0:
                xi_theory_rdata = xi_gg_mm_f
            else:
                xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gg_mm_f))

        for j in range(nbins):
            bin_j = bins_array[j]
            xi_gm_mm = block.get_double_array_1d("pk_to_xi", "xi_gm_mm_bin % s" % bin_j)
            xi_gm_mm_temp = intspline(r_array, xi_gm_mm)
            xi_gm_mm_f = xi_gm_mm_temp(r_data[j + nbins])
            xi_theory_rdata = np.hstack((xi_theory_rdata, xi_gm_mm_f))

    return xi_theory_rdata


def save_2pt(block, r_data, Pk_obs_comp, cov_obs_new, stat_type, bins_to_fit, pt_type, pt_type_values, sc_save2pt,
             save2pt_dir, def_save, do_plot=True, save_plot_dir=''):
    cov_d = np.diag(cov_obs_new)

    Pk_theory_comp = get_theory_terms(block, r_data, stat_type, bins_to_fit)

    gg_dict = {}
    gm_dict = {}
    k = 0

    str_bins_to_fit = ''

    for j in range(len(r_data)):
        if j < len(bins_to_fit):
            gg_dict['obs_r_bin' + str(bins_to_fit[j])] = r_data[j]
            gg_dict['obs_val_bin' + str(bins_to_fit[j])] = Pk_obs_comp[k:k + len(r_data[j])]
            gg_dict['obs_sigma_bin' + str(bins_to_fit[j])] = np.sqrt(cov_d[k:k + len(r_data[j])])

            gg_dict['theory_r_bin' + str(bins_to_fit[j])] = r_data[j]
            gg_dict['theory_val_bin' + str(bins_to_fit[j])] = Pk_theory_comp[k:k + len(r_data[j])]

            str_bins_to_fit += str(bins_to_fit[j]) + '_'

        else:
            gm_dict['obs_r_bin' + str(bins_to_fit[j - len(bins_to_fit)])] = r_data[j]
            gm_dict['obs_val_bin' + str(bins_to_fit[j - len(bins_to_fit)])] = Pk_obs_comp[k:k + len(r_data[j])]
            gm_dict['obs_sigma_bin' + str(bins_to_fit[j - len(bins_to_fit)])] = np.sqrt(cov_d[k:k + len(r_data[j])])

            gm_dict['theory_r_bin' + str(bins_to_fit[j - len(bins_to_fit)])] = r_data[j]
            gm_dict['theory_val_bin' + str(bins_to_fit[j - len(bins_to_fit)])] = Pk_theory_comp[k:k + len(r_data[j])]

        k = k + len(r_data[j])

    final_save_dict = {'gg': gg_dict, 'gm': gm_dict}

    save_2pt_name = save2pt_dir + 'datavec_bestfit_' + stat_type + '_zbins_' + str_bins_to_fit + '_' + pt_type + '_' + pt_type_values + '_sc_' + sc_save2pt + '_' + def_save + '.pk'

    pk.dump(final_save_dict, open(save_2pt_name, "wb"))

    if do_plot:

        colors = ['r', 'b', 'k', 'orange', 'magenta','cyan', 'r', 'b', 'k', 'orange', 'magenta','cyan']
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((20, 8))
        k = 0
        for j in range(len(r_data)):

            if j < len(bins_to_fit):
                # if j == len(bins_to_fit)-1:

                # ax[1].errorbar(r_data[j],Pk_obs_comp[k:k+len(r_data[j])],np.sqrt(cov_d[k:k+len(r_data[j])]),marker='*',linestyle='',color=colors[j])
                # ax[1].plot(r_data[j],Pk_theory_comp[k:k+len(r_data[j])],linestyle='-',color=colors[j])

                ax[1].errorbar((1.0175 ** j) * r_data[j], (
                        Pk_obs_comp[k:k + len(r_data[j])] - Pk_theory_comp[k:k + len(r_data[j])]) / Pk_theory_comp[
                                                                                                    k:k + len(
                                                                                                        r_data[j])],
                               np.sqrt(cov_d[k:k + len(r_data[j])]) / Pk_theory_comp[k:k + len(r_data[j])], marker='.',
                               linestyle='', color=colors[j], label='bin ' + str(bins_to_fit[j]))

                # ax[1].savefig('/global/u1/s/spandey/cosmosis_exp/cosmosis/y3kp-bias-model/3d_stats/bestfits/gg_bestfit_' + stat_type + '_zbins_' + "_".join(bins_to_fit) + '_' + pt_type_values + '_sc_' + sc_save2pt + '.png',dpi=240)


            else:
                # elif j == 2*len(bins_to_fit)-1 :

                # ax[0].errorbar(r_data[j],Pk_obs_comp[k:k+len(r_data[j])],np.sqrt(cov_d[k:k+len(r_data[j])]),marker='*',linestyle='',color=colors[j-len(bins_to_fit)])
                # ax[0].plot(r_data[j],Pk_theory_comp[k:k+len(r_data[j])],linestyle='-',color=colors[j-len(bins_to_fit)])

                ax[0].errorbar((1.0175 ** (j - len(bins_to_fit))) * r_data[j], (
                        Pk_obs_comp[k:k + len(r_data[j])] - Pk_theory_comp[k:k + len(r_data[j])]) / Pk_theory_comp[
                                                                                                    k:k + len(
                                                                                                        r_data[j])],
                               np.sqrt(cov_d[k:k + len(r_data[j])]) / Pk_theory_comp[k:k + len(r_data[j])], marker='.',
                               linestyle='', color=colors[j - len(bins_to_fit)])

            k = k + len(r_data[j])

        ax[0].axvspan(0.0, float(sc_save2pt.split('_')[0]), facecolor='gray', alpha=0.2)
        ax[0].axvspan(40.0, 60.0, facecolor='gray', alpha=0.2)
        ax[0].set_xlim(1.5, 55.)
        ax[0].set_xscale('log')

        ax[0].set_ylim(-0.125, 0.125)

        # ax[0].set_ylim(0.01, 10)
        # ax[0].set_yscale('log')

        ax[0].set_xlabel(r'$\rm{R \ (Mpc/h)}$', fontsize=17)
        ax[0].axhline(y=0, xmin=0, xmax=100., linestyle='--')
        # ax[0].set_ylabel(r'$\xi_{gm}$')

        if stat_type == 'gg_mm__gm_mm':
            ax[0].set_ylabel(r'$\Delta \xi_{[\rm{gm/mm}]}/\xi^{th}_{[\rm{gm/mm}]}$', fontsize=20)
        else:
            ax[0].set_ylabel(r'$\Delta \xi_{\rm{gm}}/\xi^{th}_{\rm{gm}}$', fontsize=15)

        xticks = [2,4, 8, 20, 50]
        ax[0].set_xticks(xticks)
        labels = [xticks[i] for i, t in enumerate(xticks)]
        ax[0].set_xticklabels(labels)
        ax[0].tick_params(axis='both', which='major', labelsize=15)
        ax[0].tick_params(axis='both', which='minor', labelsize=15)
        # ax[0].set_ylabel(r'$\xi_{gm}$',fontsize=15)

        ax[1].axvspan(0.0, float(sc_save2pt.split('_')[0]), facecolor='gray', alpha=0.2)
        ax[1].axvspan(40.0, 60.0, facecolor='gray', alpha=0.2)
        ax[1].set_xlim(1.5, 55.)

        ax[1].set_ylim(-0.125, 0.125)

        # ax[1].set_ylim(0.01, 10)
        # ax[1].set_yscale('log')

        ax[1].set_xscale('log')

        ax[1].set_xlabel(r'$\rm{R \ (Mpc/h)}$', fontsize=17)
        ax[1].axhline(y=0, xmin=0, xmax=100., linestyle='--')
        # ax[1].set_ylabel(r'$\xi_{gg}$')
        if stat_type == 'gg_mm__gm_mm':
            ax[1].set_ylabel(r'$\Delta \xi_{[\rm{gg/mm}]}/\xi^{th}_{[\rm{gg/mm}]}$', fontsize=20)
        else:
            ax[1].set_ylabel(r'$\Delta \xi_{\rm{gg}}/\xi^{th}_{\rm{gg}}$', fontsize=15)
        # ax[1].set_ylabel(r'$\xi_{gg}$',fontsize=15)

        ax[1].legend(fontsize=16,  loc='lower right', ncol=(len(bins_to_fit)/2))

        ax[1].set_xticks(xticks)
        labels = [xticks[i] for i, t in enumerate(xticks)]
        ax[1].set_xticklabels(labels)
        ax[1].tick_params(axis='both', which='major', labelsize=15)
        ax[1].tick_params(axis='both', which='minor', labelsize=15)

        plt.tight_layout()

        plt.savefig(
            save_plot_dir + 'delta_gm_gg_bestfit_' + stat_type + '_zbins_' + str_bins_to_fit + '_' + pt_type + '_' + pt_type_values + '_sc_' + sc_save2pt + '_' + def_save + '.png')

    return 0


def save_xi_pk(block, do_regularize_pk, do_reg_all, reg_k, reg_c, pt_type, pt_type_values, save_xi_dir='',
               save_xi_def='', do_plot=True, save_plot_dir='', save_plot_def=''):
    k_hres = block.get_double_array_1d("pk_to_xi", "k")

    if pt_type in ['oneloop_eul_bk']:
        Pk_terms_names = ['Plin', 'Pmm', 'Pd1d2', 'Pd2d2', 'Pd1s2', 'Pd2s2', 'Ps2s2', 'Pd1d3nl', 'k2Pk', 'sig4']
    elif pt_type in ['oneloop_cleft_bk']:
        Pk_terms_names = ['Plin', 'Pnl1', 'Pzel', 'PA', 'PW', 'Pd1', 'Pd1d1', 'Pd2', 'Pd2d2', 'Pd1d2', 'Ps2', 'Pd1s2',
                          'Pd2s2', 'Ps2s2', 'PD2', 'Pd1D2', 'k2Pk']
    else:
        print 'No predefined pt_type given'

    Pkth_array_khres = block.get("pk_to_xi", "Pk_all")
    xi_all = block.get("pk_to_xi", "xi_all")
    znl, r_array, xi_mm = block.get_grid("pk_to_xi", "z", "r", "xi_mm_mat")

    print 'saving xi and Pk'
    for j1 in range(len(Pk_terms_names)):
        if do_regularize_pk:
            if do_reg_all:
                save_file_name_suffix = Pk_terms_names[j1] + '_isregpk_' + str(do_reg_all) + '_regk_' + str(
                    reg_k) + '_regc_' + str(reg_c) + '_MICE_cosmo' + '.npz'
            else:
                if Pk_terms_names[j1] == 'k2Pk':
                    save_file_name_suffix = Pk_terms_names[j1] + '_isregpk_' + str(True) + '_regk_' + str(
                        reg_k) + '_regc_' + str(reg_c) + '_MICE_cosmo' + '.npz'

                else:
                    save_file_name_suffix = Pk_terms_names[j1] + '_isregpk_' + str(do_reg_all) + '_MICE_cosmo' + '.npz'
        else:
            save_file_name_suffix = Pk_terms_names[
                                        j1] + '_isregpk_' + str(do_regularize_pk) + '_MICE_cosmo' + '.npz'

        if Pk_terms_names[j1] != 'sig4':
            np.savez(save_xi_dir + 'xi_' + pt_type + '_' + save_file_name_suffix, z=znl, r=r_array, xi=xi_all[j1, :, :])
        np.savez(save_xi_dir + 'Pk_' + pt_type + '_' + save_file_name_suffix, z=znl, k=k_hres,
                 pkz=Pkth_array_khres[j1, :, :])

    if do_plot:
        fig1, ax1 = plt.subplots(1, 5, figsize=(36, 6), sharey=True)
        fig2, ax2 = plt.subplots(1, 5, figsize=(36, 6), sharey=True)

    for j in range(5):
        xi_gg = block.get_double_array_1d("pk_to_xi", "xi_gg_bin % s" % (j + 1))
        xi_gm = block.get_double_array_1d("pk_to_xi", "xi_gm_bin % s" % (j + 1))
        xi_mm = block.get_double_array_1d("pk_to_xi", "xi_mm_bin % s" % (j + 1))

        Pk_gg = block.get_double_array_1d("pk_to_xi", "Pk_gg_bin % s" % (j + 1))
        Pk_gm = block.get_double_array_1d("pk_to_xi", "Pk_gm_bin % s" % (j + 1))
        Pk_mm = block.get_double_array_1d("pk_to_xi", "Pk_mm_bin % s" % (j + 1))

        np.savez(save_xi_dir + 'xi_gg_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', r=r_array, xi=xi_gg)
        np.savez(save_xi_dir + 'xi_gm_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', r=r_array, xi=xi_gm)
        np.savez(save_xi_dir + 'xi_mm_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', r=r_array, xi=xi_mm)
        np.savez(save_xi_dir + 'Pk_gg_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', k=k_hres, Pk=Pk_gg)
        np.savez(save_xi_dir + 'Pk_gm_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', k=k_hres, Pk=Pk_gm)
        np.savez(save_xi_dir + 'Pk_mm_total_' + pt_type + '_' + pt_type_values + '_bin_' + str(
            j + 1) + '_MICE_cosmo' + save_xi_def + '.npz', k=k_hres, Pk=Pk_mm)

        if do_plot:
            ax1[j].plot(k_hres, Pk_gg, linestyle='-', color='blue', label=r'$P_{gg}$')
            ax1[j].plot(k_hres, Pk_gm, linestyle='-', color='red', label=r'$P_{gm}$')
            ax1[j].plot(k_hres, Pk_mm, linestyle='-', color='black', label=r'$P_{mm}$')
            ax1[j].set_xscale('log')
            ax1[j].set_yscale('log')
            ax1[j].set_xlabel(r'k  $(h/Mpc)$', size=17)
            if j == 0:
                ax1[j].set_ylabel(r'P(k) $(h^{3}Mpc^{-3})$', size=17)
            if j == 4:
                ax1[j].legend(fontsize=17, frameon=False, loc='upper right')
            ax1[j].tick_params(axis='both', which='major', labelsize=14)
            ax1[j].tick_params(axis='both', which='minor', labelsize=14)

            ax2[j].plot(r_array, xi_gg, linestyle='-', color='blue', label=r'$\xi_{gg}$')
            ax2[j].plot(r_array, xi_gm, linestyle='-', color='red', label=r'$\xi_{gm}$')
            ax2[j].plot(r_array, xi_mm, linestyle='-', color='black', label=r'$\xi_{mm}$')
            ax2[j].set_xscale('log')
            ax2[j].set_yscale('log')
            ax2[j].set_xlabel(r'R  $(Mpc/h)$', size=17)
            if j == 0:
                ax2[j].set_ylabel(r'$\xi$', size=17)
            if j == 4:
                ax2[j].legend(fontsize=17, frameon=False, loc='upper right')
            ax2[j].tick_params(axis='both', which='major', labelsize=14)
            ax2[j].tick_params(axis='both', which='minor', labelsize=14)

    if do_plot:
        fig1.tight_layout()
        fig1.savefig(save_plot_dir + 'Pk_bestfit_' + pt_type + '_' + pt_type_values + '_' + save_plot_def + '.png')
        fig2.tight_layout()
        fig2.savefig(save_plot_dir + 'xi_bestfit_' + pt_type + '_' + pt_type_values + '_' + save_plot_def + '.png')
        plt.close()

    return 0


def get_corr(cov):
    corr = np.zeros(cov.shape)
    for ii in xrange(0, cov.shape[0]):
        for jj in xrange(0, cov.shape[1]):
            corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])
    return corr


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


def make_plots_xi_cov(block, stat_type, no_cov_gg_gm, no_cov_zbins_only_gg_gm, no_cov_zbins_all, cov_diag, do_plot=True,
                      save_plot_dir='', save_plot_def=''):
    likes = names.likelihoods
    cov_obs_comp = block[likes, 'cov_obs_comp']
    incov_obs_comp = block[likes, 'incov_obs_comp']
    xi_theory_rdata = block[likes, 'xi_theory_rdata']
    # xi_data_gtcut = block[likes, 'xi_data_gtcut']

    if do_plot:
        print('npoints total : ' + str(len(xi_theory_rdata)))
        # cov_d = np.diag(cov_obs_comp)
        # fig, ax = plt.subplots(1, len(r_data), figsize=(8 * (len(r_data) - 1), 10), sharey=True)
        # k = 0
        # for j in range(len(r_data)):
        #     ax[j].errorbar(r_data[j], xi_data_gtcut[k:k + len(r_data[j])], np.sqrt(cov_d[k:k + len(r_data[j])]),
        #                    marker='*',
        #                    linestyle='', color='red')
        #     ax[j].plot(r_data[j], xi_theory_rdata[k:k + len(r_data[j])], linestyle='-', color='blue')
        #     ax[j].set_xscale('log')
        #     ax[j].set_xlabel(r'R  (Mpc/h)', size=12)
        #     k = k + len(r_data[j])
        # plt.tight_layout()
        # fig.savefig(save_plot_dir + 'compare_theory_data_' + save_plot_def + '.png')
        # plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        corr = ax.imshow(np.log(np.abs(cov_obs_comp)))
        fig.colorbar(corr, ax=ax)
        fig.tight_layout()
        fig.savefig(save_plot_dir + 'full_logabs_cov_' + stat_type + '_no_cov_zbins_only_gg_gm_' + str(
            no_cov_zbins_only_gg_gm) + '_no_cov_zbins_all_' + str(no_cov_zbins_all) + '_no_cov_gg_gm_' + str(
            no_cov_gg_gm) + '_cov_diag_' + str(
            cov_diag) + save_plot_def + '.png', dpi=240)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        corr = ax.imshow(np.log(np.abs(incov_obs_comp)))
        fig.colorbar(corr, ax=ax)
        fig.tight_layout()
        fig.savefig(save_plot_dir + 'full_logabs_invcov_' + stat_type + '_no_cov_zbins_only_gg_gm_' + str(
            no_cov_zbins_only_gg_gm) + '_no_cov_zbins_all_' + str(no_cov_zbins_all) + '_cov_diag_' + str(
            cov_diag) + save_plot_def + '.png', dpi=240)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        corr = ax.imshow((get_corr(cov_obs_comp)), clim=(-1.0, 1.0))
        fig.colorbar(corr, ax=ax)
        fig.tight_layout()
        fig.savefig(save_plot_dir + 'corrmat_' + stat_type + '_no_cov_zbins_only_gg_gm_' + str(
            no_cov_zbins_only_gg_gm) + '_no_cov_zbins_all_' + str(no_cov_zbins_all) + '_cov_diag_' + str(
            cov_diag) + save_plot_def + '.png', dpi=240)
        plt.close()


def setup(options):
    # read data from a 2pt file
    bins_all = ast.literal_eval(options.get_string(option_section, "bins_all", "[1, 2, 3, 4, 5]"))
    bins_to_fit = ast.literal_eval(options.get_string(option_section, "bins_to_fit", "[1, 2, 3, 4, 5]"))
    sc_save2pt = options.get_string(option_section, "sc_save")
    stat_type = options.get_string(option_section, "stat_type", 'gg_gm')
    pt_type_values = options.get_string(option_section, "pt_type_values")
    pt_type = options.get_string(option_section, "pt_type_g")
    save2pt_dir = options.get_string(option_section, "save2pt_dir")
    def_save = options.get_string(option_section, "def_save", '')
    do_regularize_pk = options.get_bool(option_section, "do_regularize", False)
    do_reg_all = options.get_bool(option_section, "do_reg_all", False)
    reg_k = options.get_double(option_section, "reg_k", 0.3)
    reg_c = options.get_double(option_section, "reg_c", 1000.)
    cov_diag = options.get_bool(option_section, "cov_diag", False)
    no_cov_zbins_only_gg_gm = options.get_bool(option_section, "no_cov_zbins_only_gg_gm", False)
    no_cov_zbins_all = options.get_bool(option_section, "no_cov_zbins_all", False)
    no_cov_gg_gm = options.get_bool(option_section, "no_cov_gg_gm", False)

    use_mean_z = options.get_bool(option_section, "use_mean_z", True)
    do_plot = options.get_bool(option_section, "do_plot", True)
    save_plot_dir = options.get_string(option_section, "save_plot_dir",
                                       '/global/project/projectdirs/des/shivamp/actxdes/data_set/mice_sims/measurements/')

    bins_to_rem = copy.deepcopy(bins_all)
    for bins in bins_to_fit:
        bins_to_rem.remove(bins)

    filename = options.get_string(option_section, "2PT_FILE")
    data = pk.load(open(filename, 'rb'))

    r_obs, data_obs, cov_obs = data['sep'], data['mean'], data['cov']
    r_obs_new, data_obs_new, cov_obs_new = import_data(r_obs, data_obs, cov_obs, bins_to_rem, bins_to_fit, bins_all,
                                                       stat_type)

    return r_obs_new, data_obs_new, cov_obs_new, stat_type, bins_to_fit, use_mean_z, pt_type_values, pt_type, sc_save2pt, \
           save2pt_dir, def_save, do_plot, save_plot_dir, do_regularize_pk, do_reg_all, reg_k, reg_c, cov_diag, \
           no_cov_zbins_only_gg_gm, no_cov_zbins_all, no_cov_gg_gm


def execute(block, config):
    r_obs_new, data_obs_new, cov_obs_new, stat_type, bins_to_fit, use_mean_z, pt_type_values, pt_type, sc_save2pt, save2pt_dir, \
    def_save, do_plot, save_plot_dir, do_regularize_pk, do_reg_all, reg_k, reg_c, cov_diag, no_cov_zbins_only_gg_gm, \
    no_cov_zbins_all, no_cov_gg_gm = config

    save_2pt(block, r_obs_new, data_obs_new, cov_obs_new, stat_type, bins_to_fit, pt_type, pt_type_values,
             sc_save2pt, save2pt_dir, def_save, do_plot=do_plot, save_plot_dir=save_plot_dir)

    make_plots_xi_cov(block, stat_type, no_cov_gg_gm, no_cov_zbins_only_gg_gm, no_cov_zbins_all, cov_diag,
                      do_plot=do_plot, save_plot_dir=save_plot_dir, save_plot_def=def_save)
    # save_xi_pk(block, do_regularize_pk, do_reg_all, reg_k, reg_c, pt_type, pt_type_values, save_xi_dir=save2pt_dir,
    #            save_xi_def=def_save, do_plot=do_plot, save_plot_dir=save_plot_dir)

    return 0


def cleanup(config):
    pass
