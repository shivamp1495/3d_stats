# coding:utf-8
import os
import ctypes as ct
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as intspline
import scipy.interpolate as interp
import scipy as sp
from cosmosis.datablock import names, option_section, BlockError
import re
import sys
import copy
import time
import pickle as pk
import pdb

sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/cosmosis-des-library/tcp/fast_pt/')
import FASTPT_2_1.FASTPT as FASTPT
from FASTPT_2_1.P_extend import k_extend
import pdb

sys.path.insert(0, os.environ['COSMOSIS_SRC_DIR'] + '/cosmosis-des-library/tcp/fast_pt/non_linear_bias/cleft_code/')
import cleftpool as cpool


def Pk2corr(r, k_array, Pk_array):
    toint = (k_array ** 2) * Pk_array * (np.sin(k_array * r)) / (k_array * r)
    valf = (1 / (2 * np.pi ** 2)) * sp.integrate.simps(toint, k_array)
    return valf


def Pk2corr_mat(r, k_array, Pk_mat):
    nz_p, nk_p = Pk_mat.shape
    k_mat = np.tile(k_array, (nz_p, 1))
    toint = np.multiply(np.multiply(np.multiply(k_mat, k_mat), Pk_mat), np.divide(np.sin(k_array * r), (k_array * r)))
    valf = (1 / (2 * np.pi ** 2)) * sp.integrate.simps(toint, k_array)
    return valf


def reg_Pk(Pk_mat1, Pk_mat2, k_array, tk, c_val=1.):
    nz_p, nk_p = Pk_mat1.shape
    ind_tk = np.where(k_array > tk)[0][0]
    Pk1_val = Pk_mat1[:, ind_tk]
    Pk2_val = Pk_mat2[:, ind_tk]
    Pk2_scale = (np.tile(Pk1_val / Pk2_val, (nk_p, 1))).T
    Pk_mat2_scaled = np.multiply(Pk_mat2, Pk2_scale)
    tanh_arg_mat = np.tile(c_val * (np.log(k_array) - np.log(k_array[ind_tk])), (nz_p, 1))
    tanh_func = np.tanh(tanh_arg_mat)
    Pk_mat_smooth = Pk_mat1 + np.multiply((1. + tanh_func) / 2., (Pk_mat2_scaled - Pk_mat1))
    return Pk_mat_smooth


def reg_Pk_gaussian(Pk_mat1, k_array, tk, c_val=1.):
    nz_p, nk_p = Pk_mat1.shape
    Pk_mat2 = np.tile(np.exp(-k_array ** 2), (nz_p, 1))
    ind_tk = np.where(k_array > tk)[0][0]
    Pk1_val = Pk_mat1[:, ind_tk]
    Pk2_val = Pk_mat2[:, ind_tk]
    Pk2_scale = (np.tile(Pk1_val / Pk2_val, (nk_p, 1))).T
    Pk_mat2_scaled = np.multiply(Pk_mat2, Pk2_scale)
    tanh_arg_mat = np.tile(c_val * (np.log(k_array) - np.log(k_array[ind_tk])), (nz_p, 1))
    tanh_func = np.tanh(tanh_arg_mat)
    Pk_mat_smooth = Pk_mat1 + np.multiply((1. + tanh_func) / 2., (Pk_mat2_scaled - Pk_mat1))
    return Pk_mat_smooth


def get_zmean(zcent, nz_bin):
    prob_zcent = nz_bin
    delz = zcent[1] - zcent[0]
    zmean = (np.sum(prob_zcent * zcent * delz)) / (np.sum(prob_zcent * delz))
    return zmean


def get_PXm_terms(bias_param, Pk_th_array, pt_type=None):
    if pt_type in ['oneloop_eul_bk']:
        Pklin, Pk_halofit, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, k2Pk, sig4 = Pk_th_array[0], Pk_th_array[1], \
                                                                                   Pk_th_array[2], \
                                                                                   Pk_th_array[3], Pk_th_array[4], \
                                                                                   Pk_th_array[5], \
                                                                                   Pk_th_array[6], Pk_th_array[7], \
                                                                                   Pk_th_array[8], Pk_th_array[9]

        b1E, b2E, bsE, b3nl, bk = bias_param[0], bias_param[1], bias_param[2], bias_param[3], bias_param[4]
        PXmNL = b1E * Pk_halofit + (b2E / 2.) * Pd1d2 + (bsE / 2.) * Pd1s2 + (b3nl / 2.) * sig3nl + bk * k2Pk
        PXmNL_terms = [b1E * Pk_halofit, (b2E / 2.) * Pd1d2, (bsE / 2.) * Pd1s2, (b3nl / 2.) * sig3nl, bk * k2Pk]

    elif pt_type in ['oneloop_cleft_bk']:
        Pklin, Pk_halofit, Pzel, PA, PW, Pd1, Pd1d1, Pd2, Pd2d2, Pd1d2, Ps2, Pd1s2, Pd2s2, Ps2s2, PD2, Pd1D2, k2Pk = \
            Pk_th_array[0], Pk_th_array[1], Pk_th_array[2], Pk_th_array[3], Pk_th_array[4], Pk_th_array[5], Pk_th_array[
                6], \
            Pk_th_array[7], Pk_th_array[8], Pk_th_array[9], Pk_th_array[10], Pk_th_array[11], Pk_th_array[12], \
            Pk_th_array[13], Pk_th_array[14], Pk_th_array[15], Pk_th_array[16]

        b1E, b1L, b2L, bsL, bk = bias_param[0], bias_param[1], bias_param[2], bias_param[3], bias_param[4]

        PXmNL = b1E * Pk_halofit + (b1L / 2.) * Pd1 + (b2L / 2.) * Pd2 + (bsL / 2.) * Ps2 + bk * k2Pk
        PXmNL_terms = [b1E * Pk_halofit, (b1L / 2.) * Pd1, (b2L / 2.) * Pd2, (bsL / 2.) * Ps2, bk * k2Pk]

    else:
        print 'give correct pt_type'
        PXmNL, PXmNL_terms = None, None

    return PXmNL, PXmNL_terms


def get_PXX_terms_bins(bias_param_bin1, bias_param_bin2, Pk_th_array, pt_type=None):
    if pt_type in ['oneloop_eul_bk']:
        Pklin, Pk_halofit, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, k2Pk, sig4 = Pk_th_array[0], Pk_th_array[1], \
                                                                                   Pk_th_array[2], \
                                                                                   Pk_th_array[3], Pk_th_array[4], \
                                                                                   Pk_th_array[5], \
                                                                                   Pk_th_array[6], Pk_th_array[7], \
                                                                                   Pk_th_array[8], Pk_th_array[9]

        b1E1, b2E1, bsE1, b3nl1, bk1 = bias_param_bin1[0], bias_param_bin1[1], bias_param_bin1[2], \
                                       bias_param_bin1[3], bias_param_bin1[4]

        b1E2, b2E2, bsE2, b3nl2, bk2 = bias_param_bin2[0], bias_param_bin2[1], bias_param_bin2[2], \
                                       bias_param_bin2[3], bias_param_bin2[4]

        PXXNL = (b1E1 * b1E2) * Pk_halofit + (1. / 2.) * (b1E1 * b2E2 + b1E2 * b2E1) * Pd1d2 + (1. / 4.) * (
                b2E1 * b2E2) * Pd2d2 + (1. / 2.) * (b1E1 * bsE2 + b1E2 * bsE1) * Pd1s2 + (1. / 4.) * (
                        b2E2 * bsE1 + b2E1 * bsE2) * Pd2s2 + (1. / 4.) * (bsE1 * bsE2) * Ps2s2 + (1. / 2.) * (
                        b1E1 * b3nl2 + b1E2 * b3nl1) * sig3nl + (b1E1 * bk2 + b1E2 * bk1) * k2Pk
        PXXNL_terms = [(b1E1 * b1E2) * Pk_halofit, (1. / 2.) * (b1E1 * b2E2 + b1E2 * b2E1) * Pd1d2,
                       (1. / 4.) * (b2E1 * b2E2) * Pd2d2, (1. / 2.) * (b1E1 * bsE2 + b1E2 * bsE1) * Pd1s2,
                       (1. / 4.) * (b2E2 * bsE1 + b2E1 * bsE2) * Pd2s2, (1. / 4.) * (bsE1 * bsE2) * Ps2s2,
                       (1. / 2.) * (b1E1 * b3nl2 + b1E2 * b3nl1) * sig3nl,
                       (b1E1 * bk2 + b1E2 * bk1) * k2Pk]

    elif pt_type in ['oneloop_cleft_bk']:
        Pklin, Pk_halofit, Pzel, PA, PW, Pd1, Pd1d1, Pd2, Pd2d2, Pd1d2, Ps2, Pd1s2, Pd2s2, Ps2s2, PD2, Pd1D2, k2Pk = \
            Pk_th_array[0], Pk_th_array[1], Pk_th_array[2], Pk_th_array[3], Pk_th_array[4], Pk_th_array[5], Pk_th_array[
                6], \
            Pk_th_array[7], Pk_th_array[8], Pk_th_array[9], Pk_th_array[10], Pk_th_array[11], Pk_th_array[12], \
            Pk_th_array[13], Pk_th_array[14], Pk_th_array[15], Pk_th_array[16]

        b1E1, b1L1, b2L1, bsL1, bk1 = bias_param_bin1[0], bias_param_bin1[1], bias_param_bin1[2], bias_param_bin1[3], \
                                      bias_param_bin1[4]
        b1E2, b1L2, b2L2, bsL2, bk2 = bias_param_bin2[0], bias_param_bin2[1], bias_param_bin2[2], bias_param_bin2[3], \
                                      bias_param_bin2[4]

        PXXNL = (b1E1 * b1E2) * Pk_halofit + (1. / 2.) * ((b1L1 + b1L2) * Pd1) + (b1L1 * b1L2) * Pd1d1 + (
                1. / 2.) * (b1L2 * b2L1 + b1L1 * b2L2) * Pd1d2 + (1. / 2.) * (
                        b2L1 + b2L2) * Pd2 + (b2L1 * b2L2) * Pd2d2 + (1. / 2.) * (
                        bsL1 + bsL2) * Ps2 + (bsL1 * bsL2) * Ps2s2 + (
                        1. / 2.) * (bsL2 * b1L1 + bsL1 * b1L2) * Pd1s2 + (
                        1. / 2.) * (b2L2 * bsL1 + b2L1 * bsL2) * Pd2s2 + (b1E1 * bk2 + b1E2 * bk1) * k2Pk
        PXXNL_terms = [(b1E1 * b1E2) * Pk_halofit, (1. / 2.) * ((b1L1 + b1L2) * Pd1), (b1L1 * b1L2) * Pd1d1,
                       (1. / 2.) * (b1L2 * b2L1 + b1L1 * b2L2) * Pd1d2, (1. / 2.) * (b2L1 + b2L2) * Pd2,
                       (b2L1 * b2L2) * Pd2d2, (1. / 2.) * (bsL1 + bsL2) * Ps2, (bsL1 * bsL2) * Ps2s2,
                       (1. / 2.) * (bsL2 * b1L1 + bsL1 * b1L2) * Pd1s2, (1. / 2.) * (b2L2 * bsL1 + b2L1 * bsL2) * Pd2s2,
                       (b1E1 * bk2 + b1E2 * bk1) * k2Pk]
    else:
        print 'give correct pt_type'
        PXXNL, PXXNL_terms = None, None

    return PXXNL, PXXNL_terms


def get_xiXX_terms_bins(bias_param_bin1, bias_param_bin2, xi_all_array, pt_type=None):
    if pt_type in ['oneloop_eul_bk']:
        xi_lin, xi_halofit, xi_d1d2, xi_d2d2, xi_d1s2, xi_d2s2, xi_s2s2, sig3nl, k2Pk_xi = xi_all_array[0], \
                                                                                           xi_all_array[1], \
                                                                                           xi_all_array[2], \
                                                                                           xi_all_array[3], \
                                                                                           xi_all_array[4], \
                                                                                           xi_all_array[5], \
                                                                                           xi_all_array[6], \
                                                                                           xi_all_array[7], \
                                                                                           xi_all_array[8]

        b1E1, b2E1, bsE1, b3nl1, bk1 = bias_param_bin1[0], bias_param_bin1[1], bias_param_bin1[2], \
                                       bias_param_bin1[3], bias_param_bin1[4]

        b1E2, b2E2, bsE2, b3nl2, bk2 = bias_param_bin2[0], bias_param_bin2[1], bias_param_bin2[2], \
                                       bias_param_bin2[3], bias_param_bin2[4]

        xi_XXNL = (b1E1 * b1E2) * xi_halofit + (1. / 2.) * (b1E1 * b2E2 + b1E2 * b2E1) * xi_d1d2 + (1. / 4.) * (
                b2E1 * b2E2) * xi_d2d2 + (
                          1. / 2.) * (b1E1 * bsE2 + b1E2 * bsE1) * xi_d1s2 + (1. / 4.) * (
                          b2E2 * bsE1 + b2E1 * bsE2) * xi_d2s2 + (
                          1. / 4.) * (bsE1 * bsE2) * xi_s2s2 + (1. / 2.) * (b1E1 * b3nl2 + b1E2 * b3nl1) * sig3nl + (
                          b1E1 * bk2 + b1E2 * bk1) * k2Pk_xi
        xi_XXNL_terms = [(b1E1 * b1E2) * xi_halofit, (1. / 2.) * (b1E1 * b2E2 + b1E2 * b2E1) * xi_d1d2,
                         (1. / 4.) * (b2E1 * b2E2) * xi_d2d2,
                         (1. / 2.) * (b1E1 * bsE2 + b1E2 * bsE1) * xi_d1s2,
                         (1. / 4.) * (b2E2 * bsE1 + b2E1 * bsE2) * xi_d2s2, (1. / 4.) * (bsE1 * bsE2) * xi_s2s2,
                         (1. / 2.) * (b1E1 * b3nl2 + b1E2 * b3nl1) * sig3nl,
                         (b1E1 * bk2 + b1E2 * bk1) * k2Pk_xi]

    elif pt_type in ['oneloop_cleft_bk']:
        xi_lin, xi_halofit, xi_zel, xi_A, xi_W, xi_d1, xi_d1d1, xi_d2, xi_d2d2, xi_d1d2, xi_s2, xi_d1s2, xi_d2s2, xi_s2s2, xi_D2, xi_d1D2, k2Pk_xi = \
            xi_all_array[0], xi_all_array[1], xi_all_array[2], xi_all_array[3], xi_all_array[4], xi_all_array[5], \
            xi_all_array[
                6], \
            xi_all_array[7], xi_all_array[8], xi_all_array[9], xi_all_array[10], xi_all_array[11], xi_all_array[12], \
            xi_all_array[13], xi_all_array[14], xi_all_array[15], xi_all_array[16]

        b1E1, b1L1, b2L1, bsL1, bk1 = bias_param_bin1[0], bias_param_bin1[1], bias_param_bin1[2], bias_param_bin1[3], \
                                      bias_param_bin1[4]
        b1E2, b1L2, b2L2, bsL2, bk2 = bias_param_bin2[0], bias_param_bin2[1], bias_param_bin2[2], bias_param_bin2[3], \
                                      bias_param_bin2[4]

        xi_XXNL = (b1E1 * b1E2) * xi_halofit + (1. / 2.) * ((b1L1 + b1L2) * xi_d1) + (b1L1 * b1L2) * xi_d1d1 + (
                1. / 2.) * (b1L2 * b2L1 + b1L1 * b2L2) * xi_d1d2 + (1. / 2.) * (
                          b2L1 + b2L2) * xi_d2 + (b2L1 * b2L2) * xi_d2d2 + (1. / 2.) * (
                          bsL1 + bsL2) * xi_s2 + (bsL1 * bsL2) * xi_s2s2 + (
                          1. / 2.) * (bsL2 * b1L1 + bsL1 * b1L2) * xi_d1s2 + (
                          1. / 2.) * (b2L2 * bsL1 + b2L1 * bsL2) * xi_d2s2 + (b1E1 * bk2 + b1E2 * bk1) * k2Pk_xi
        xi_XXNL_terms = [(b1E1 * b1E2) * xi_halofit, (1. / 2.) * ((b1L1 + b1L2) * xi_d1), (b1L1 * b1L2) * xi_d1d1,
                         (1. / 2.) * (b1L2 * b2L1 + b1L1 * b2L2) * xi_d1d2, (1. / 2.) * (b2L1 + b2L2) * xi_d2,
                         (b2L1 * b2L2) * xi_d2d2, (1. / 2.) * (bsL1 + bsL2) * xi_s2, (bsL1 * bsL2) * xi_s2s2,
                         (1. / 2.) * (bsL2 * b1L1 + bsL1 * b1L2) * xi_d1s2,
                         (1. / 2.) * (b2L2 * bsL1 + b2L1 * bsL2) * xi_d2s2,
                         (b1E1 * bk2 + b1E2 * bk1) * k2Pk_xi]

    else:
        print 'give correct pt_type'
        xi_XXNL, xi_XXNL_terms = None, None

    return xi_XXNL, xi_XXNL_terms


def get_xiXm_terms(bias_param, xi_all_array, pt_type=None):
    if pt_type in ['oneloop_eul_bk']:
        xi_lin, xi_halofit, xi_d1d2, xi_d2d2, xi_d1s2, xi_d2s2, xi_s2s2, sig3nl, k2Pk_xi = xi_all_array[0], \
                                                                                           xi_all_array[1], \
                                                                                           xi_all_array[2], \
                                                                                           xi_all_array[3], \
                                                                                           xi_all_array[4], \
                                                                                           xi_all_array[5], \
                                                                                           xi_all_array[6], \
                                                                                           xi_all_array[7], \
                                                                                           xi_all_array[8]

        b1E, b2E, bsE, b3nl, bk = bias_param[0], bias_param[1], bias_param[2], bias_param[3], bias_param[4]

        xi_XmNL = b1E * xi_halofit + (b2E / 2.) * xi_d1d2 + (bsE / 2.) * xi_d1s2 + (b3nl / 2.) * sig3nl + bk * k2Pk_xi
        xi_XmNL_terms = [b1E * xi_halofit, (b2E / 2.) * xi_d1d2, (bsE / 2.) * xi_d1s2, (b3nl / 2.) * sig3nl,
                         bk * k2Pk_xi]

    elif pt_type in ['oneloop_cleft_bk']:
        xi_lin, xi_halofit, xi_zel, xi_A, xi_W, xi_d1, xi_d1d1, xi_d2, xi_d2d2, xi_d1d2, xi_s2, xi_d1s2, xi_d2s2, xi_s2s2, xi_D2, xi_d1D2, k2Pk_xi = \
            xi_all_array[0], xi_all_array[1], xi_all_array[2], xi_all_array[3], xi_all_array[4], xi_all_array[5], \
            xi_all_array[
                6], \
            xi_all_array[7], xi_all_array[8], xi_all_array[9], xi_all_array[10], xi_all_array[11], xi_all_array[12], \
            xi_all_array[13], xi_all_array[14], xi_all_array[15], xi_all_array[16]

        b1E, b1L, b2L, bsL, bk = bias_param[0], bias_param[1], bias_param[2], bias_param[3], bias_param[4]

        xi_XmNL = b1E * xi_halofit + (b1L / 2.) * xi_d1 + (b2L / 2.) * xi_d2 + (bsL / 2.) * xi_s2 + bk * k2Pk_xi
        xi_XmNL_terms = [b1E * xi_halofit, (b1L / 2.) * xi_d1, (b2L / 2.) * xi_d2, (bsL / 2.) * xi_s2, bk * k2Pk_xi]

    else:
        print 'give correct pt_type'
        xi_XmNL, xi_XmNL_terms = None, None

    return xi_XmNL, xi_XmNL_terms


def get_Pktharray(output_nl_grid, klin, knl, Pkzlin, Pnl_kz, usePNL_for_Pk=True, pt_type=None, Pk_terms_names=None):
    klinlog = np.log(klin)

    Pk0lin = Pkzlin[0, :]

    ind = np.where(klin > 0.03)[0][0]

    Growth = np.sqrt(Pkzlin[:, ind] / Pkzlin[0, ind])

    nk = 4 * len(klin)  # higher res increases runtime, decreases potential ringing at low-k
    # eps = 1e-6
    # eps = 0.
    # kmin = np.log10((1. + eps) * klin[0])
    # kmax = np.log10((1. - eps) * klin[-1])

    kmin = -6.0
    kmax = 3.0

    klin_fpt = np.logspace(kmin, kmax, nk)
    k1log = np.log(klin_fpt)
    #    pinterp=interp1d(klinlog,np.log(Pk0lin),bounds_error=False, fill_value="extrapolate")
    plininterp = interp1d(klinlog, np.log(Pk0lin), fill_value='extrapolate', bounds_error=False)

    ## This interpolation should be at the input bounds. Extrapolate used to avoid failure due to numerical noise. No actual extrapolation is done. We could handle this using a margin or something else
    Plin_klin_fpt = np.exp(plininterp(k1log))

    if (knl[0] < klin_fpt[0]) or (knl[-1] > klin_fpt[-1]):
        EK1 = k_extend(klin_fpt, np.log10(knl[0]), np.log10(knl[-1]))
        klin_fpt = EK1.extrap_k()
        Plin_klin_fpt = EK1.extrap_P_low(Plin_klin_fpt)
        Plin_klin_fpt = EK1.extrap_P_high(Plin_klin_fpt)

    klin_fpt_log = np.log(klin_fpt)
    knl_log = np.log(knl)

    if output_nl_grid:
        temp = intspline(klin_fpt_log, np.log(Plin_klin_fpt))
        Plin = np.exp(temp(knl_log))
        Pnl1 = Pnl_kz
        if usePNL_for_Pk:
            knl2mat = np.tile(knl ** 2, (Pnl_kz.shape[0], 1))
            k2Pnl1 = np.multiply(knl2mat, Pnl_kz)
        else:
            k2Pnl1 = np.outer(Growth ** 2, (knl ** 2) * Plin)
    else:
        Plin = Plin_klin_fpt
        Pnl1 = Pnl_kz
        if usePNL_for_Pk:
            k2Pnl1 = np.outer(Growth ** 2, (klin_fpt ** 2) * Pnl[0, :])
        else:
            k2Pnl1 = np.outer(Growth ** 2, (klin_fpt ** 2) * Plin)

    Plin_kz = np.outer(Growth ** 2, Plin)

    n_pad = len(klin_fpt)

    if pt_type in ['oneloop_eul_bk']:

        fastpt = FASTPT.FASTPT(klin_fpt, to_do=['one_loop_dd'], low_extrap=-5, high_extrap=3, n_pad=n_pad)
        PXXNL_b1b2bsb3nl = fastpt.one_loop_dd_bias_b3nl(Plin_klin_fpt, C_window=.75)
        if output_nl_grid:

            temp = intspline(klin_fpt_log, (PXXNL_b1b2bsb3nl[2]))
            Pd1d2 = np.outer(Growth ** 4, (temp(knl_log)))
            temp = intspline(klin_fpt_log, np.log(PXXNL_b1b2bsb3nl[3]))
            Pd2d2 = np.outer(Growth ** 4, np.exp(temp(knl_log)))
            temp = intspline(klin_fpt_log, (PXXNL_b1b2bsb3nl[4]))
            Pd1s2 = np.outer(Growth ** 4, (temp(knl_log)))
            temp = intspline(klin_fpt_log, (PXXNL_b1b2bsb3nl[5]))
            Pd2s2 = np.outer(Growth ** 4, (temp(knl_log)))
            temp = intspline(klin_fpt_log, np.log(PXXNL_b1b2bsb3nl[6]))
            Ps2s2 = np.outer(Growth ** 4, np.exp(temp(knl_log)))
            temp = intspline(klin_fpt_log, (PXXNL_b1b2bsb3nl[7]))
            sig3nl = np.outer(Growth ** 4, (temp(knl_log)))
            sig4 = np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[8] * np.ones_like(knl))
        else:
            [Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, sig4] = [
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[2]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[3]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[4]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[5]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[6]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[7]),
                np.outer(Growth ** 4, PXXNL_b1b2bsb3nl[8] * np.ones_like(
                    PXXNL_b1b2bsb3nl[0]))]

        Pk_th_array = [Plin_kz, Pnl1, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl, k2Pnl1, sig4]


    elif pt_type in ['oneloop_cleft_bk']:
        cl = cpool.CLEFT(k=klin_fpt, p=Plin_klin_fpt)
        Pk_th_array = np.zeros((len(Pk_terms_names), len(Growth), len(knl)))

        kinput_ind = np.where((knl < 4.) & (knl > 1e-5))[0]
        kinput = knl[kinput_ind]

        # pdb.set_trace()
        pk_table_f = cpool.get_nonlinear_kernels_wgroth(cl, kinput, koutput=knl, Growth=Growth, do_analysis=False,
                                                        Pnl_toplot=Pnl1)

        Pk_th_array[1:-1, :, :] = pk_table_f
        Pk_th_array[0, :, :] = Plin_kz
        Pk_th_array[1, :, :] = Pnl1
        Pk_th_array[-1, :, :] = k2Pnl1
    else:
        Pk_th_array = None
        print('give correct pt_type')

    if output_nl_grid:
        kout = knl
    else:
        kout = klin_fpt

    return Pk_th_array, kout


def get_bias_param_bin_i(block, bin_i, bias_section, constraint_b2, pt_type=None):
    b1E = block[bias_section, "b1E_bin%s" % (bin_i + 1)]
    if pt_type in ['oneloop_eul_bk']:
        if constraint_b2:
            param_name = ['b1E', 'c1b2E', 'bsE', 'b3nlE', 'bkE']
            param_defaults = [None, 0.0, (-4. / 7.) * (b1E - 1), (b1E - 1), 0.0]

            param_array_bin = np.array([])
            for j in xrange(len(param_name)):
                params = param_name[j]
                if params == 'c1b2E':
                    param_value = block.get_double(bias_section, params + "_bin%s" % (bin_i + 1), param_defaults[
                        j]) - 2.143 * param_value_b1 + 0.929 * param_value_b1 ** 2 + 0.008 * param_value_b1 ** 3
                else:
                    param_value = block.get_double(bias_section, params + "_bin%s" % (bin_i + 1), param_defaults[j])
                    if params == 'b1E':
                        param_value_b1 = param_value

                param_array_bin = np.append(param_array_bin, param_value)

        else:
            param_name = ['b1E', 'b2E', 'bsE', 'b3nlE', 'bkE']
            param_defaults = [None, 0.0, (-4. / 7.) * (b1E - 1), (b1E - 1), 0.0]

            param_array_bin = np.array([])
            for j in xrange(len(param_name)):
                params = param_name[j]
                param_array_bin = np.append(param_array_bin,
                                            block.get_double(bias_section, params + "_bin%s" % (bin_i + 1),
                                                             param_defaults[j]))

    elif pt_type in ['oneloop_cleft_bk']:
        param_name = ['b1E', 'b1L', 'b2L', 'bsL', 'bkE']
        param_defaults = [None, b1E - 1, 0.0, 0.0, 0.0]

        param_array_bin = np.array([])
        for j in xrange(len(param_name)):
            params = param_name[j]
            param_array_bin = np.append(param_array_bin,
                                        block.get_double(bias_section, params + "_bin%s" % (bin_i + 1),
                                                         param_defaults[j]))

    else:
        print 'No predefined pt_type given'
        param_array_bin = None

    return param_array_bin


def setup(options):
    nz_dir = options.get_string(option_section, "nz_dir")
    rmin = options.get_double(option_section, "rmin", 0.1)
    rmax = options.get_double(option_section, "rmax", 50)
    nrbin = options.get_double(option_section, "nrbin", 90)

    k_hres_min = options.get_double(option_section, "k_hres_min", 0.0001)
    k_hres_max = options.get_double(option_section, "k_hres_max", 100)
    n_k_hres_bin = options.get_double(option_section, "n_k_hres_max", 50000)

    constraint_b2 = options.get_bool(option_section, "constraint_b2", False)

    output_nl_grid = options.get_bool(option_section, "output_nl_grid", True)
    do_regularize_pk = options.get_bool(option_section, "do_regularize_pk", False)
    do_reg_all = options.get_bool(option_section, "do_reg_all", False)
    reg_k = options.get_double(option_section, "reg_k", 0.3)
    reg_c = options.get_double(option_section, "reg_c", 1000.)

    do_save_xi = options.get_bool(option_section, "do_save_xi", False)
    save_xi_def = options.get_string(option_section, "save_xi_def", '')
    use_mean_z = options.get_bool(option_section, "use_mean_z", True)
    pt_type = options.get_string(option_section, "pt_type_g", '')

    r_array = np.logspace(np.log10(rmin), np.log10(rmax), nrbin)
    k_hres = np.logspace(np.log10(k_hres_min), np.log10(k_hres_max), n_k_hres_bin)

    config = {'nz_dir': nz_dir, 'r_array': r_array, 'k_hres': k_hres,  'bias_section': str(
        options.get_string(option_section, "bias_section")), 'constraint_b2': constraint_b2,
              'do_save_xi': do_save_xi,  'save_xi_def': save_xi_def,
               'use_mean_z': use_mean_z, 'pt_type': pt_type,
              'output_nl_grid': output_nl_grid, 'do_regularize_pk': do_regularize_pk, 'do_reg_all': do_reg_all,
              'reg_k': reg_k, 'reg_c': reg_c}

    return config


def execute(block, config):
    nz_dir = config['nz_dir']
    r_array = config['r_array']
    do_save_xi = config['do_save_xi']
    use_mean_z = config['use_mean_z']
    k_hres = config['k_hres']

    bias_section = config['bias_section']
    constraint_b2 = config['constraint_b2']
    pt_type = config['pt_type']

    output_nl_grid = config['output_nl_grid']
    do_regularize_pk = config['do_regularize_pk']
    do_reg_all = config['do_reg_all']
    reg_k = config['reg_k']
    reg_c = config['reg_c']

    block.put_double_array_1d("pk_to_xi", "r", r_array)
    block.put_double_array_1d("pk_to_xi", "k", k_hres)

    name_lin = names.matter_power_lin
    name_nl = names.matter_power_nl

    klin, zlin, Pkzlin = block[name_lin, "k_h"], block[name_lin, "z"], block[name_lin, "P_k"]
    knl, znl, Pnl_kz = block[name_nl, "k_h"], block[name_nl, "z"], block[name_nl, "P_k"]
    z_pk = znl

    if pt_type in ['oneloop_eul_bk']:
        Pk_terms_names = ['Plin', 'Pmm', 'Pd1d2', 'Pd2d2', 'Pd1s2', 'Pd2s2', 'Ps2s2', 'Pd1d3nl', 'k2Pk', 'sig4']
        xi_gg = np.zeros((len(Pk_terms_names) - 1, len(znl), len(r_array)))
    elif pt_type in ['oneloop_cleft_bk']:
        Pk_terms_names = ['Plin', 'Pnl1', 'Pzel', 'PA', 'PW', 'Pd1', 'Pd1d1', 'Pd2', 'Pd2d2', 'Pd1d2', 'Ps2', 'Pd1s2',
                          'Pd2s2', 'Ps2s2', 'PD2', 'Pd1D2', 'k2Pk']
        xi_gg = np.zeros((len(Pk_terms_names), len(znl), len(r_array)))
    else:
        print 'No predefined pt_type given'
        Pk_terms_names = None

    Pkth_array, karray = get_Pktharray(output_nl_grid, klin, knl, Pkzlin, Pnl_kz, pt_type=pt_type,
                                       Pk_terms_names=Pk_terms_names)

    Pkth_array_khres = np.zeros((len(Pk_terms_names), len(znl), len(k_hres)))

    for j1 in range(len(Pk_terms_names)):

        print 'processing Pk ' + str(Pk_terms_names[j1])

        P_gg_khres = np.zeros((len(znl), len(k_hres)))
        for i in range(len(znl)):
            P_gg_j1_i = Pkth_array[j1][i, :]

            Pgg_temp = intspline(karray, P_gg_j1_i)
            Pgg_term_interp = Pgg_temp(k_hres)
            P_gg_khres[i, :] = Pgg_term_interp

        if Pk_terms_names[j1] == 'Plin':
            P_lin_khres = P_gg_khres
            P_gg_khres_reg = P_gg_khres
        else:
            if do_regularize_pk:
                if do_reg_all:
                    P_gg_khres_reg = reg_Pk(P_gg_khres, P_lin_khres, k_hres, reg_k, c_val=reg_c)
                else:
                    if Pk_terms_names[j1] == 'k2Pk':
                        # P_gg_khres_reg = reg_Pk(P_gg_khres, P_lin_khres, k_hres, reg_k, c_val=reg_c)
                        P_gg_khres_reg = reg_Pk_gaussian(P_gg_khres, k_hres, reg_k, c_val=reg_c)
                    else:
                        P_gg_khres_reg = P_gg_khres
            else:
                P_gg_khres_reg = P_gg_khres

        if Pk_terms_names[j1] != 'sig4':
            for k in range(len(r_array)):
                xi_gg_f = Pk2corr_mat(r_array[k], k_hres, P_gg_khres_reg)
                xi_gg[j1, :, k] = xi_gg_f

        Pkth_array_khres[j1, :, :] = P_gg_khres_reg

    xi_all = xi_gg
    xi_mm = xi_all[1]
    Pk_mm = Pkth_array_khres[1]

    for j in range(5):

        xi_gm_bin_f, xi_gg_bin_f, xi_mm_bin_f = np.zeros(len(r_array)), np.zeros(len(r_array)), np.zeros(len(r_array))
        xi_gm_mm_bin_f, xi_gg_mm_bin_f = np.zeros(len(r_array)), np.zeros(len(r_array))

        filename = nz_dir + 'nz_g_m_' + '_zbin_' + str(j + 1) + '_dsg_' + str(1) + '_dsm_' + str(1) + '.pk'
        nz_data = pk.load(open(filename, 'rb'))

        nz_g, nz_m, nz_z = nz_data['nz_g'], nz_data['nz_m'], nz_data['nz_z']

        zmean_g = get_zmean(nz_z, nz_g)
        zmean_m = get_zmean(nz_z, nz_m)
        zmean_gm = (zmean_g + zmean_m) / 2.

        param_array_bin1 = get_bias_param_bin_i(block, j, bias_section, constraint_b2, pt_type=pt_type)
        param_array_bin2 = get_bias_param_bin_i(block, j, bias_section, constraint_b2, pt_type=pt_type)

        xi_gg, _ = get_xiXX_terms_bins(param_array_bin1, param_array_bin2, xi_all, pt_type=pt_type)
        xi_gm, _ = get_xiXm_terms(param_array_bin1, xi_all, pt_type=pt_type)

        for k in range(len(r_array)):
            xi_gm_temp = intspline(z_pk, xi_gm[:, k])
            xi_gg_temp = intspline(z_pk, xi_gg[:, k])
            xi_mm_temp = intspline(z_pk, xi_mm[:, k])

            if use_mean_z:
                xi_gm_bin_f[k] = xi_gm_temp(zmean_gm)
                xi_gg_bin_f[k] = xi_gg_temp(zmean_g)
                xi_mm_bin_f[k] = xi_mm_temp(zmean_m)
                xi_gg_mm_bin_f[k] = xi_gg_bin_f[k] / xi_mm_bin_f[k]
                xi_gm_mm_bin_f[k] = xi_gm_bin_f[k] / xi_mm_bin_f[k]
            else:
                xi_gm_bin_f[k] = (sp.integrate.simps(xi_gm_temp(nz_z) * nz_g * nz_m, nz_z)) / (
                    sp.integrate.simps(nz_g * nz_m, nz_z))
                xi_gg_bin_f[k] = (sp.integrate.simps(xi_gg_temp(nz_z) * nz_g * nz_g, nz_z)) / (
                    sp.integrate.simps(nz_g * nz_g, nz_z))
                if j == 0:
                    xi_mm_bin_f[k] = (sp.integrate.simps(xi_mm_temp(nz_z) * nz_m * nz_m, nz_z)) // (
                        sp.integrate.simps(nz_m * nz_m, nz_z))

        if do_save_xi:
            Pk_gg, _ = get_PXX_terms_bins(param_array_bin1, param_array_bin2, Pkth_array_khres, pt_type=pt_type)
            Pk_gm, _ = get_PXm_terms(param_array_bin1, Pkth_array_khres, pt_type=pt_type)
            Pk_gm_bin_f, Pk_gg_bin_f, Pk_mm_bin_f = np.zeros(len(k_hres)), np.zeros(len(k_hres)), np.zeros(len(k_hres))
            for i in range(len(k_hres)):
                Pk_gm_temp = intspline(z_pk, Pk_gm[:, i])
                Pk_gg_temp = intspline(z_pk, Pk_gg[:, i])
                Pk_mm_temp = intspline(z_pk, Pk_mm[:, i])

                Pk_gm_bin_f[i] = Pk_gm_temp(zmean_gm)
                Pk_gg_bin_f[i] = Pk_gg_temp(zmean_g)
                Pk_mm_bin_f[i] = Pk_mm_temp(zmean_m)

            block.put_double_array_1d("pk_to_xi", "Pk_gg_bin % s" % (j + 1), Pk_gg_bin_f)
            block.put_double_array_1d("pk_to_xi", "Pk_gm_bin % s" % (j + 1), Pk_gm_bin_f)
            block.put_double_array_1d("pk_to_xi", "Pk_mm_bin % s" % (j + 1), Pk_mm_bin_f)

            block.put_grid("pk_to_xi", "z", z_pk, "r", r_array, "xi_gg_mat_bin % s" % (j + 1), xi_gg)
            block.put_grid("pk_to_xi", "z", z_pk, "r", r_array, "xi_gm_mat_bin % s" % (j + 1), xi_gm)

            block.put_grid("pk_to_xi", "z", z_pk, "r", r_array, "xi_gg_mm_mat_bin % s" % (j + 1), xi_gg / xi_mm)
            block.put_grid("pk_to_xi", "z", z_pk, "r", r_array, "xi_gm_mm_mat_bin % s" % (j + 1), xi_gm / xi_mm)

            if j == 0:
                block.put("pk_to_xi", "Pk_all", Pkth_array_khres)
                block.put("pk_to_xi", "xi_all", xi_all)
                block.put_grid("pk_to_xi", "z", z_pk, "r", r_array, "xi_mm_mat", xi_mm)

        block.put_double_array_1d("pk_to_xi", "xi_gg_bin % s" % (j + 1), xi_gg_bin_f)
        block.put_double_array_1d("pk_to_xi", "xi_gm_bin % s" % (j + 1), xi_gm_bin_f)
        block.put_double_array_1d("pk_to_xi", "xi_mm_bin % s" % (j + 1), xi_mm_bin_f)

        block.put_double_array_1d("pk_to_xi", "xi_gg_mm_bin % s" % (j + 1), xi_gg_mm_bin_f)
        block.put_double_array_1d("pk_to_xi", "xi_gm_mm_bin % s" % (j + 1), xi_gm_mm_bin_f)

        block.put_double("pk_to_xi", "zmean_gg_bin % s" % (j + 1), zmean_g)
        block.put_double("pk_to_xi", "zmean_gm_bin % s" % (j + 1), zmean_gm)

        block.put_double("pk_to_xi", "zmean_gg_mm_bin % s" % (j + 1), zmean_g)
        block.put_double("pk_to_xi", "zmean_gm_mm_bin % s" % (j + 1), zmean_gm)

    return 0


def cleanup(config):
    pass
