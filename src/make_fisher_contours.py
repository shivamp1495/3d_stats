import numpy as np
import pdb
import matplotlib
import sys, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from configobj import ConfigObj
from configparser import ConfigParser

# Color = ['#0072b1', '#009d73', '#d45e00', 'k', 'grey', 'yellow']

colors = ['red','blue','orange','magenta']
# linestyles = ['-','--','-.',':']

linestyles = ['-','-','-','-']

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True


# font = {'size': 18} 
# matplotlib.rc('font', **font)
# plt.rc('text', usetex=False)
# plt.rc('font', family='serif')


# Get ellipse given 2D covariance matrix
def get_ellipse(cov, contour_levels=[1]):
    sigmax = np.sqrt(cov[0, 0])
    sigmay = np.sqrt(cov[1, 1])
    sigmaxy = cov[1, 0]

    # Silly
    all_sigma_list = np.array([1, 2, 3])
    all_alpha_list = np.array([1.52, 2.48, 3.44])
    alpha_list = np.zeros(len(contour_levels))
    for ii in xrange(0, len(contour_levels)):
        match = np.where(all_sigma_list == contour_levels[ii])[0]
        alpha_list[ii] = all_alpha_list[match]

    num_points = 5000
    rot_points = np.zeros((len(alpha_list), 2, 2 * num_points))

    for ai in xrange(0, len(alpha_list)):
        alpha = alpha_list[ai]
        alphasquare = alpha ** 2.

        # Rotation angle
        if sigmax != sigmay:
            theta = 0.5 * np.arctan((2. * sigmaxy / (sigmax ** 2. - sigmay ** 2.)))
        if sigmaxy == 0.:
            theta = 0.
        if sigmaxy != 0. and sigmax == sigmay:
            # is this correct?
            theta = np.pi / 4.

        # Determine major and minor axes
        eigval, eigvec = np.linalg.eig(cov)
        major = np.sqrt(eigval[0])
        minor = np.sqrt(eigval[1])
        if sigmax > sigmay:
            asquare = major ** 2.
            bsquare = minor ** 2.
        if sigmax <= sigmay:
            asquare = minor ** 2.
            bsquare = major ** 2.
            theta += np.pi / 2.

        # Get ellipse defined by pts
        xx = np.linspace(-0.99999 * np.sqrt(alphasquare * asquare), 0.99999 * np.sqrt(alphasquare * asquare),
                         num=num_points)
        yy = np.sqrt(alphasquare * bsquare * (1 - (xx ** 2.) / (alphasquare * asquare)))
        minusy = -yy
        points = np.vstack((np.append(xx, -xx), np.append(yy, minusy)))

        # Rotation matrix
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for pi in xrange(0, 2 * num_points):
            rot_points[ai, :, pi] = np.dot(rot_matrix, points[:, pi])

    return rot_points


def get_points_from_fisher(cov, indices, sigma_levels):
    cov_reduced = cov[indices, :]
    cov_reduced = cov_reduced[:, indices]

    points = get_ellipse(cov_reduced, contour_levels=sigma_levels)
    return points


def plot_contours(F_mat_wpriors_array, fid_values_array, param_labels_array, sigma_levels,labels_array=None, params_toplot=None,param_names_array=None):

    if params_toplot is not None:
        nparam = len(params_toplot)
    else:
        nparam = F_mat_wpriors_array[0].shape[0]

    figx = nparam * 3.
    fig, ax = plt.subplots(nparam, nparam, figsize=(figx, figx))

    labels_text = r''


    for sj in range(len(param_labels_array)):
        F_mat_wpriors = F_mat_wpriors_array[sj]
        fid_values = fid_values_array[sj]
        param_labels = param_labels_array[sj]
        param_names = param_names_array[sj]
        label = labels_array[sj]
        labels_text += label + '\n'

        param_cov = np.linalg.inv(F_mat_wpriors)

        if params_toplot is not None:
            ind_toplot = []
            for j in range(len(params_toplot)):
                ind_toplot.append(param_names.index(params_toplot[j]))

            ind_toplot = np.array(ind_toplot)

            select_ind_cov = np.array(np.zeros(param_cov.shape), dtype=bool)
            for j1 in range(len(param_names)):
                for j2 in range(len(param_names)):
                    if (j1 in ind_toplot) and (j2 in ind_toplot):
                        select_ind_cov[j1,j2] = True
                    else:
                        select_ind_cov[j1, j2] = False

            # param_cov = param_cov[select_ind_cov]
            param_cov = (param_cov[:, ind_toplot])[ind_toplot, :]
            fid_values = fid_values[ind_toplot]
            param_labels = [param_labels[j] for j in ind_toplot]
            F_mat_wpriors = np.linalg.inv(param_cov)

            # pdb.set_trace()
            # controls plotting range in units of standard deviation
        max_sd = 5.

        # Determine plot limits
        param_ranges = []
        for parami in xrange(0, nparam):
            varparami = param_cov[parami, parami]
            param_ranges.append((fid_values[parami] - max_sd * np.sqrt(varparami),
                                 fid_values[parami] + max_sd * np.sqrt(varparami)))


        # fig.subplots_adjust(hspace = 0., wspace = 0.)

        # rows
        for parami in xrange(0, nparam):
            # cols
            for paramj in xrange(0, nparam):
                # ellipses on lower triangle
                if parami > paramj:
                    points = get_points_from_fisher(param_cov, [parami, paramj], sigma_levels)
                    for ii in xrange(0, len(sigma_levels)):
                        ax[parami, paramj].plot(points[ii, 1, :] + fid_values[paramj],
                                                points[ii, 0, :] + fid_values[parami], lw=1,label=label,color = colors[sj],ls=linestyles[sj])
                        if sj == nparam-1:
                            ax[parami, paramj].set_xlim((param_ranges[paramj][0], param_ranges[paramj][1]))
                            ax[parami, paramj].set_ylim((param_ranges[parami][0], param_ranges[parami][1]))
                # Get rid of upper triangle
                if paramj > parami:
                    if sj == nparam-1:
                        fig.delaxes(ax[parami, paramj])
                # 1d gaussian on diagonal
                if parami == paramj:
                    varparami = param_cov[parami, paramj]
                    xx = np.linspace(param_ranges[parami][0], param_ranges[parami][1], num=100)
                    yy = np.exp(-((xx - fid_values[parami]) ** 2.) / (2. * varparami))
                    if nparam > 1:
                        ax_handle = ax[parami, paramj]
                    else:
                        ax_handle = ax
                    ax_handle.plot(xx, yy, lw=1,label=label,color = colors[sj],ls=linestyles[sj])
                    if sj == nparam - 1:
                        ax_handle.set_xlim((param_ranges[parami][0], param_ranges[parami][1]))
                        ax_handle.set_ylim((0., 1.1 * np.max(yy)))
                        ax_handle.set_yticklabels([])
                        ax_handle.yaxis.set_ticks_position('none')
                        ax_handle.set_yticklabels([])
                if paramj > 0:
                    if sj == nparam - 1:
                        ax[parami, paramj].set_yticklabels([])
                if paramj == 0:
                    if nparam > 1:
                        ax_handle = ax[parami, paramj]
                    else:
                        ax_handle = ax
                    if sj == nparam - 1:
                        ax_handle.set_ylabel(param_labels[parami], fontsize=13)
                if parami == nparam - 1:
                    if nparam > 1:
                        ax_handle = ax[parami, paramj]
                    else:
                        ax_handle = ax
                    if sj == nparam - 1:
                        ax_handle.set_xlabel(param_labels[paramj], fontsize=13)

    # pdb.set_trace()
    handles, labels = ax[0][0].get_legend_handles_labels()


    # ax_list = fig.axes
    #
    # naxis_total = len(ax_list)
    # nfigures_oneaxis = int(np.sqrt(naxis_total))
    #
    # ax_list[nfigures_oneaxis ].text(0.8, 0.9, labels_text,
    #                                    horizontalalignment='center',
    #                                    verticalalignment='center',
    #                                    transform=ax_list[nfigures_oneaxis - 2].transAxes, fontsize=18,
    #                                    color='k')


    fig.legend(handles, labels, loc='upper right',fontsize=16,frameon=False)

    return fig


def get_fid_vals_paramnames(fisher_mat_file, values_file):
    config_val = ConfigObj(values_file)
    infile = open(fisher_mat_file, 'r')

    first_line = infile.readline()
    first_line_split = first_line.split()

    first_element_split = list(first_line_split[0])
    first_line_split[0] = ''.join(first_element_split[1:])

    # pdb.set_trace()

    fid_values = []
    params_names = []

    for j in range(len(first_line_split)):
        ln = list(first_line_split[j])
        ln_cut_ind = ln.index('-')
        section_name = ''.join(ln[:ln_cut_ind])
        vary_param_name_split = ln[ln_cut_ind + 2:]

        if ('bias' in section_name) or ('pk_to_xi' in section_name):
            e_ind = vary_param_name_split.index('e')
            vary_param_name_split[e_ind] = 'E'
        vary_param_name = ''.join(vary_param_name_split)

        params_vals = (config_val[section_name][vary_param_name]).split()
        if len(params_vals) == 1:
            param_value = float(params_vals[0])

        if len(params_vals) == 3:
            param_value = float(params_vals[1])

        fid_values.append(param_value)
        params_names.append(vary_param_name)

    return np.array(fid_values), params_names


latex_names_dict = {'omega_m': r'$\Omega_m$', 'omega_b': r'$\Omega_b$', 'omega_k': r'$\Omega_k$', 'h0': r'$h$',
                    'n_s': r'$n_s$', 'sigma8_input': r'$\sigma_8$', 'A_s': r'$A_s$', 'b1E_bin1': r'$b_1^E({\rm bin1})$',
                    'b1E_bin2': r'$b_1^E({\rm bin2})$', 'b1E_bin3': r'$b_1^E({\rm bin3})$',
                    'b1E_bin4': r'$b_1^E({\rm bin4})$', 'b1E_bin5': r'$b_1^E({\rm bin5})$',
                    'b2E_bin1': r'$b_2^E({\rm bin1})$', 'b2E_bin2': r'$b_2^E({\rm bin2})$',
                    'b2E_bin3': r'$b_2^E({\rm bin3})$', 'b2E_bin4': r'$b_2^E({\rm bin4})$',
                    'b2E_bin5': r'$b_2^E({\rm bin5})$', 'bkE_bin1': r'$b_k^E({\rm bin1})$',
                    'bkE_bin2': r'$b_k^E({\rm bin2})$', 'bkE_bin3': r'$b_k^E({\rm bin3})$',
                    'bkE_bin4': r'$b_k^E({\rm bin4})$', 'bkE_bin5': r'$b_k^E({\rm bin5})$',
                    'bsE_bin1': r'$b_s^E({\rm bin1})$', 'bsE_bin2': r'$b_s^E({\rm bin2})$',
                    'bsE_bin3': r'$b_s^E({\rm bin3})$', 'bsE_bin4': r'$b_s^E({\rm bin4})$',
                    'bsE_bin5': r'$b_s^E({\rm bin5})$', 'b3nlE_bin1': r'$b_{\rm 3nl}^E({\rm bin1})$',
                    'b3nlE_bin2': r'$b_{\rm 3nl}^E({\rm bin2})$', 'b3nlE_bin3': r'$b_{\rm 3nl}^E({\rm bin3})$',
                    'b3nlE_bin4': r'$b_{\rm 3nl}^E({\rm bin4})$', 'b3nlE_bin5': r'$b_{\rm 3nl}^E({\rm bin5})$'}


def makefisher(ini_files,params_toplot=None,labels_array=None,sigma_levels = np.array([1.])):
    fisher_mat_array, fid_vals_array, param_labels_array,params_names_array = [],[],[],[]

    for ini_file in ini_files:
        config_run = ConfigObj(ini_file)
        fisher_mat_file = config_run['output']['filename']
        values_file = config_run['pipeline']['values']

        fisher_mat = np.loadtxt(fisher_mat_file)
        fid_vals, params_names = get_fid_vals_paramnames(fisher_mat_file, values_file)

        param_labels = []
        for param in params_names:
            param_labels.append(latex_names_dict[param])
        fisher_mat_array.append(fisher_mat)
        fid_vals_array.append(fid_vals)
        param_labels_array.append(param_labels)
        params_names_array.append(params_names)

    fig = plot_contours(fisher_mat_array, fid_vals_array, param_labels_array,sigma_levels,labels_array=labels_array,params_toplot=params_toplot,param_names_array=params_names_array)
    return fig



if __name__ == '__main__':

    fisherinifile = sys.argv[1]
    config_fish = ConfigObj(fisherinifile, unrepr=True)

    inifiles = config_fish['DEFAULT']['inifiles']
    savefile = config_fish['DEFAULT']['savefile']
    labels_array = config_fish['DEFAULT']['labels_array']

    params_toplot = config_fish['DEFAULT']['params_toplot']

    fig = makefisher(inifiles,params_toplot=params_toplot,labels_array=labels_array)

    fig.savefig(savefile)

