import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

font = {'size': 18}
matplotlib.rc('font', **font)
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

sc = np.array([2,3,4,8])

npoints = np.array([160,140,120,90])

chi2_b1e = np.array([2371.11,794.81,630.181,151.447])

redchi2_b1e = np.array([15.297,5.887,5.48,1.782])

chi2_b1b2bsb3nl = np.array([1299.52,131.901,98.846,69.427])

redchi2_b1b2bsb3nl = np.array([8.663,1.015,0.895,0.868])

chi2_b1b2bsb3nlbk = np.array([1290.96,128.63,95.951,66.268])

redchi2_b1b2bsb3nlbk = np.array([8.903,1.029,0.914,0.884])

chi2_b1b2bsb3nl_0p5 = np.array([1299.572,132.937,97.596,70.456])

redchi2_b1b2bsb3nl_0p5 = np.array([8.664,1.023,0.887,0.881])

chi2_dict = {'b1e':chi2_b1e,'b1b2bsb3nl':chi2_b1b2bsb3nl,'b1b2bsb3nlbk':chi2_b1b2bsb3nlbk,'b1b2bsb3nl_0p5':chi2_b1b2bsb3nl_0p5}

redchi2_dict = {'b1e':redchi2_b1e,'b1b2bsb3nl':redchi2_b1b2bsb3nl,'b1b2bsb3nlbk':redchi2_b1b2bsb3nlbk,'b1b2bsb3nl_0p5':redchi2_b1b2bsb3nl_0p5}

nvar_dict = {'b1e':5,'b1b2bsb3nl':10,'b1b2bsb3nlbk':15,'b1b2bsb3nl_0p5':10}

labels_dict = {'b1e':'Linear Bias','b1b2bsb3nl':'1 Loop PT','b1b2bsb3nlbk':'1Loop PT+bk','b1b2bsb3nl_0p5':'1Loop PT + bk + b2(b1)'}

pt_types = chi2_dict.keys()

sigma_redchi2_dict = {}

for pt in pt_types:
    sigma_redchi2_dict[pt] = np.sqrt(2. / (npoints - nvar_dict[pt]))

fig, ax = plt.subplots(1, 2)
# fig, ax = plt.subplots(2, 1, figsize=(9, 24), sharex=True)
fig.set_size_inches((18, 7))
colors = ['r', 'b', 'k', 'orange', 'magenta']
ls_all = ['-', '--', '-.', ':', ':']
k = 0
for pt in pt_types:
    print pt
    ax[0].plot(sc, chi2_dict[pt], color=colors[k], label=labels_dict[pt], ls=ls_all[k])
    ax[1].plot(sc, redchi2_dict[pt], color=colors[k], label=labels_dict[pt], ls=ls_all[k])
    ax[1].plot(sc, 1 - sigma_redchi2_dict[pt], ls=ls_all[k])
    ax[1].plot(sc, 1 + sigma_redchi2_dict[pt], ls=ls_all[k])
    ax[1].fill_between(sc, 1 - sigma_redchi2_dict[pt], 1 + sigma_redchi2_dict[pt], alpha=0.05)
    k += 1



xticks = [2,3,4,8]
ax[0].set_xticks(xticks)
ax[1].set_xticks(xticks)
labels = [xticks[i] for i, t in enumerate(xticks)]
ax[0].set_xticklabels(labels)
ax[1].set_xticklabels(labels)
ax[1].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='minor', labelsize=15)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[0].tick_params(axis='both', which='minor', labelsize=15)

ax[0].set_yscale('log')
ax[1].set_yscale('log')

ax[0].set_xscale('log')
ax[1].set_xscale('log')

ax[0].legend(fontsize=17, frameon=False, loc='upper right')
ax[0].set_xlabel(r'Scale Cut for $\xi_{gg}$ and $\xi_{gm}$')
ax[1].set_xlabel(r'Scale Cut for $\xi_{gg}$ and $\xi_{gm}$')
ax[1].set_ylabel(r'Reduced $\chi^2$')
ax[0].set_ylabel(r'$\chi^2$')
plt.tight_layout()
fig.savefig('chi2_redchi2_comp_log.png')




