import anesthetic.read
import matplotlib.pyplot as plt
import pesummary
from pesummary.io import read
import h5py
import warnings
warnings.filterwarnings('ignore')
from pesummary.utils.utils import logger
import logging
logger.setLevel(logging.CRITICAL)
import pandas as pd
import bilby
from anesthetic import NestedSamples, read_chains, MCMCSamples
import anesthetic
import numpy as np
from anesthetic import read_chains, make_1d_axes, make_2d_axes
from scipy.stats import gaussian_kde
from scipy import integrate
from scipy.integrate import simpson
import matplotlib as mpl

# Enable LaTeX-style font rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']

file_name = 'GW170817_GWTC-1.hdf5'
label = 'ExtendedDataFig2_Reweighted_a'

#Sample1 = anesthetic.read_chains('GW170817_Canonical_40_200.csv')
# Sample2 = anesthetic.read_chains('GW170817_flat_z_prior_40_200_points.csv')
#Sample3 = anesthetic.read_chains('GW170817_250_uncertainty_40_200_3.csv')

#Sample2 = anesthetic.read_chains('GW170817_flat_z_prior_50_225_3.csv')

Sample1 = anesthetic.read_chains('GW170817_Canonical_40_200_4.csv')
Sample2 = anesthetic.read_chains('GW170817_flat_z_prior_40_200_points.csv')
Sample3 = anesthetic.read_chains('GW170817_250_uncertainty_40_200_4.csv')
Sample4 = anesthetic.read_chains('GW170817_flat_z_prior_reweighted_to_dLPrior.csv')

H0_1 = Sample4['H_0']
fig, axes = make_2d_axes(params=['H_0'], figsize=(10, 6))
ax = axes["H_0"]['H_0']
axes.tick_params(grid_alpha=0)
ax.set_xlim(25,250)
fig.tight_layout()
ax.set_xticks(np.arange(25, 251, 25))
Sample1.plot_2d(axes, kind='kde_1d', color='black', alpha=0.8, label='Canonical')
Sample2.plot_2d(axes, kind='kde_1d', color='b', alpha=0.8, label='Flat z prior (sampled)')
Sample4.plot_2d(axes, kind='kde_1d', color='m', alpha=0.8, label='Canonical (reweighted flat z prior)')
Sample3.plot_2d(axes, kind='kde_1d', color='r', alpha=0.8, label='$250$ km s$^{-1}$ Uncertainty')

#ax.set_yticks(np.arange(0, 1.1, 0.1))
axes.axlines({'H_0': [63.722832826195514, 87.65858205914093]}, ls='--', color='black')
#axes.axlines({'H_0': [59.38955679392043, 119.48937712151974]}, ls=':', color='black')

# Sample2 (new sample prior)
axes.axlines({'H_0': [63.58350560107286, 122.51287949476476]}, ls='--', color='b')
# axes.axlines({'H_0': [62.136333571192765, 176.14896448788102]}, ls=':', color='b')


# Sample4 (reweighted samples)
axes.axlines({'H_0': [62.93082029114855, 91.29625352480802]}, ls='--', color='m')
# axes.axlines({'H_0': [64.57700263876588, 100.2333190510584]}, ls='--', color='m')
#axes.axlines({'H_0': [61.26771125795835, 133.16827288409073]}, ls=':', color='b')

axes.axlines({'H_0': [62.16091346244711, 91.50206127530998]}, ls='--', color='r')
#axes.axlines({'H_0': [55.54329479158159, 128.03227039230248]}, ls=':', color='r')


axes.axspans({'H_0': [65.7, 68.2]}, edgecolor='none', alpha=0.3, upper=False, color='#6DE6AC')
axes.axspans({'H_0': [69.76, 76.72]}, edgecolor='none', alpha=0.3, upper=False, color='#F19851')
axes.axspans({'H_0': [66.93-0.62, 66.93+0.62]}, edgecolor='none', alpha=0.3, upper=False, color='#0CDE79', label = 'Planck')
axes.axspans({'H_0': [73.24-1.74, 73.24+1.74]}, edgecolor='none', alpha=0.3, upper=False, color='#E87317', label = 'SHoES')

for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
    
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_ylabel('$P(H_0)$ (km$^{-1}$ s Mpc)')

kde = gaussian_kde(H0_1)
x_eval = np.linspace(min(H0_1) - 1, max(H0_1) + 1, 10000)
pdf_values = kde(x_eval)
map_value = x_eval[np.argmax(pdf_values)]
print(f'MAP value of H0_1: {map_value}')

ax.legend(frameon=False)
plt.savefig(f'{label}.pdf')
plt.show()
#map_df = pd.DataFrame({'MAP_H_0': [map_value]})
#map_df.to_csv(f'{label}_MAP.csv', index=False)
