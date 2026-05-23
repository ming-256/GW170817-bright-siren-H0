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
from scipy.integrate import simps
import matplotlib as mpl

# Enable LaTeX-style font rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']

file_name = 'GW170817_GWTC-1.hdf5'
label = 'ExtendedDataFig2_40_140'

Sample1 = anesthetic.read_chains('Z_GW170817_Heterodyned_NRTidal_Constrained_Bilby_H0_50_140_2.csv')
Sample2 = anesthetic.read_chains('GW170817_flat_z_prior.csv')
Sample2 = anesthetic.read_chains('GW170817_flat_z_prior_points.csv')
Sample3 = anesthetic.read_chains('GW170817_250_uncertainty.csv')

H0_1 = Sample2['H_0']
fig, axes = make_2d_axes(params=['H_0'], figsize=(10, 6))
ax = axes["H_0"]['H_0']
axes.tick_params(grid_alpha=0)
ax.set_xlim(50,140)
map_value = 69.84009427510232
fig.tight_layout()
ax.set_xticks(np.arange(50, 141, 10))
Sample1.plot_2d(axes, kind='kde_1d', color='black', alpha=0.8, label='Canonical')
Sample2.plot_2d(axes, kind='kde_1d', color='blue', alpha=0.8, label='Flat z prior')
Sample3.plot_2d(axes, kind='kde_1d', color='red', alpha=0.8, label='$250$ km s$^{-1}$ Uncertainty')
# Canonical
axes.axlines({'H_0': [63.4125506648017, 85.3083951666764]}, ls='--', color='black')
#axes.axlines({'H_0': [59.244207520944926, 114.48835150851752]}, ls=':', color='black')
# Flat z prior
axes.axlines({'H_0': [64.5948013244207, 104.31068161712444]}, ls='--', color='blue')
#axes.axlines({'H_0': [63.66324933231993, 135.8381226301914]}, ls=':', color='blue')
# 250 km s$^{-1}$ Uncertainty
axes.axlines({'H_0': [62.047217956400615, 90.72774262886465]}, ls='--', color='red')
#axes.axlines({'H_0': [56.828251932785754, 123.40167438237252]}, ls=':', color='red')

axes.axspans({'H_0': [65.7, 68.2]}, edgecolor='none', alpha=0.3, upper=False, color='#6DE6AC')
axes.axspans({'H_0': [69.76, 76.72]}, edgecolor='none', alpha=0.3, upper=False, color='#F19851')
axes.axspans({'H_0': [66.93-0.62, 66.93+0.62]}, edgecolor='none', alpha=0.3, upper=False, color='#0CDE79', label = 'Planck$^{10}$')
axes.axspans({'H_0': [73.24-1.74, 73.24+1.74]}, edgecolor='none', alpha=0.3, upper=False, color='#E87317', label = 'SHoES$^{11}$')
ax.set_xlabel('$H_0$ (km s$^{-1}$ Mpc$^{-1}$)')
ax.set_ylabel('$P(H_0)$ (km s$^{-1}$ Mpc$^{-1}$)')
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
