#%%

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import smplotlib

# %%
# Open the fits file
data = fits.open('rho_Ui_mu_ns_ne.fits')

#%%

plt.plot(data[0].data)
plt.yscale('log')
plt.ylabel(data[0].header['EXTNAME'])

plt.show()
# %%

plt.plot(data[1].data)
plt.ylabel(data[1].header['EXTNAME'])

plt.show()

# %%

plt.plot(data[2].data)
plt.ylabel(data[2].header['EXTNAME'])

plt.show()

# %%
for extension in data[3].data:
    plt.plot(extension)
plt.ylabel(data[3].header['EXTNAME'])

# plt.yscale('log')
plt.show()
# %%
plt.plot(data[4].data)
plt.ylabel(data[4].header['EXTNAME'])
plt.yscale('log')

plt.show()

# %%
