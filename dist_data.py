import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
import gala.coordinates as gc


# Distance data

# Load table containing RRL data
data_dist_rrl_table = Table.read(fits.open('gd1-RRL.fits'))

# Convert to stream coordinates
data_dist_rrl_coords_icrs = coord.SkyCoord(ra=np.array(data_dist_rrl_table['ra'])*u.deg, dec=np.array(data_dist_rrl_table['dec'])*u.deg, frame='icrs')
data_dist_rrl_coords_stream = data_dist_rrl_coords_icrs.transform_to('gd1koposov10')
data_dist_rrl_coords_stream = np.array([data_dist_rrl_coords_stream.phi1.value, data_dist_rrl_coords_stream.phi2.value])

# Select three RRL closest to stream
data_dist_rrl_coords_stream = data_dist_rrl_coords_stream[:,[0,1,2]]


# Absolute magnitude calculated from Mabs_G=0.32*[Fe/H]+1.11, with GD-1's [Fe/H]=-2.3
Mabs_G = 0.374

# Calculate extinction
EBV_sfd = np.array(data_dist_rrl_table['EBV_sfd'])
ag = 2.27*EBV_sfd

# Apparent magnitude
phot_g_mean_mag = np.array(data_dist_rrl_table['phot_g_mean_mag'])

# Calculate distance from absolute and apparent magnitudes and extinction
dist_rrl = 10**(0.2 * (phot_g_mean_mag - ag - Mabs_G + 5))*1e-3


# Create array with phi1 and distances of RRL
data_dist_rrl = np.array([data_dist_rrl_coords_stream[0], dist_rrl[[0,1,2]]])


# Load table containing BHB data
data_dist_bhb_table = Table.read(fits.open('gd1-BHB.fits'))

# Convert to stream coordinates
data_dist_bhb_coords_icrs = coord.SkyCoord(ra=np.array(data_dist_bhb_table['ra'])*u.deg, dec=np.array(data_dist_bhb_table['dec'])*u.deg, frame='icrs')
data_dist_bhb_coords_stream = data_dist_bhb_coords_icrs.transform_to('gd1koposov10')
data_dist_bhb_coords_stream = np.array([data_dist_bhb_coords_stream.phi1.value, data_dist_bhb_coords_stream.phi2.value])


# Calculate distances
mapp_g = np.array(data_dist_bhb_table['g0'])
mapp_r = np.array(data_dist_bhb_table['r0'])

gr = np.array(mapp_g - mapp_r)

par_gr = [0.397623, -0.391735, 2.72942, 29.1128, 113.569]
mabs_g = par_gr[0]+par_gr[1]*gr+par_gr[2]*gr**2+par_gr[3]*gr**3+par_gr[4]*gr**4

dist_bhb = 10**(0.2*(mapp_g-mabs_g+5)) * 1e-3

# Create array with phi1 and distances of BHBs, excluding one off-track value
data_dist_bhb = np.array([data_dist_bhb_coords_stream[0], dist_bhb])[:,:-1]



# plot RRL and BHB distances
fig, ax = plt.subplots()

ax.scatter(data_dist_bhb[0], data_dist_bhb[1], s=10, marker='.', c='tab:blue')
ax.scatter(data_dist_rrl[0], data_dist_rrl[1], s=10, marker='.', c='tab:orange')

ax.set_xlabel('$\phi_1 [^\circ]$')
ax.set_ylabel('Distance [kpc]')

plt.show()
plt.close()

