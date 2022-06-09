import numpy as np
import os, sys
import astropy.io.fits as fits
from astropy.table import Table
import scipy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import cosmolopy.distance as cd
from astropy.constants import c
from scipy.interpolate import interp1d


# constants and units:
cosmoUNIT = FlatLambdaCDM(H0=67.74 * u.km / u.s / u.Mpc, Om0=0.308900)
h = 0.6774
H0=67.74 * u.km / u.s / u.Mpc
omega_M_0 = 0.308900
omega_lambda_0 = 1 - 0.3089
cosmo = cosmoUNIT


z_array = np.arange(0, 7.5, 0.001)                  # for the redshift, from now up to z=7.5 put the intervals of 0.001
d_C = cosmo.comoving_distance(z_array)              # to find the co-moving distance with the given redshift
dc_mpc = (d_C).value
dc_interpolation = interp1d(z_array, dc_mpc)


# Coordinate conversion:
def get_xyz(RARA, DECDEC, ZZZZ):
    # from RA, DE and r => polar coordinate:
    phi   = ( RARA   - 180 ) * np.pi / 180.
    theta = ( DECDEC + 90 ) * np.pi / 180.
    rr    = dc_interpolation( ZZZZ )
    # convert polar coordinate to cartesian:
    xx = rr * np.cos( phi   ) * np.sin( theta )
    yy = rr * np.sin( phi   ) * np.sin( theta )
    zz = rr * np.cos( theta )
    return np.array(list(xx)), np.array(list(yy)), np.array(list(zz))


clu = fits.open('/home/farnoosh/Desktop/Master_Thesis/Data/eFEDS/efeds_clusters_full_20210814.fits')[1].data
gal  = fits.open('/home/farnoosh/Desktop/Master_Thesis/Data/GAMA/gkvScienceCatv02_mask_stellarMass.fits')[1].data

g_ok = (gal['NQ']>2) & (gal['Z']>0.1) & (gal['Z']<0.3)          # only use the galaxies with these conditions (NQ:normalized redshift quality)
GAL = Table(gal[g_ok])


c_ok = (clu['z']>0.12) & (clu['z']<0.28)                        # only use the clusters that are fine
CLU = Table(clu[c_ok])
print('files are opened')
print(len(GAL), 'number of acceptable galaxies')
#print(CLU, 'clusters')
print(len(CLU), 'number of acceptable clusters')

x_C, y_C, z_C = get_xyz( CLU['RA'],  CLU['DEC'], CLU['z'])
x_G, y_G, z_G = get_xyz( GAL['RA'],  GAL['DEC'], GAL['Z'])

from sklearn.neighbors import BallTree
coord_cat_C = np.transpose([x_C, y_C, z_C])
coord_cat_G = np.transpose([x_G, y_G, z_G])
tree_G = BallTree(coord_cat_G)
tree_C = BallTree(coord_cat_C)

Q1, D1 = tree_G.query_radius(coord_cat_C, r=1, return_distance = True)

GiC = GAL[np.hstack((Q1))]                          # galaxies in the cluster
#print('galaxies in the cluster, CatID:',GiC)

# histogram of stellar mss and magnitudes or fluxes
gkv = "/home/farnoosh/Desktop/Master_Thesis/Data/GAMA"
p2_IN = os.path.join(gkv, 'gkvScienceCatv02_mask_stellarMass.fits')
GAMA    = Table.read(p2_IN)
print('GAMA length:',len(GAMA))

Z_min = 0.1
Z_max = 0.3
z_sel = ( GAMA['logmstar']>0 ) & (GAMA['Z']> Z_min) & (GAMA['Z']< Z_max) & (GAMA['SC']>=7) & (GAMA['NQ']>2) & ( GAMA['duplicate']==False ) & ( GAMA['mask']==False ) & ( GAMA['starmask']==False ) & ( GAMA['Tycho20Vmag10']==False ) & ( GAMA['Tycho210Vmag11'] == False ) & ( GAMA['Tycho211Vmag115']==False )& ( GAMA['Tycho2115Vmag12']==False )
GAMA = GAMA[z_sel]
print('we have this many' ,len(GAMA), 'of galaxies with the redshift between ', Z_min, "and" , Z_max, ', and the the stellar masses are >0')
