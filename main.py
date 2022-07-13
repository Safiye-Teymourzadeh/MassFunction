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
Om0 = 0.308900
omega_lambda_0 = 1 - 0.3089
cosmo = cosmoUNIT


z_array = np.arange(0, 7.5, 0.001)                  # for the redshift, from now up to z=7.5 put the intervals of 0.001
d_C = cosmo.comoving_distance(z_array)              # Co-moving distance with the given redshift: between 0 to 7.5 with the intervals of 0.001)
dc_mpc = (d_C).value                                # Co-moving distance in mega parsec
dc_interpolation = interp1d(z_array, dc_mpc)        # find the Co-moving distance for each interval of the redshift that we defined


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


        #clu = fits.open (os.path.join(os.environ['HOME'], '/home/farnoosh/MassFunctionST/data/eFEDS/efeds_clusters_full_20210814.fits'))[1].data
        #print(len(RA), 'length RA in clustr')
Z_min = 0.1
Z_max = 0.3
clu = fits.open('/home/farnoosh/MassFunctionST/data/eFEDS/efeds_clusters_full_20210814.fits')[1].data           # read the cluster catalogue
GAMA  = fits.open('/home/farnoosh/MassFunctionST/data/GAMA/gkvScienceCatv02_mask_stellarMass.fits')[1].data     # read th galaxy catalogue

# properties of galaxies that we want:
#        |mass of the star      |with the redshift    |and a redshift     |sience sample     |normalized                                     |the object is inside
#        |must be bigger than   |of bigger than 0.1   |smaller than 0.3   |better than 7     |redshift                                       |of the GAMA survey
#        |mass of the sun                                                                    |quality
z_sel = ( GAMA['logmstar']>0 ) & (GAMA['Z']> Z_min) & (GAMA['Z']< Z_max) & (GAMA['SC']>=7) & (GAMA['NQ']>2) & ( GAMA['duplicate']==False ) & ( GAMA['mask']==False ) & ( GAMA['starmask']==False ) & ( GAMA['Tycho20Vmag10']==False ) & ( GAMA['Tycho210Vmag11'] == False ) & ( GAMA['Tycho211Vmag115']==False )& ( GAMA['Tycho2115Vmag12']==False )

#make a table from the aproperiate galaxies:
GAL = Table(GAMA[z_sel])

# properties of clusters that we want:
c_ok = (clu['z_final']>0.12) & (clu['z_final']<0.28)
CLU = Table(clu[c_ok])
print('files are opened')       #just to check that files are opened
print(len(GAL), 'number of acceptable galaxies')
print(len(CLU), 'number of acceptable clusters')


# get the position of the objects in cartesian coordinate and with the radius of = redshift
x_C, y_C, z_C = get_xyz( CLU['RA'],  CLU['DEC'], CLU['z_final'])
x_G, y_G, z_G = get_xyz( GAL['RAcen'],  GAL['Deccen'], GAL['Z'])

from sklearn.neighbors import BallTree
coord_cat_C = np.transpose([x_C, y_C, z_C]) #attach kon the coordinate of the clusters (Cartesian)
coord_cat_G = np.transpose([x_G, y_G, z_G])
tree_G = BallTree(coord_cat_G)
tree_C = BallTree(coord_cat_C)


#if the different distance between the galaxy and the radius of the cluster is less than 1 mega parsec then is is accepted
Q1, D1 = tree_G.query_radius(coord_cat_C, r=1, return_distance = True)


#galaxies that Are in the cluster:
GiC = GAL[np.hstack((Q1))]
print('Catalogue ID of the galaxies that are in the clusters', GiC)


mbins = np.arange(8,12,0.1)
x_hist = mbins[:-1]+0.1/2.
H1 = np.histogram(GAL['logmstar'], bins=mbins)[0]
Print("H1: ", H1)
plt.title("Histogram of the mass stars in the galaxy catalogue")
plt.show()


H2 = np.histogram(GiC['logmstar'], bins=mbins)[0]
print("H2", H2)
plt.title("Histogram of mass stars in the clustrs")
plt.hist(GAL['logmstar'], bins=mbins)[0]
plt.show()

# compute volumes


#v1 =
#v2 =

# Vmax


# figure showing H1/volume vs x_hist and same for H2

# histogram of stellar mss and magnitudes or fluxes
#gkv = "/home/farnoosh/Desktop/Master_Thesis/Data/GAMA"
#p2_IN = os.path.join(gkv, 'gkvScienceCatv02_mask_stellarMass.fits')
#GAMA    = Table.read(p2_IN)
#print('GAMA length:',len(GAMA))


#z_sel = ( GAMA['logmstar']>0 ) & (GAMA['Z']> Z_min) & (GAMA['Z']< Z_max) & (GAMA['SC']>=7) & (GAMA['NQ']>2) & ( GAMA['duplicate']==False ) & ( GAMA['mask']==False ) & ( GAMA['starmask']==False ) & ( GAMA['Tycho20Vmag10']==False ) & ( GAMA['Tycho210Vmag11'] == False ) & ( GAMA['Tycho211Vmag115']==False )& ( GAMA['Tycho2115Vmag12']==False )
#GAMA = GAMA[z_sel]
#print('we have this many' ,len(GAMA), 'of galaxies with the redshift between ', Z_min, "and" , Z_max, ', and the the stellar masses are >0')

