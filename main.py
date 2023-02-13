import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# constants and units:
cosmoUNIT = FlatLambdaCDM(H0=67.74 * u.km / u.s / u.Mpc, Om0=0.308900)
h = 0.6774
H0=67.74 * u.km / u.s / u.Mpc
Om0 = 0.308900
omega_lambda_0 = 1 - 0.3089
cosmo = cosmoUNIT


z_array = np.arange(0, 7.5, 0.001)                  # for the redshift, from now up to z=7.5 put the intervals of 0.001
d_C = cosmo.comoving_distance(z_array)              # Co-moving distance with the given redshift: between 0 and 7.5 with the intervals of 0.001)
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



clu = fits.open('/home/safiye/safiye/data1/eFEDS/efeds_clusters_full_20210814.fits')[1].data           # read the cluster catalogue
GAMA  = fits.open('/home/safiye/safiye/data1/GAMA/gkvScienceCatv02_mask_stellarMass.fits')[1].data     # read th galaxy catalogue
print('files are opened')

Z_min = 0.1
Z_max = 0.3
# properties of galaxies that we want:
#        |mass of the star      |with the redshift    |and a redshift     |sience sample     |normalized                                     |the object is inside
#        |must be bigger than   |of bigger than 0.1   |smaller than 0.3   |better than 7     |redshift                                       |of the GAMA survey
#        |mass of the sun                                                                    |quality
z_sel = ( GAMA['logmstar']>0 ) & (GAMA['Z']> Z_min) & (GAMA['Z']< Z_max) & (GAMA['SC']>=7) & (GAMA['NQ']>2) & ( GAMA['duplicate']==False ) & ( GAMA['mask']==False ) & ( GAMA['starmask']==False ) & ( GAMA['Tycho20Vmag10']==False ) & ( GAMA['Tycho210Vmag11'] == False ) & ( GAMA['Tycho211Vmag115']==False )& ( GAMA['Tycho2115Vmag12']==False )

f_sky = 60/(129600/np.pi)
total_volume_G = f_sky * (cosmo.comoving_volume(Z_max) - cosmo.comoving_volume(Z_min))
total_volume_C = f_sky * (cosmo.comoving_volume(Z_max-0.02) - cosmo.comoving_volume(Z_min+0.02))

#make a table from the aproperiate galaxies:
GAL = Table(GAMA[z_sel])

# properties of clusters that we want:
c_ok = (clu['z_final']>Z_min+0.02) & (clu['z_final']<Z_max-0.02)
CLU = Table(clu[c_ok])
print('number of acceptable galaxies:', len(GAL))
print('number of acceptable clusters:', len(CLU))


# get the position of the objects in cartesian coordinate and with the radius of redshift*CoMovingDistance
x_C, y_C, z_C = get_xyz( CLU['RA'],  CLU['DEC'], CLU['z_final'])
x_G, y_G, z_G = get_xyz( GAL['RAcen'],  GAL['Deccen'], GAL['Z'])
dist_C = (x_C**2+ y_C**2 + z_C**2)**0.5
dist_G = (x_G**2+ y_G**2 + z_G**2)**0.5

from sklearn.neighbors import BallTree
coord_cat_C = np.transpose([x_C, y_C, z_C]) #attach kon the coordinate of the clusters (Cartesian)
coord_cat_G = np.transpose([x_G, y_G, z_G])
tree_G = BallTree(coord_cat_G)
tree_C = BallTree(coord_cat_C)

#if the different distance between the galaxy and the radius of the cluster is less than 1 mega parsec then is is accepted
Q1, D1 = tree_G.query_radius(coord_cat_C, r=1.7, return_distance = True)

#galaxies that Are in the cluster:
GiC = GAL[np.hstack((Q1))]

#define mass bins in Logarithm of (M_star / M_sun)
mbins = np.arange(8,13,0.1)
x_hist = (mbins[:-1]+mbins[1:])/2
H1 = np.histogram(GAL['logmstar'], bins=mbins)[0]
H2 = np.histogram(GiC['logmstar'], bins=mbins)[0]

#print("H1: galaxy catalouge =", H1)
print('Total number of the Glaxies from sum(H1):', sum(H1))
print('Total number of galaxies that are in the clustrs, sum(H2)):',sum(H2))

# Plot1:
plt.title("Number of Galaxies and Clusters in each bin based on their mass")
plt.hist(GAL['logmstar'], bins=mbins, color="slateblue", edgecolor="darkslateblue", label='all galaxies')[0]
plt.hist(GiC['logmstar'], bins=mbins, color="turquoise", edgecolor="teal", label='in cluster')[0]
plt.xlabel("Log(M/$M_{\odot}$’)")
plt.ylabel("Number")
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('stellar_mass_histogram.png')
plt.clf()


# compute volumes
# Eq.2 of Driver et al. 2022        Y(x): Y=co-moving dist limit & x=mass limit    => co-moving distant limit of mass limit
def y_D22(x):
    A = -0.016
    K = 2742.0
    C = 0.9412
    B = 1.1483
    M = 11.815
    nu = 1.691
    y = A + (K - A) / (C + np.e ** (B * (x - M))) ** (1 / nu)
    return y


# volume of the galaxies (V_G) and the cluster (V_C)
# co-moving volume of the mass limits based on their co-moving diatances
v_G = 4 / 3 * np.pi * (y_D22(GAL['logmstar'])) ** 3
v_C = 4 / 3 * np.pi * (y_D22(GiC['logmstar'])) ** 3

# make an array of galaxy masses => then divide them to the volume of the all galaxies
Hv_G = np.histogram(GAL['logmstar'], bins=mbins, weights=np.ones_like(GAL['logmstar']) / total_volume_G.value)[0]
Hv_C = np.histogram(GiC['logmstar'], bins=mbins, weights=np.ones_like(GiC['logmstar']) / total_volume_C.value)[0]

# Plot tw0:  H1_V = (H1 / v_G)    number density of galaxies per co-moving volume
plt.title(" Number-Density of galaxies and clusters in each bin based on their mass")
plt.step(x_hist, Hv_G, color="slateblue", label='all galaxies')
plt.step(x_hist, Hv_C, color="turquoise", label='in cluster')
plt.yscale('log')
plt.xlabel("Log(M/$M_{\odot}$’)")
plt.ylabel("Number Density(Mpc^-3)")
plt.show()
plt.savefig('stellar_mass_histogram_over_volume.png')
plt.clf()

# Plot three: CoMoving Distance based on stellar masses
plt.plot(GAL['logmstar'], dist_G, 'o', markersize=1)
plt.xlabel('Stellar Mass (Log(M/$M_{\odot}$’))')
plt.ylabel('CoMoving Distance(Mpc)')
plt.title('CoMoving Distance vs Stellar Mass')
plt.show()


# Plot four: limited CoMoving Distance based on stellar masses
plt.plot(GAL['logmstar'], np.log(dist_G), 'o', markersize=0.03)
plt.xlabel('Stellar Mass (Log(M/$M_{\odot}$’))')
plt.ylabel('CoMoving Distance(Mpc)')
plt.title('CoMoving Distance vs Stellar Mass')
plt.savefig('CoMoving Dist vs Stellar Mass.png')
plt.xlim(8, 12)
plt.show()
plt.clf()
