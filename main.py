import numpy as np
import os
import astropy.io.fits as fits
import scipy
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import cosmolopy.distance as cd
from astropy.constants import c

#cosmoUNIT = FlatLambdaCDM(H0=67.74 * u.km / u.s / u.Mpc, omega_M_0=0.308900)
h = 0.6774
H0=67.74 * u.km / u.s / u.Mpc
omega_M_0 = 0.308900
omega_lambda_0 = 0.7
#cosmo = cosmoUNIT



clst = fits.open('/home/farnoosh/Desktop/Master_Thesis/Data/eFEDS/eFEDS_clusters_V3.2.fits')[1]
gal  = fits.open('/home/farnoosh/Desktop/Master_Thesis/Data/GAMA/SpecObjv27.fits')[1]


ra_clst = clst.data['RA'] * (np.pi/180)      # read the right-ascension of each cluster and convert it into radian
ra_gal = gal.data['RA'] * (np.pi/180)        # read the right-ascension of each galaxy and convert it into radian

dec_clst = clst.data['DEC'] * (np.pi/180)    # read the declination of each cluster and convert it into radian
dec_gal = gal.data['DEC'] * (np.pi/180)      # read the declination of each galaxy and convert it into radian

z_clst = clst.data['z']                      # read the redshift of each cluster
z_gal = gal.data['Z']                        # read the redshift of each galaxy
d_gal = gal.data['DIST']                     # read the distance to the galaxy _ in Mega parsec


# find the co-moving distance to each galaxy with the distance of d_gal  _ in Mega parsec:
cosmo = {'omega_M_0': 0.308900, 'omega_lambda_0': 0.7, 'h': 0.6774}
cosmo = cd.set_omega_k_0(cosmo)
cd_gal = []
for i in range(len(d_gal)):
    cd_gal.append(cd.comoving_distance_transverse(d_gal[i], **cosmo))



# find the co-moving distance to the center of each cluster using its redshift_ in Mega parsec:
cosmo = {'omega_M_0': 0.308900, 'omega_lambda_0': 0.7, 'h': 0.6774}
cosmo = cd.set_omega_k_0(cosmo)
cd_clst = []
for i in range(len(z_clst)):
    cd_clst.append(cd.comoving_distance_transverse(z_clst[i], **cosmo))




# positions in Cartesian
x_clst = cd_clst * np.cos(ra_clst)
y_clst = cd_clst * np.sin(dec_clst)

x_gal = cd_gal * np.cos(ra_gal)
y_gal = cd_gal * np.sin(dec_gal)

R_clst = np.sin(clst.data['R_SNR_MAX']) * cd_clst           # radius of the cluster  _ in Mega parsec



# check if a galaxy is in a cluster:
selected_galaxy= []
for galaxy in range(len(y_gal)):
    for cluster in range(len(y_clst)):
        if ((x_clst[cluster] - x_gal[galaxy])**2 + (y_clst[cluster] - y_gal[galaxy])**2 + (cd_clst[cluster] - cd_gal[galaxy])**2) <= R_clst[cluster]**2:
            selected_galaxy.append(cluster)
            print('Galaxy number: ', galaxy, 'in cluster number: ', cluster)



## to read the mass of the galaxies which are in the cluster
#for i in range (len(selected_galaxy)):
    #print(i, selected_galaxy[i], gal.data['mass'][i])




