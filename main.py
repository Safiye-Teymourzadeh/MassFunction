import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import astropy.constants as const



# constants and units:
cosmoUNIT = FlatLambdaCDM(
    H0=70.0 * u.km / u.s / u.Mpc,
    Om0=0.30)
h = 0.6777
cosmo = cosmoUNIT


z_array = np.arange(0, 7.5, 0.001)                  # for the redshift, from now up to z=7.5 put the intervals of 0.001
d_C = cosmo.comoving_distance(z_array)              # Co-moving distance of the given intervals' redshift. IT IS NOT THE DISTANCE OF THE OBJECTS. in unit os Mpc
dc_mpc = (d_C).value                                # Co-moving distance of the given intervals' redshift in Mega Parsec
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



clu = fits.open('/home/farnoosh/farnoosh/Master_Thesis/Data/eFEDS/efeds_clusters_full_20210814.fits')[1].data
GAMA  = fits.open('/home/farnoosh/farnoosh/Master_Thesis/Data/GAMA/gkvScienceCatv02_mask_stellarMass.fits')[1].data
# PC MPE:
# clu = fits.open('/home/safiye/safiye/data1/eFEDS/efeds_clusters_full_20210814.fits')[1].data
# GAMA  = fits.open('/home/safiye/safiye/data1/GAMA/gkvScienceCatv02_mask_stellarMass.fits')[1].data
print('files are opened')



# based on Driver et al. 2022
Z_min = 0.0
Z_max = 0.1


# properties of galaxies that we want:
#        |mass of the star      |with the redshift    |and a redshift     |science sample    |normalized                                     |the object is inside
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


# get the position of the objects in cartesian coordinate
x_C, y_C, z_C = get_xyz( CLU['RA'],  CLU['DEC'], CLU['z_final'])
x_G, y_G, z_G = get_xyz( GAL['RAcen'],  GAL['Deccen'], GAL['Z'])
dist_C = (x_C**2+ y_C**2 + z_C**2)**0.5
dist_G = (x_G**2+ y_G**2 + z_G**2)**0.5     #Euclidean distance between the position of the galaxies in Cartesian coordinate system


# physical distance of the objects:



from sklearn.neighbors import BallTree
coord_cat_C = np.transpose([x_C, y_C, z_C])
coord_cat_G = np.transpose([x_G, y_G, z_G])
tree_G = BallTree(coord_cat_G)
tree_C = BallTree(coord_cat_C)

#if the different distance between the galaxy and the radius of the cluster is less than 1 mega parsec then is is accepted
Q1, D1 = tree_G.query_radius(coord_cat_C, r=1.7, return_distance = True)

#galaxies that Are in the cluster:
GiC = GAL[np.hstack((Q1))]

#define mass bins in Logarithm of (M_star / M_sun)
mbins = np.arange(8,12,0.1)
x_hist = (mbins[:-1]+mbins[1:])/2
H1 = np.histogram(GAL['logmstar'], bins=mbins)[0]
H2 = np.histogram(GiC['logmstar'], bins=mbins)[0]

#print("H1: galaxy catalouge =", H1)
print('Total number of the Glaxies from sum(H1):', sum(H1))
print('Total number of galaxies that are in the clustrs, sum(H2)):',sum(H2))







# Plot 1:
plt.title("1_ Number of Galaxies and Clusters in each bin based on their mass")
plt.hist(GAL['logmstar'], bins=100, color="slateblue", edgecolor="darkslateblue", label='all galaxies')[0]
plt.hist(GiC['logmstar'], bins=30,  color="turquoise", edgecolor="teal", label='in cluster')[0]
plt.xlabel("Log(M/$M_{\odot}$)")
plt.ylabel("Number")
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('stellar_mass_histogram.png')
plt.clf()




def y_D22(x):
    """
    Richards curve from GAMA based on Table 5, Eq. 2 from Driver et al. 2022
    :param x: log10 of stellar mass limit
    :return: co moving distance  in Mpc
    """
    A = -0.016
    K = 2742.0
    C = 0.9412
    B = 1.1483
    M = 11.815
    nu = 1.691
    y = A + (K - A) / (C + np.e ** (-B * (x - M))) ** (1 / nu)
    return y

def inverted_y_D22(y):
    """
    returns the inverted function of the richards curve
    :param y: comoving distance
    :return: log mass
    """
    A = -0.016
    K = 2742.0
    C = 0.9412
    B = 1.1483
    M = 11.815
    nu = 1.691
    fraction_part = (A - K)/(A - y)
    logarithm_part = np.log(fraction_part**nu - C)
    x = (B * M - logarithm_part) / B
    return x


def plot_adjusted_richards_curve(log_stellar_mass, logM_min, logM_max):
    adjusted_richards_curve = np.zeros(len(log_stellar_mass))
    for i, log_mass in enumerate(log_stellar_mass):
        if log_mass < logM_min:
            adjusted_richards_curve[i] = y_D22(logM_min)
        elif log_mass > logM_max:
            adjusted_richards_curve[i] = y_D22(logM_max)
        else:
            adjusted_richards_curve[i] = y_D22(log_mass)

    return adjusted_richards_curve

def find_logMmin_andmax(z_min, z_max):
    comoving_min, comoving_max = cosmo.comoving_distance(z_min).value, cosmo.comoving_distance(z_max).value

    logm_min, logm_max = inverted_y_D22(comoving_min), inverted_y_D22(comoving_max)

    return logm_min, logm_max


x_array = np.arange(0,12,0.1)
yARR = y_D22(x_array)

logm_min, logm_max = find_logMmin_andmax(z_min=0.0013, z_max=0.1)
adjusted_richards = plot_adjusted_richards_curve(log_stellar_mass=x_array, logM_min=logm_min, logM_max=logm_max)


# volume of the galaxies (V_G) and the cluster (V_C)
# co-moving volume of the mass limits based on their co-moving distances
v_G = 4 / 3 * np.pi * (y_D22(GAL['logmstar'])) ** 3
v_C = 4 / 3 * np.pi * (y_D22(GiC['logmstar'])) ** 3

# make an array of galaxy masses => then divide them to the volume of the all galaxies
Hv_G = np.histogram(GAL['logmstar'], bins=mbins, weights=np.ones_like(GAL['logmstar']) / total_volume_G.value)[0]
Hv_C = np.histogram(GiC['logmstar'], bins=mbins, weights=np.ones_like(GiC['logmstar']) / total_volume_C.value)[0]




# Plot 2:  H1_V = (H1 / v_G)    number density of galaxies per co-moving volume
plt.title("2_ Number-Density of galaxies and clusters in each bin based on their mass")
plt.step(x_hist, Hv_G, color="slateblue", label='all galaxies')
plt.step(x_hist, Hv_C, color="turquoise", label='in cluster')
plt.yscale('log')
plt.xlabel("Log(M/$M_{\odot}$)")
plt.ylabel("Number Density(Mpc^-3)")
plt.show()
plt.savefig('stellar_mass_histogram_over_volume.png')
plt.clf()


# Plot 3: CoMoving Distance based on stellar masses
plt.plot(GAL['logmstar'], dist_G, 'o', markersize=1)
plt.xlabel('Stellar Mass (Log(M/$M_{\odot}$)  $h_{70}^{-2}$)')
plt.ylabel('CoMoving Distance(Mpc $h_{70}^{-1}$)')
plt.plot(x_array, adjusted_richards, 'k--')
plt.title('3_ CoMoving Distance vs Stellar Mass for All Data')
plt.xlim(0,12)
plt.ylim(0, 450)
plt.show()
plt.savefig('CoMoving Distance vs Stellar Mass')
plt.clf()




#Mass to Light ratio ( r_band):
Ms_over_Msun = GAL['mstar']
# three elements are non-positive, so we need to remove them, length= 23097-3 = 23094
mask = Ms_over_Msun <= 0
non_positive_elements = Ms_over_Msun[mask]
remove_indices = np.where(mask)[0]
Ms_over_Msun_corrected = np.delete(Ms_over_Msun, remove_indices)


flux = GAL['flux_rt'] * u.Jy
flux_r_corrected = np.delete(flux, remove_indices)

CoMov_Gal_objects = GAL['comovingdist']
CoMov_Gal_objects_corrected = np.delete(CoMov_Gal_objects, remove_indices)

Z_corrected = np.delete(GAL['Z'], remove_indices)


# physical distance:
d_G_physical = CoMov_Gal_objects_corrected * (1 + Z_corrected) * u.Mpc

# luminosity distance:
d_G_Lum_corrected = cosmo.luminosity_distance(Z_corrected)

# luminosity
lum_G = 4 * np.pi * (d_G_Lum_corrected.to(u.m))**2 * (flux_r_corrected.to(u.W/u.m**2/u.Hz))*u.Hz
lum_G_over_lSun = lum_G / const.L_sun

ML_ratio_solar = np.log10 ( Ms_over_Msun_corrected / lum_G_over_lSun )


mass_mask = Ms_over_Msun_corrected < 1e10
Ms_over_Msun_corrected_new = Ms_over_Msun_corrected[mass_mask]
lum_G_over_lSun_new = lum_G_over_lSun[mass_mask]
ML_ratio_massLimitted = np.log10(Ms_over_Msun_corrected_new / lum_G_over_lSun_new)

average_ML_ratio = np.mean(ML_ratio_solar)
print("Average mass to light ratio: ", average_ML_ratio)

average_ML_ratio_limitted = np.mean(ML_ratio_massLimitted)
print("Average mass to light ratio for limitted mass: ", average_ML_ratio_limitted)


#Plot 5:
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_ML = scaler.fit_transform(ML_ratio_solar.reshape(-1, 1))
scaled_MLRlim = scaler.fit_transform(ML_ratio_massLimitted.reshape(-1, 1))

plt.title("5_ Mass-to-Light ratio.")
plt.hist(scaled_ML, bins=100 , color="green", edgecolor="blue", label= "all data-3\navg= 14.76")[0]
plt.hist(scaled_MLRlim, bins=100,  color="pink", edgecolor="red", label='M < 10**10 $M_{\odot}$\navg= 14.68')[0]
plt.xlabel("Log(M/L) [$M_{\odot}$/$L_{\odot}$]")
plt.ylabel("Number")
plt.legend()
plt.show()
plt.savefig('Mass-to-Light.png')
plt.clf()


