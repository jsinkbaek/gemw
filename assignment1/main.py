import numpy as np
# import astropy as ap
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack, unique, setdiff, Column
from astropy import units as un
import astropy.coordinates as coord
import time as tm

path_to_data = '/data/astronomy/gaia/dr2/'
fnames = []
with open(path_to_data+'fnames', 'r') as f:
    for line in f:
        fnames.append(line.split())
# fnames.append(['GaiaSource_6714230465835878784_6917528443525529728.csv'])


i = 0
lfnames = len(fnames)
for name in fnames:
    # Read a data file
    F = Table.read(path_to_data+name[0], include_names=['parallax', 'ra', 'dec', 'pmra',
                                                        'pmdec', 'radial_velocity'],
                   format='csv')
    if i == 0:
        # Initialize arrays
        par = np.ma.empty((F['parallax'].size, lfnames))
        ra = np.ma.empty((F['ra'].size, lfnames))
        dec = np.ma.empty((F['dec'].size, lfnames))
        pmra = np.ma.empty((F['pmra'].size, lfnames))
        pmdec = np.ma.empty((F['pmdec'].size, lfnames))
        rvel = np.ma.empty((F['radial_velocity'].size, lfnames))

    else:
        # Check if array lengthening must be implemented to accomodate new data
        if F['parallax'].size > par[:, i-1].size:
            print('Warning, array lengthening done, very slow')
            par_temp = np.ma.empty_like(par, shape=(F['parallax'].size, lfnames))
            ra_temp = np.ma.empty_like(par, shape=(F['ra'].size, lfnames))
            dec_temp = np.ma.empty_like(par, shape=(F['dec'].size, lfnames))
            pmra_temp = np.ma.empty_like(par, shape=(F['pmra'].size, lfnames))
            pmdec_temp = np.ma.empty_like(par, shape=(F['pmdec'].size, lfnames))
            rvel_temp = np.ma.empty_like(par, shape=(F['radial_velocity'].size, lfnames))
            for k in range(0, i-1):
                par_temp[0:par[:, k].size, k] = par[:, k]
                ra_temp[0:ra[:, k].size, k] = ra[:, k]
                dec_temp[0:dec[:, k].size, k] = dec[:, k]
                pmra_temp[0:pmra[:, k].size, k] = pmra[:, k]
                pmdec_temp[0:pmdec[:, k].size, k] = pmdec[:, k]
                rvel_temp[0:rvel[:, k].size, k] = rvel[:, k]
            par = np.ma.copy(par_temp)
            ra = np.ma.copy(ra_temp)
            dec = np.ma.copy(dec_temp)
            pmra = np.ma.copy(pmra_temp)
            rvel = np.ma.copy(rvel_temp)

            del par_temp, ra_temp, dec_temp, pmra_temp, rvel_temp

    # Place data from file into arrays
    par[0:F['parallax'].size, i] = F['parallax']
    ra[0:F['ra'].size, i] = F['ra']
    dec[0:F['dec'].size, i] = F['dec']
    pmra[0:F['pmra'].size, i] = F['pmra']
    pmdec[0:F['pmdec'].size, i] = F['pmdec']
    rvel[0:F['radial_velocity'].size, i] = F['radial_velocity']

    print(name[0])
    print(F.info)
    del F
    print(par.shape)
    i += 1
    print('i = ', i)

# # Solar parameters wrt Galaxy, LSR and galactic plane
X_GC_sun = 8 * un.kpc
Z_GC_sun = 0.025 * un.kpc
U_LSR = 11.1 * un.km/un.s
V_LSR = 12.24 * un.km/un.s
W_LSR = 7.25 * un.km/un.s
v_cir = 220 * un.km/un.s    # circular velocity of galaxy at solar radius

# Galactocentric velocity of Sun
vX_GC_sun = -U_LSR
vY_GC_sun = V_LSR + v_cir
vZ_GC_sun = W_LSR

# # Coordinate units input, see units.pdf for reference
ra = ra * un.degree
dec = dec * un.degree
dist = 1/np.ma.abs(par) * un.parsec
pmra = pmra * un.mas/un.year
pmdec = pmdec * un.mas/un.year
rvel = rvel * un.km/un.s

# # ICRS objects
icrs = coord.ICRS(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec,
                  radial_velocity=rvel)

# Galactocentric frame
gc = coord.Galactocentric(galcen_distance=X_GC_sun, galcen_v_sun=
                          coord.CartesianDifferential([-vX_GC_sun, vY_GC_sun, vZ_GC_sun]),
                          z_sun=Z_GC_sun)

galcen = icrs.transform_to(gc)

# Stars GC cartesian coordinates
x = galcen.x.to(un.kpc).value   # in kpc
y = galcen.y.to(un.kpc).value   # in kpc

plt.plot(x, y, '.')
plt.plot(-X_GC_sun, 0, 'b*', markersize=8)
plt.xlim([-20, 15])
plt.ylim([-20, 20])
plt.xlabel('kpc')
plt.ylabel('kpc')
plt.show()
