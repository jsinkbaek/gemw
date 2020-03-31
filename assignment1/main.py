import numpy as np
# import astropy as ap
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack, unique, setdiff, Column
from astropy import units as un
import astropy.coordinates as coord
import time as tm
import seaborn as sns

path_to_data = '/data/astronomy/gaia/dr2/'
fnames = []
with open(path_to_data+'fnames', 'r') as f:
    j = 0
    for line in f:
        fnames.append(line.split())
        j += 1
        if j == 8:
            break
# fnames.append(['GaiaSource_6714230465835878784_6917528443525529728.csv'])


i = 0
lfnames = len(fnames)


for name in fnames:
    print('i = ', i)

    # Read a data file
    F = Table.read(path_to_data+name[0], include_names=['parallax', 'ra', 'dec', 'pmra',
                                                        'pmdec', 'radial_velocity'],
                   format='csv', guess=False, fast_reader={'chunk_size': 500 * 1000000})

    if i == 0:
        # Initialize arrays
        par = np.ma.empty((F['parallax'].size, lfnames))
        ra = np.ma.empty((F['ra'].size, lfnames))
        dec = np.ma.empty((F['dec'].size, lfnames))
        pmra = np.ma.empty((F['pmra'].size, lfnames))
        pmdec = np.ma.empty((F['pmdec'].size, lfnames))
        rvel = np.ma.empty((F['radial_velocity'].size, lfnames))

        par[:] = np.nan
        ra[:] = np.nan
        dec[:] = np.nan
        pmra[:] = np.nan
        pmdec[:] = np.nan
        rvel[:] = np.nan

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

    # Convert to arrays filled wtih nan values instead of masked array
    par = np.ma.filled(par, fill_value=np.nan)
    ra = np.ma.filled(ra, fill_value=np.nan)
    dec = np.ma.filled(dec, fill_value=np.nan)
    pmra = np.ma.filled(pmra, fill_value=np.nan)
    pmdec = np.ma.filled(pmdec, fill_value=np.nan)
    rvel = np.ma.filled(rvel, fill_value=np.nan)

    print(name[0])
    print(F.info)
    del F
    i += 1

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
dist = 1000/np.abs(par) * un.pc
pmra = pmra * un.mas/un.year
pmdec = pmdec * un.mas/un.year
rvel = rvel * un.km/un.s

# # ICRS objects
icrs = coord.ICRS(ra=ra, dec=dec, distance=dist/1000, pm_ra_cosdec=pmra, pm_dec=pmdec,
                  radial_velocity=rvel)

# Galactocentric frame
gc = coord.Galactocentric(galcen_distance=X_GC_sun, galcen_v_sun=
                          coord.CartesianDifferential([-vX_GC_sun, vY_GC_sun, vZ_GC_sun]),
                          z_sun=Z_GC_sun)

galcen = icrs.transform_to(gc)

# LSR frame
lsr = icrs.transform_to(coord.LSR)

# Stars GC cartesian coordinates
x = galcen.x.to(un.kpc).value   # in kpc
y = galcen.y.to(un.kpc).value   # in kpc
z = galcen.z.to(un.kpc).value   # in kpc
d_gc = np.sqrt(x**2 + y**2 + z**2)  # distance from GC in kpc

# Stars LSR velocities less than 100pc
dist_lsr = dist  # lsr.distance.to(un.pc).value  # in pc, unnecessary since LSR transform does not change coordinate
# reference point
arg_100pc = np.where(dist_lsr < 100)
arg_100pc_extra = np.where(dist.value < 100)
U = lsr.radial_velocity.to(un.km/un.s).value    #
pmra_lsr = lsr.pm_ra_cosdec.to(un.mas/un.yr).value
pmdec_lsr = lsr.pm_dec.to(un.mas/un.yr).value
pm_lsr = [pmra_lsr, pmdec_lsr] * un.mas/un.yr
V, W = (pm_lsr * dist_lsr*un.pc).to(un.km/un.s, un.dimensionless_angles())

dist_100pc = dist_lsr[arg_100pc]
dist_100pc_extra = dist[arg_100pc_extra]
print('dist_lsr.size', dist_lsr.size)
print('dist_100pc.size', dist_100pc.size)
print('extra', dist_100pc_extra.size)
U_100pc = U[arg_100pc]
V_100pc = V[arg_100pc].value
W_100pc = W[arg_100pc].value

print(V_100pc.dtype)

# # Scheming
plt.figure()
plt.plot(np.sort(dist[dist.value/1000 < 100].ravel() / 1000), 'r*')

plt.figure()
plt.plot(x, y, '.', markersize=2)
# sns.set_palette(sns.color_palette('hls', 8))
plt.plot(-X_GC_sun, 0, 'b*', markersize=8)
plt.xlim([-12.5, 7.5])
plt.ylim([-20, 20])
plt.xlabel('X kpc')
plt.ylabel('Y kpc')
plt.show()

plt.figure()
plt.plot(x, z, '.', markersize=2)
plt.xlabel('X kpc')
plt.ylabel('Z kpc')
# plt.show(block=False)

plt.figure()
plt.plot(x, d_gc, 'b.', markersize=2)
plt.plot(y, d_gc, 'r.', markersize=2)
plt.plot(z, d_gc, 'y.', markersize=2)
plt.xlabel('Galactocentric carthesian coordinate in kpc')
plt.ylabel('Distance from galactic centre in kpc')
#plt.xlim([-75, 75])
#plt.ylim([-5, 100])

# plt.figure()
# plt.plot(U_100pc, dist_100pc, '*', markersize=2)
# plt.ylabel('Distance from LSR in parsec')
# plt.xlabel('Velocity component in km/s')

# plt.figure()
# plt.plot(V_100pc, dist_100pc, '*', markersize=2)

# plt.figure()
# plt.plot(W_100pc, dist_100pc, '*', markersize=2)

# Histogram LSR velocities less than 100pc
nbins = 50
u100 = np.copy(U_100pc)
v100 = np.copy(V_100pc)
w100 = np.copy(W_100pc)

plt.figure()
#plt.hist(u100, bins=nbins)

plt.figure()

#plt.hist(v100, bins=nbins)

plt.figure()

#plt.hist(w100, bins=nbins)



