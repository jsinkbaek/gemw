import numpy as np
# import astropy as ap
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table, join, vstack, unique, setdiff, Column
from astropy import units as un
import astropy.coordinates as coord
from mpl_toolkits.mplot3d import Axes3D
import time as tm
import seaborn as sns


def uvw_histograms(u, v, w, d, bins=50, dlims=((0, 100), (100, 500)), ncols=2, brange=(-500, 500)):
    # Collapse into 1-D array
    u = u.ravel()
    v = v.ravel()
    w = w.ravel()
    d = d.ravel()

    # Find NaN values
    nan_list = np.isnan(u) + np.isnan(v) + np.isnan(w) + np.isnan(d)

    # Remove NaN values
    u, v, w, d = u[~nan_list], v[~nan_list], w[~nan_list], d[~nan_list]

    fig1, ax1 = plt.subplots(nrows=int(np.ceil(len(dlims) / ncols)), ncols=ncols)
    fig2, ax2 = plt.subplots(nrows=int(np.ceil(len(dlims) / ncols)), ncols=ncols)
    fig3, ax3 = plt.subplots(nrows=int(np.ceil(len(dlims) / ncols)), ncols=ncols)
    num = range(0, len(dlims))
    for (i, ax1i, ax2i, ax3i) in zip(num, ax1.flat, ax2.flat, ax3.flat):
        lim = dlims[i]
        idx = np.where((d > lim[0]) & (d < lim[1]))
        u_i, v_i, w_i = u[idx], v[idx], w[idx]
        ax1i.hist(u_i, bins=bins, range=brange)
        ax2i.hist(v_i, bins=bins, range=brange)
        ax3i.hist(w_i, bins=bins, range=brange)

        ax1i.set_xlabel('(' + str(lim[0]) + ', ' + str(lim[1]) + ') pc')
        ax2i.set_xlabel('(' + str(lim[0]) + ', ' + str(lim[1]) + ') pc')
        ax3i.set_xlabel('(' + str(lim[0]) + ', ' + str(lim[1]) + ') pc')

    fig1.text(0.5, 0.04, 'U (km/s)', ha='center', va='center')
    fig2.text(0.5, 0.04, 'V (km/s)', ha='center', va='center')
    fig3.text(0.5, 0.04, 'W (km/s)', ha='center', va='center')

    plt.show()


def phi_distr_plot(phi, bins=50):
    # Collapse into 1-D array
    phi = phi.ravel()

    # Find NaN values
    nan_list = np.isnan(phi)

    # Remove nan values
    phi = phi[~nan_list]

    # Plot histogram
    plt.hist(phi, bins=bins)
    plt.xlabel('GC cylindric angle [rad]')
    plt.ylabel('Counts')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.show()


def vt_plot(vT, R):
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    # Collapse into 1-D array
    vT = vT.ravel()
    R = R.ravel()

    # Find NaN values
    nan_list = np.isnan(vT)
    nan_list += np.isnan(R)

    # Remove NaN values
    vT, R = vT[~nan_list], R[~nan_list]

    # Find values within limits
    lim = (0, 30)
    idx = np.where((R > lim[0]) & (R < lim[1]))
    R_lim = R[idx]
    vT_lim = vT[idx]
    sT = np.abs(vT_lim)

    # Interpolate
    rlin = np.linspace(R_lim.min(), R_lim.max(), 1000)
    itp = interp1d(R_lim, sT)

    # Smooth
    window_size, poly_order = 101, 3
    vt_sg = savgol_filter(itp(rlin), window_size, poly_order)

    # Plot
    plt.plot(R_lim, sT, '*', markersize=2, rasterized=True)
    plt.plot(rlin, vt_sg, 'k')
    plt.xlabel('Galactocentric radius [kpc]')
    plt.ylabel('Rotation speed [km/s]')
    plt.ylim([-10, 2000])
    plt.show()
    # plt.savefig('/data/astronomy/gaia/dr2/vt.pdf', dpi=300)
    # plt.close()


def cylinder_plot(x, y, z, x_0, y_0, z_0, func, cyl_rad=5, cyl_h=30, ncols=2, mirror=True, negative=False,
                  hist_cyl_h=((-10, -2), (-2, -0.5), (-0.5, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.5), (0.1, 2), (2, 10)),
                  brange=(-500, 500),
                  bins=50):
    matplotlib.rcParams.update({'font.size': 14})
    # Collapse into 1-D array
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    func = func.ravel()

    # Find NaN values
    nan_list = np.isnan(x)
    nan_list += np.isnan(y)
    nan_list += np.isnan(z)
    nan_list += np.isnan(func)

    # Remove NaN values
    x, y, z, func = x[~nan_list], y[~nan_list], z[~nan_list], func[~nan_list]

    # # Add figure for first 2 plots
    fig = plt.figure()
    # ax = fig.add_subplot(111)

    # # 3D Galaxy cylinder plot
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_3d.scatter(x, y, z, c='b', s=10, rasterized=True, alpha=0.01)

    # Cylinder
    x_cyl = np.linspace(-cyl_rad, cyl_rad, 100)
    if mirror is True:
        z_cyl = np.linspace(-cyl_h, cyl_h, 100)
    else:
        z_cyl = np.linspace(0, cyl_h, 100)
    Xc, Zc = np.meshgrid(x_cyl, z_cyl)
    Yc = np.sqrt(1-Xc**2)

    # Plot draw parameters
    rcount = 200
    ccount = 200
    ax_3d.plot_surface(Xc+x_0, Yc+y_0, Zc+z_0, alpha=0.7, rcount=rcount, ccount=ccount, color='r', rasterized=True)
    ax_3d.plot_surface(Xc+x_0,-Yc+y_0, Zc+z_0, alpha=0.7, rcount=rcount, ccount=ccount, color='r', rasterized=True)

    # Labels and limits
    ax_3d.set_xlabel('Y [kpc]')
    ax_3d.set_ylabel('X [kpc]')
    ax_3d.set_zlabel('Z [kpc]')
    ax_3d.set_xlim([-10, 10])
    ax_3d.set_ylim([-10, 10])
    ax_3d.set_zlim([-20, 20])
    ax_3d.view_init(elev=30, azim=30)

    # Cylinder center point
    ax_3d.scatter(x_0, y_0, z_0, c='y', marker='X', s=22, depthshade=False)

    # # Func plot
    ax_func = fig.add_subplot(122)
    x_c, y_c, z_c = x - x_0, y - y_0, z - z_0
    if mirror is True:
        condition = (z_c < cyl_h) & (z_c > -cyl_h) & (x_c**2 + y_c**2 < cyl_rad**2)
    elif negative is True:
        condition = (z_c > cyl_h) & (z_c <= 0) & (x_c ** 2 + y_c ** 2 < cyl_rad ** 2)
    else:
        condition = (z_c < cyl_h) & (z_c >= 0) & (x_c ** 2 + y_c ** 2 < cyl_rad ** 2)
    idx = np.where(condition)
    ax_func.plot(z_c[idx], func[idx], 'r*', rasterized=True)

    # Labels and limits
    ax_func.set_xlabel('Z from Sun [kpc]')
    ax_func.set_ylabel('W [km/s]')
    ax_func.set_ylim([-2000, 2000])

    # fig.savefig('/data/astronomy/gaia/dr2/cylinder.pdf', dpi=400)

    # # Histograms
    fig_h, ax_h = plt.subplots(nrows=int(np.ceil(len(hist_cyl_h) / ncols)), ncols=ncols, sharex=True)
    for i, axi in enumerate(ax_h.flat):
        condition = (z_c < hist_cyl_h[i][1]) & (z_c > hist_cyl_h[i][0]) & (x_c**2 + y_c**2 < cyl_rad**2)
        idx = np.where(condition)
        axi.hist(func[idx], range=brange, bins=bins)
        axi.yaxis.set_visible(True)
        axi.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axi.set_title('(' + str(hist_cyl_h[i][0]) + ', ' + str(hist_cyl_h[i][1]) + ') kpc')

    fig_h.text(0.5, 0.04, 'W (km/s)', ha='center', va='center')
    fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')

    # fig_h.savefig('/data/astronomy/gaia/dr2/zhist.pdf')
    plt.show()

    # return ax_3d, ax_func, ax_h


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
del i

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
icrs = coord.ICRS(ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec,
                  radial_velocity=rvel)

# # Galactocentric frame
gc = coord.Galactocentric(galcen_distance=X_GC_sun, galcen_v_sun=
                          coord.CartesianDifferential([-vX_GC_sun, vY_GC_sun, vZ_GC_sun]),
                          z_sun=Z_GC_sun)

galcen = icrs.transform_to(gc)

# # LSR frame
lsr = icrs.transform_to(coord.LSR)

# # GC cartesian coordinates
x = galcen.x.to(un.kpc).value   # in kpc
y = galcen.y.to(un.kpc).value   # in kpc
z = galcen.z.to(un.kpc).value   # in kpc
d_gc = np.sqrt(x**2 + y**2 + z**2)  # distance from GC in kpc

# # GC cylindrical coordinates
galcen.set_representation_cls(coord.CylindricalRepresentation, s=coord.CylindricalDifferential)

R_kpc = galcen.rho.to(un.kpc).value
phi_rad = galcen.phi.to(un.rad).value
z_kpc = galcen.z.to(un.kpc).value

vR_kms = galcen.d_rho.to(un.km/un.s).value
vT_kms = -(galcen.d_phi.to(un.rad/un.s)*galcen.rho.to(un.km) / (1.*un.radian)).value
vz_kms = galcen.d_z.to(un.km/un.s).value

# # Converting back to LSR velocity components
U = vR_kms
V = vT_kms - v_cir.value
W = vz_kms

# # Stars LSR velocities less than 100pc
dist_lsr = dist.value  # lsr.distance.to(un.pc).value  # in pc, unnecessary since LSR transform
arg_100pc = np.where(dist_lsr < 100)
# U = lsr.radial_velocity.to(un.km/un.s).value    #
# pmra_lsr = lsr.pm_ra_cosdec.to(un.mas/un.yr).value
# pmdec_lsr = lsr.pm_dec.to(un.mas/un.yr).value
# pm_lsr = [pmra_lsr, pmdec_lsr] * un.mas/un.yr

# V, W = (pm_lsr * dist_lsr*un.pc).to(un.km/un.s, un.dimensionless_angles())
# V = V.value
# W = W.value

# dist_100pc = dist_lsr[arg_100pc]
# U_100pc = U[arg_100pc]
# V_100pc = V[arg_100pc]
# W_100pc = W[arg_100pc]

# # Scheming
plt.ioff()

if False:
    plt.figure()
    plt.plot(x, y, '.', markersize=2, rasterized=True)
    # sns.set_palette(sns.color_palette('hls', 8))
    plt.plot(-X_GC_sun, 0, 'b*', markersize=8)
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.xlabel('X kpc')
    plt.ylabel('Y kpc')
    plt.savefig('/data/astronomy/gaia/dr2/xy.pdf', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(x, z, '.', markersize=2, rasterized=True)
    # sns.set_palette(sns.color_palette('hls', 8))
    plt.plot(-X_GC_sun, Z_GC_sun, 'b*', markersize=8)
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.xlabel('X kpc')
    plt.ylabel('Z kpc')
    plt.savefig('/data/astronomy/gaia/dr2/xz.pdf', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(x, y, '.', markersize=2, rasterized=True)
    # sns.set_palette(sns.color_palette('hls', 8))
    plt.plot(-X_GC_sun, 0, 'b*', markersize=8)
    plt.xlim([-300, 700])
    plt.ylim([-500, 500])
    plt.xlabel('X kpc')
    plt.ylabel('Y kpc')
    plt.savefig('/data/astronomy/gaia/dr2/xy2.pdf', dpi=300)
    plt.close()

    plt.figure()
    plt.plot(dist_lsr / 1000, d_gc, 'r*', markersize=2, rasterized=True)
    plt.xlabel('Parallax distance in kpc')
    plt.ylabel('Cartesian distance in kpc')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.savefig('/data/astronomy/gaia/dr2/dist.pdf', dpi=400)
    plt.close()

    phi_distr_plot(phi_rad, bins=50)

    uvw_histograms(U, V, W, dist.value, bins=100, brange=(-350, 350), dlims=((0, 50), (50, 100),
                                                                             (100, 200), (200, 400),
                                                                             (400, 1000), (1000, 80000)))
vt_plot(vT_kms, R_kpc)
# cylinder_plot(x, y, z, -X_GC_sun.value, -0, -Z_GC_sun.value, W)










