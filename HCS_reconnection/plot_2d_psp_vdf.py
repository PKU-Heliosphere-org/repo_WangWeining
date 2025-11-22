#%%
'''BY ZIQI WU @ 2025/05/22, BASED ON CODES BY DIE DUAN'''
import pyspedas
import os
import sys
os.environ["CDF_LIB"] = "D:/cdf3.8.0_64bit_VS2015/lib"
sys.path.append("D:/Research/Codes/Hybrid-vpic/HybridVPIC-main/reconnection/PSP_Data_Analysis_WZQ")
import cdflib
import os
import astropy.constants as constant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import scipy.interpolate
import scipy.spatial
import pytplot
# MY MODULES
from load_read_psp_data import *
from utils import *
from calc_psp_data import *
from plot_body_positions import get_rlonlat_psp_carr, get_rlonlat_solo_carr, get_rlonlat_earth_carr
#%%

### USER INPUT BEGIN ###
SC_dir = 'D:/Research/Data/PSP_data/'
pyspedas.psp.config.CONFIG['local_data_dir'] = SC_dir

# load SPI and MAG data for plotting
psp_trange = ['2022-02-25/12:20', '2022-02-25/12:40']
psp_trange = ['2022-02-17/13:48','2022-02-18/01:40']
psp_trange = ['2022-12-12/03:00','2022-12-12/12:00']
psp_trange = ['2022-06-02/17:00','2022-06-02/18:00']
psp_trange = ['2021-08-10/00:00','2021-08-10/02:30']
# psp_trange = ['2024-03-29/23:00','2024-03-29/23:30']
# psp_trange = ['2024-06-29/23:30','2024-06-30/01:00']
# psp_trange = ['2022-02-25/12:20','2022-02-25/12:40']
# psp_trange = ['2022-09-06/17:00','2022-09-06/18:00']
# psp_trange = ['2022-02-17/13:48','2022-02-17/23:59']
# psp_trange = ['2023-09-27/19:20','2023-09-27/20:30']
### USER INPUT END ###

t_range = psp_trange
# print(pyspedas.get_data_dir())
B_vars = pyspedas.psp.fields(trange=t_range, datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True, no_update=True)
spi_vars = pyspedas.psp.spi(trange=t_range, datatype='sf00_l3_mom', level='l3',
                            time_clip=True, no_update=True)
# B_vars=pytplot.cdf_to_tplot("D:/Research/Data/PSP_data/psp_fld_l2_mag_rtn_4_sa_per_cyc_20210810_v02.cdf")
# spi_vars=pytplot.cdf_to_tplot("D:/Research/Data/PSP_data/psp_swp_spe_sf0_l3_pad_20210810_v04.cdf")
print(B_vars, spi_vars)
print(pytplot.tplot_names())
# import pytplot
# import astropy.constants as const
# from scipy.interpolate import interp1d

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def B_2_n_interp(Btime, Bx, By, Bz, n_time):
    from scipy.interpolate import interp1d
    Bx_interp = interp1d(Btime, Bx, fill_value='extrapolate')
    By_interp = interp1d(Btime, By, fill_value='extrapolate')
    Bz_interp = interp1d(Btime, Bz, fill_value='extrapolate')
    Bx_new = Bx_interp(n_time)
    By_new = By_interp(n_time)
    Bz_new = Bz_interp(n_time)
    return Bx_new, By_new, Bz_new


def griddata_tri(data, x, y):
    cart_temp = np.array([x, y])
    points = np.stack(cart_temp).T
    delaunay = scipy.spatial.Delaunay(points)
    return scipy.interpolate.LinearNDInterpolator(delaunay, data)



# =====INPUT BEGIN=====
beg_dt = datetime(2021,4,29,0)
end_dt = datetime(2021,4,29,12)
beg_dt = datetime(2022,2,25,12,20)
end_dt = datetime(2022,2,25,12,40)
# beg_dt = datetime(2022,9,6,17)
# end_dt = datetime(2022,9,6,18)
# beg_dt = datetime(2022,6,2,17)
# end_dt = datetime(2022,6,2,18)
# beg_dt = datetime(2022,2,17,13,48)
# end_dt = datetime(2022,2,17,23,59)
# beg_dt = datetime(2023,9,27,19,20)
# end_dt = datetime(2023,9,27,20,30)
beg_dt = datetime(2021,8,10,0)
end_dt = datetime(2021,8,10,2,30)
# beg_dt = datetime(2024,3,29,23,00)
# end_dt = datetime(2024,3,29,23,30)
# beg_dt = datetime(2024,6,29,23,30)
# end_dt = datetime(2024,6,30,1,0)
pad_energy_ev = 314
pad_clim = [9., 10.5]
pad_norm_clim = [-1.5, -0.]
i_encounter = 8
mag_type = '4sa'
inst = False
# =====INPUT END=====

beg_dt_str = beg_dt.strftime('%Y%m%dT%H%M%S')
end_dt_str = end_dt.strftime('%Y%m%dT%H%M%S')

print('Time Range: ' + beg_dt_str + '-' + end_dt_str)
#%%
## ------READ SPE PAD-----
epochpade, timebinpade, epochpade, EfluxVsPAE, PitchAngle, Energy_val = read_spe_data(beg_dt, end_dt, ver='04')
# Calculate normalized PAD
norm_EfluxVsPAE = EfluxVsPAE * 0
for i in range(12):
    norm_EfluxVsPAE[:, i, :] = EfluxVsPAE[:, i, :] / np.nansum(EfluxVsPAE, 1)
# Choose energy channel
pad_energy_ind = np.argmin(abs(Energy_val[0, :] - pad_energy_ev))
pad_energy_ev_str = '%.2f' % Energy_val[0, pad_energy_ind]
print('PAD Energy Channel: ' + pad_energy_ev_str + 'eV')
# Choose clim. zmin/max1 for PAD; zmin/max2 for norm_PAD
pad_clim_min = pad_clim[0]
pad_clim_max = pad_clim[1]
pad_norm_clim_min = pad_norm_clim[0]
pad_norm_clim_max = pad_norm_clim[1]

##  -----READ MAG-----
epochmag, Br, Bt, Bn, Babs = read_mag_data(beg_dt, end_dt, mag_type=mag_type)

## -----READ SPI data-----
epochpmom, densp, vp_r, vp_t, vp_n, Tp, EFLUX_VS_PHI_p, PHI_p, T_tensor_p, MagF_inst_p = read_spi_data(beg_dt,
                                                                                                       end_dt,
                                                                                                       species='0',
                                                                                                       is_inst=False)
Babs_epochp = interp_epoch(Babs, epochmag, epochpmom)
Br_epochp = interp_epoch(Br, epochmag, epochpmom)
plasma_beta = calc_plasma_beta(densp, Tp, Babs_epochp)
# %%


# %%
import matplotlib.gridspec as gridspec
from spacepy import pycdf
import spacepy.time as spt
# make vdf inside SB
spi_l2_dir = 'D:/Research/Data/PSP_data/'
spi_name = 'psp_swp_spi_sf00_l2_8dx32ex8a_20210810_v04.cdf'
spi_mom_name = 'psp_swp_spi_sf00_l3_mom_20210810_v04.cdf'
spi_name_2 = ''#'psp_swp_spi_sf00_l2_8dx32ex8a_20240630_v04.cdf'
spi_mom_name_2 = ''#'psp_swp_spi_sf00_l3_mom_20240630_v04.cdf'
# spi_name = 'psp_swp_spi_sf00_l3_mom_20220906_v04.cdf'
spi_file = os.path.join(spi_l2_dir, spi_name)
spi_file_2 = os.path.join(spi_l2_dir, spi_name_2)
data = pycdf.concatCDF([pycdf.CDF(spi_file)])#,pycdf.CDF(spi_file_2)])
# print(data.keys())
epoch = spt.Ticktock(data['Epoch'], 'CDFepoch').ISO
epoch = np.array([datetime.fromisoformat(s) for s in epoch])
vdf_time = epoch#cdflib.cdfepoch.to_datetime(epoch)
data_spi = pycdf.concatCDF([pycdf.CDF(os.path.join(spi_l2_dir, spi_mom_name))])#,pycdf.CDF(os.path.join(spi_l2_dir, spi_mom_name_2))])
print(data_spi.keys())
epochmom_ = data_spi['Epoch']
epochmom_ = spt.Ticktock(epochmom_, 'CDFepoch').ISO
epochmom_ = np.array([datetime.fromisoformat(s) for s in epochmom_])
timebinmom_ = (epochmom_ > beg_dt) & (epochmom_ < end_dt)

pmom_time = epochmom_[timebinmom_]
if pmom_time.dtype == 'O':
    pmom_time = np.array(pmom_time, dtype='datetime64[ns]')

# 转换 vdf_time（假设是 object 数组）
if vdf_time.dtype == 'O':
    vdf_time = np.array(vdf_time, dtype='datetime64[ns]')
print(type(pmom_time[0]))
energybin = data['ENERGY']  # 'UNITS': 'eV'
thetabin = data['THETA']
phibin = data['PHI']
rotmat_sc_inst = data['ROTMAT_SC_INST']
eflux = data['EFLUX']  # Units: 'eV/cm2-s-ster-eV'
speedbin = np.sqrt(2 * energybin * constant.e.value / constant.m_p.value) / 1000  # km/s

nflux = eflux / energybin
vdf = nflux / constant.e.value * constant.m_p.value / speedbin ** 2 / 100  # s^3/m^6
import pytplot

import spacepy
print(spacepy.__version__)
t_pmom=np.array(spt.Ticktock(epochpmom).ISO,dtype=datetime)
print("epochpmom 类型:", type(pmom_time))
print("epoch 类型:", type(epoch))
print(len(pmom_time))
#%%

def plot_vdf_frame(vdftime, i_, frame='rtn', pos=[0, 0], color='r', ):
    ftsize = 15
    # 将下面的原始图代码改为subfigure版本
    fig = plt.figure(dpi=100, figsize=(20, 25), constrained_layout=False)
    # fig,axs = plt.subplots(figsize=(12,10))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    spec = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[0], hspace=0.01)
    spec2 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[1], wspace=0.6, width_ratios=[1, 1, 1, 1, 0.05],
                                             hspace=0.0)
    # spec3 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[2], wspace=0.4, width_ratios=[1, 1, 1, 1],
    #                                          hspace=0.2)
    axs = []
    x_label = 0.04
    y_epoch_mark = 16.5
    r_psp_carr_pmom, lon_psp_carr_pmom, lat_psp_carr_pmom = get_rlonlat_psp_carr(epochpmom, for_psi=False)
   
    i = 0
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochpmom, r_psp_carr_pmom, 'k-', linewidth=1)
    ax2 = axs[i].twinx()
    ax2.plot(epochpmom, np.rad2deg(lon_psp_carr_pmom), 'r-', linewidth=1)
    ax2.set_ylabel('Carrington\n Longitude (deg)', fontsize=ftsize)
    ax2.tick_params(labelsize=20)
    axs[i].set_ylabel('Radial\n Distance (Rs)', fontsize=ftsize)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.7, '(a)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='k', weight='semibold'))
    va0 = Babs_epochp[-1].mean()*1e-9/np.sqrt(densp[-1].mean()*1.6726e-27*4*3.1415*1e-7*1e6)/1e3
    axs[i].tick_params(labelsize=20)
    i = 1
    axs.append(fig.add_subplot(spec[i, :]))
    pos = axs[i].pcolormesh(epochpade, PitchAngle[0][:], np.log10(np.array(norm_EfluxVsPAE[:, :, pad_energy_ind])).T,
                            cmap='jet', vmax=pad_norm_clim_max, vmin=pad_norm_clim_min)
    axs[i].set_ylabel('Pitch\n Angle (deg)', fontsize=ftsize)
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.1, '(b) e-PAD (' + pad_energy_ev_str + ' eV)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='w', weight='semibold'))
    axs[i].tick_params(labelsize=20)

    i = 2
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochmag, Br, 'k-', label='Br', zorder=4)
    axs[i].plot(epochmag, Bt, 'r-', label='Bt', zorder=1)
    axs[i].plot(epochmag, Bn, 'b-', label='Bn', zorder=2)
    axs[i].plot(epochmag, np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2), 'm-', label='|B|', zorder=3)
    axs[i].set_ylabel('B\n(nT)', fontsize=ftsize)
    axs[i].legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0., fontsize=18)
    # axs[i].plot([dt_tmp, dt_tmp], [-400, 400], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.3, '(c)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='k', weight='semibold'))
    axs[i].tick_params(labelsize=20)

    i = 3
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochpmom, vp_r, 'k-')
    # ax5.plot([dt_tmp, dt_tmp], [-3., 3.], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].set_ylabel(r'$V_r (km/s)$', fontsize=ftsize)
    # ax5.set_xlabel('Time', fontsize=8)
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.7, '(d)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='k', weight='semibold'))
    axs[i].tick_params(labelsize=20)

    i = 4
    axs.append(fig.add_subplot(spec[i, :]))

    ax2 = axs[i].twinx()
    ax2.plot(epochpmom, Tp, 'r-', linewidth=1)
    ax2.set_ylabel('$T_p$ \n $(eV)$', color='r', fontsize=ftsize)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    ax2.tick_params(labelsize=20)
    axs[i].plot(epochpmom, densp, 'k-', linewidth=1)
    axs[i].set_ylabel('$N_p$ \n$(cm^{-3})$', fontsize=ftsize)
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].tick_params(axis="x", which='both', direction="in", pad=-15, labelbottom=False)
    plt.text(x_label, 0.7, '(e)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='k', weight='semibold'))
    axs[i].tick_params(labelsize=20)

    i = 5
    axs.append(fig.add_subplot(spec[i, :]))
    axs[i].plot(epochpmom, np.log10(plasma_beta), 'k-')
    # axs[i].plot([dt_tmp, dt_tmp], [-3., 3.], 'r--')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())

    axs[i].set_ylabel(r'$\lg\beta$', fontsize=ftsize)
    axs[i].set_xlabel('Time (UTC)', fontsize=25)
    axs[i].set_xlim([epochpmom[0], epochpmom[-1]])
    plt.text(x_label, 0.7, '(f)', transform=plt.gca().transAxes,
             fontdict=dict(fontsize=20, color='k', weight='semibold'))
    tind_pmom, ttime_pmom = find_nearest(pmom_time, vdftime)
    tind, ttime = find_nearest(vdf_time, vdftime)
    axs[i].axvline(x=ttime, ymax=6, c="blue", linewidth=2, zorder=0, clip_on=False, linestyle='--')
    axs[0].text(ttime, y_epoch_mark, r'$t_1$', fontsize=30)
    B_trange = [np.datetime_as_string((vdf_time[tind] - np.timedelta64(3500000))),
                np.datetime_as_string((vdf_time[tind] + np.timedelta64(3500000)))]
    B_vars = pyspedas.psp.fields(trange=B_trange, datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                                 no_update=True)

    B_vec = pytplot.get(B_vars[0])
    axs[i].tick_params(labelsize=20)
    print(ttime)
    B_ave_rtn = B_vec.y.mean(axis=0)
    B_ave_abs = np.linalg.norm(B_ave_rtn)
    print(B_ave_rtn)
    B_para_rtn = B_ave_rtn / B_ave_abs
    print(B_para_rtn)
    B_perp2_rtn = np.cross(B_para_rtn, np.array([-1, 0, 0]))
    B_perp2_rtn = B_perp2_rtn / np.linalg.norm(B_perp2_rtn)
    print(B_perp2_rtn)
    B_perp1_rtn = np.cross(B_perp2_rtn, B_para_rtn)
    B_perp1_rtn = B_perp1_rtn / np.linalg.norm(B_perp1_rtn)
    print(B_perp1_rtn)
    rotmat_mfa_rtn = np.array([B_para_rtn, B_perp1_rtn, B_perp2_rtn])
    np.matmul(rotmat_mfa_rtn, np.array([1, 0, 0]))
    # calc vxyz in inst, sc and rtn frame
    temp_vdf0 = vdf[tind, :]
    good_ind = (~np.isnan(temp_vdf0))#  & (temp_vdf0 > 0)

    speed = speedbin[tind, good_ind]
    theta = thetabin[tind, good_ind]
    phi = phibin[tind, good_ind]

    vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    vz_inst = speed * np.sin(np.deg2rad(theta))
    v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
    v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

    temp_vdf = temp_vdf0[good_ind]

    vdf_max_ind = temp_vdf.argmax()
    vdf_max_vrtn = v_rtn[:, vdf_max_ind]

    v0_rtn = v_rtn - vdf_max_vrtn[:, None]

    # temp_vdf[vx_inst > -100] = 0
    # temp_vdf[vx_inst > -100] = 0

    # plot in mfa frame
    if frame == 'mfa':
        v0_mfa = np.matmul(rotmat_mfa_rtn, v0_rtn)
        points = v0_mfa.T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        print(v0_mfa.max(axis=1), v0_mfa.min(axis=1))
        grid_vx, grid_vy, grid_vz = np.meshgrid(np.linspace(-400, 400, 100),
                                                np.linspace(-400, 400, 100), np.linspace(-400, 400, 100), indexing='ij')
        grid_vdf_mfa1 = vdf_interp((grid_vx, grid_vy, grid_vz))
        grid_vdf_mfa_max_ind1 = np.unravel_index(np.nanargmax(grid_vdf_mfa1), grid_vdf_mfa1.shape)

        time2 = ttime

        levels = [-11.5, -11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5]
        time = time2.tolist().strftime('%H:%M:%S')

        grid_vdf = grid_vdf_mfa1
        grid_vdf_max_ind = grid_vdf_mfa_max_ind1
        print(grid_vdf_max_ind)
        axs.append(fig.add_subplot(spec2[:, :]))
        axs[i_].contourf(grid_vx[:, grid_vdf_max_ind[1], :], grid_vz[:, grid_vdf_max_ind[1], :],
                         np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]), levels,
                         cmap='jet')
        axs[i_].set_xlabel('$V_\\parallel$ (km/s)', fontsize=ftsize)
        axs[i_].set_ylabel('$V_{\\perp2}$ (km/s)', fontsize=ftsize)
        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((-400, 400))
        axs[i_].set_ylim((-400, 400))
        axs[i_].set_title('@' + time, color=color)
    elif frame == 'rtn':
        temp_vdf = temp_vdf0[good_ind]
        print(pytplot.tplot_names())
        psp_vel = pytplot.get('psp_spi_SC_VEL_RTN_SUN')
        psp_vel_vec = psp_vel.y
        psp_vel = psp_vel_vec.mean(axis=0)
        # make linear delaunay interpolation in rtn frame
        cart_temp = v_rtn
        points = np.stack(cart_temp).T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        # plot vdf in v-vmax frame (eliminate bulk speed)
        levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.75, -6.5, -6.25, -6., -5.75, -5.5,-5.25, -5.]
        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]
        print(vdf_max_vrtn)
        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        # make linear delaunay interpolation in v0_rtn frame
        points = v0_rtn.T
        delaunay = scipy.spatial.Delaunay(points)
        temp_vdf[vx_inst > -100] = 0
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

        grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-800, 800, 200),
                                                np.linspace(-500, 500, 125),
                                                np.linspace(-500, 500, 125), indexing='ij')

        print(grid_vr.shape)
        grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
        grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

        grid_vdf = grid_vdf0
        grid_vdf_max_ind = grid_vdf_max_ind0

        axs.append(fig.add_subplot(spec2[:, 0]))
        psp_vel = psp_vel.value

        pos = axs[i_].contourf(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                              grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                              np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                              levels, cmap='jet', extend='both')
        """
        明天把这里的np.log10(grid_vdf[:, grid_vdf_max_ind[1], :])里面的VDF数据拿来拟合core+beam能谱，可以做一下比较之类的，然后分别选取HDJ内外的结果，一个用来设定初始条件，
        另一个用来和模拟结果比较
        """
        axs[i_].contour(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                       grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                       np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                       levels, colors='k',
                       linewidths=.5, linestyles='solid', negative_linestyles='solid')

        axs[i_].arrow(vdf_max_vrtn[0] + psp_vel[0],  # - 110 * rotmat_mfa_rtn[0, 0],
                     vdf_max_vrtn[2] + psp_vel[2],  # - 110 * rotmat_mfa_rtn[0, 2],
                     110 * rotmat_mfa_rtn[0, 0],
                     110 * rotmat_mfa_rtn[0, 2],
                     width=10., head_width=30., color='c')
        axs[i_].text(vdf_max_vrtn[0] + psp_vel[0],
                    vdf_max_vrtn[2] + psp_vel[2] + 50,
                    r'$\vec{B}$', color='c')
        axs[i_].set_xlabel('$V_R$ (km/s)', fontsize=ftsize)
        axs[i_].set_ylabel('$V_N$ (km/s)', fontsize=ftsize)

        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((0, 900))
        axs[i_].set_ylim((-450, 450))
        
        axs[i_].set_title('VDF-$t_1$(outside HCS)', color='b', fontsize=25)
        axs[i_].tick_params(labelsize=ftsize)
        print(type(ttime))
        axs[i_].text(20,365,'(g)', fontsize=20,weight='semibold')
        '''
        SPEC_2
        '''

        axs.append(fig.add_subplot(spec2[:, 1]))
        time_2 = np.datetime64('2021-08-10T00:29:58')
        tind, ttime = find_nearest(vdf_time, time_2)
        axs[0].text(ttime, y_epoch_mark, r'$t_2$', fontsize=30)
        axs[5].axvline(x=ttime, ymax=6, c="blue", linewidth=2, zorder=0, clip_on=False, linestyle='--')
        temp_vdf0 = vdf[tind, :]
        good_ind = (~np.isnan(temp_vdf0))#  & (temp_vdf0 > 0)

        speed = speedbin[tind, good_ind]
        theta = thetabin[tind, good_ind]
        phi = phibin[tind, good_ind]

        vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        vz_inst = speed * np.sin(np.deg2rad(theta))
        v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
        v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

        temp_vdf = temp_vdf0[good_ind]

        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]

        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        temp_vdf = temp_vdf0[good_ind]
        psp_vel = pytplot.get('psp_spi_SC_VEL_RTN_SUN')
        psp_vel_vec = psp_vel.y
        psp_vel = psp_vel_vec.mean(axis=0)
        # make linear delaunay interpolation in rtn frame
        cart_temp = v_rtn
        points = np.stack(cart_temp).T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        # plot vdf in v-vmax frame (eliminate bulk speed)
        levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.75, -6.5, -6.25, -6., -5.75, -5.5,-5.25, -5.]
        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]
        print(vdf_max_vrtn)
        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        # make linear delaunay interpolation in v0_rtn frame
        points = v0_rtn.T
        delaunay = scipy.spatial.Delaunay(points)
        temp_vdf[vx_inst > -100] = 0
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

        grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-800, 800, 200),
                                                np.linspace(-500, 500, 125),
                                                np.linspace(-500, 500, 125), indexing='ij')

        print(grid_vr.shape)
        grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
        grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

        grid_vdf = grid_vdf0
        grid_vdf_max_ind = grid_vdf_max_ind0

        # axs.append(fig.add_subplot(spec2[:, 0]))
        psp_vel = psp_vel.value
        i_+=1
        pos = axs[i_].contourf(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                              grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                              np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                              levels, cmap='jet', extend='both')
        axs[i_].contour(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                       grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                       np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                       levels, colors='k',
                       linewidths=.5, linestyles='solid', negative_linestyles='solid')
        axs[i_].arrow(vdf_max_vrtn[0] + psp_vel[0],  # - 110 * rotmat_mfa_rtn[0, 0],
                     vdf_max_vrtn[2] + psp_vel[2],  # - 110 * rotmat_mfa_rtn[0, 2],
                     110 * rotmat_mfa_rtn[0, 0],
                     110 * rotmat_mfa_rtn[0, 2],
                     width=10., head_width=30., color='c')
        axs[i_].set_xlabel('$V_R$ (km/s)', fontsize=ftsize)
        axs[i_].set_ylabel('$V_N$ (km/s)', fontsize=ftsize)
        axs[i_].tick_params(labelsize=ftsize)

        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((0, 900))
        axs[i_].set_ylim((-450, 450))
        axs[i_].set_title('VDF-$t_2$(Edge of HCS)', color='b', fontsize=25)
        axs[i_].text(20,365,'(h)', fontsize=20,weight='semibold')


        '''
        SPEC_3
        '''
        axs.append(fig.add_subplot(spec2[:, 2]))
        time_3 = np.datetime64('2021-08-10T01:15:58')
        tind, ttime = find_nearest(vdf_time, time_3)
        axs[5].axvline(x=ttime, ymax=6, c="blue", linewidth=2, zorder=0, clip_on=False, linestyle='--')
        axs[0].text(ttime, y_epoch_mark, r'$t_3$', fontsize=30)
        temp_vdf0 = vdf[tind, :]
        good_ind = (~np.isnan(temp_vdf0))#  & (temp_vdf0 > 0)

        speed = speedbin[tind, good_ind]
        theta = thetabin[tind, good_ind]
        phi = phibin[tind, good_ind]

        vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        vz_inst = speed * np.sin(np.deg2rad(theta))
        v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
        v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

        temp_vdf = temp_vdf0[good_ind]

        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]

        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        temp_vdf = temp_vdf0[good_ind]
        psp_vel = pytplot.get('psp_spi_SC_VEL_RTN_SUN')
        psp_vel_vec = psp_vel.y
        psp_vel = psp_vel_vec.mean(axis=0)
        # make linear delaunay interpolation in rtn frame
        cart_temp = v_rtn
        points = np.stack(cart_temp).T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        # plot vdf in v-vmax frame (eliminate bulk speed)
        levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.75, -6.5, -6.25, -6., -5.75, -5.5,-5.25, -5.]
        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]
        print(vdf_max_vrtn)
        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        # make linear delaunay interpolation in v0_rtn frame
        points = v0_rtn.T
        delaunay = scipy.spatial.Delaunay(points)
        temp_vdf[vx_inst > -100] = 0
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

        grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-800, 800, 200),
                                                np.linspace(-500, 500, 125),
                                                np.linspace(-500, 500, 125), indexing='ij')

        print(grid_vr.shape)
        grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
        grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

        grid_vdf = grid_vdf0
        grid_vdf_max_ind = grid_vdf_max_ind0

        # axs.append(fig.add_subplot(spec2[:, 0]))
        psp_vel = psp_vel.value
        i_+=1
        pos = axs[i_].contourf(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                              grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                              np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                              levels, cmap='jet', extend='both')
        axs[i_].contour(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                       grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                       np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                       levels, colors='k',
                       linewidths=.5, linestyles='solid', negative_linestyles='solid')
        axs[i_].arrow(vdf_max_vrtn[0] + psp_vel[0],  # - 110 * rotmat_mfa_rtn[0, 0],
                     vdf_max_vrtn[2] + psp_vel[2],  # - 110 * rotmat_mfa_rtn[0, 2],
                     110 * rotmat_mfa_rtn[0, 0],
                     110 * rotmat_mfa_rtn[0, 2],
                     width=10., head_width=30., color='c')
        axs[i_].set_xlabel('$V_R$ (km/s)', fontsize=ftsize)
        axs[i_].set_ylabel('$V_N$ (km/s)', fontsize=ftsize)

        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((0, 900))
        axs[i_].set_ylim((-450, 450))
        axs[i_].set_title('VDF-$t_3$(Inside HCS)', color='b', fontsize=25)
        axs[i_].tick_params(labelsize=ftsize)
        axs[i_].text(20,365,'(i)', fontsize=20,weight='semibold')

        axs.append(fig.add_subplot(spec2[:, 3]))
        time_2 = np.datetime64('2021-08-10T02:10:58')
        
        tind, ttime = find_nearest(vdf_time, time_2)
        axs[5].axvline(x=ttime, ymax=6, c="blue", linewidth=2, zorder=0, clip_on=False, linestyle='--')
        axs[0].text(ttime, y_epoch_mark, r'$t_4$', fontsize=30)
        temp_vdf0 = vdf[tind, :]
        good_ind = (~np.isnan(temp_vdf0))#  & (temp_vdf0 > 0)

        speed = speedbin[tind, good_ind]
        theta = thetabin[tind, good_ind]
        phi = phibin[tind, good_ind]

        vx_inst = speed * np.cos(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
        vy_inst = speed * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
        vz_inst = speed * np.sin(np.deg2rad(theta))
        v_sc = np.matmul(rotmat_sc_inst, np.array([vx_inst, vy_inst, vz_inst]))
        v_rtn = np.array([-v_sc[2, :], v_sc[0, :], -v_sc[1, :]])

        temp_vdf = temp_vdf0[good_ind]

        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]

        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        temp_vdf = temp_vdf0[good_ind]
        psp_vel = pytplot.get('psp_spi_SC_VEL_RTN_SUN')
        psp_vel_vec = psp_vel.y
        psp_vel = psp_vel_vec.mean(axis=0)
        # make linear delaunay interpolation in rtn frame
        cart_temp = v_rtn
        points = np.stack(cart_temp).T
        delaunay = scipy.spatial.Delaunay(points)
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)
        # plot vdf in v-vmax frame (eliminate bulk speed)
        levels = [-11, -10.5, -10, -9.5, -9, -8.5, -8, -7.5, -7, -6.75, -6.5, -6.25, -6., -5.75, -5.5,-5.25, -5.]
        vdf_max_ind = temp_vdf.argmax()
        vdf_max_vrtn = v_rtn[:, vdf_max_ind]
        print(vdf_max_vrtn)
        v0_rtn = v_rtn - vdf_max_vrtn[:, None]
        # make linear delaunay interpolation in v0_rtn frame
        points = v0_rtn.T
        delaunay = scipy.spatial.Delaunay(points)
        temp_vdf[vx_inst > -100] = 0
        vdf_interp = scipy.interpolate.LinearNDInterpolator(delaunay, temp_vdf)

        grid_vr, grid_vt, grid_vn = np.meshgrid(np.linspace(-800, 800, 200),
                                                np.linspace(-500, 500, 125),
                                                np.linspace(-500, 500, 125), indexing='ij')

        print(grid_vr.shape)
        grid_vdf0 = vdf_interp((grid_vr, grid_vt, grid_vn))
        grid_vdf_max_ind0 = np.unravel_index(np.nanargmax(grid_vdf0), grid_vdf0.shape)

        grid_vdf = grid_vdf0
        grid_vdf_max_ind = grid_vdf_max_ind0

        # axs.append(fig.add_subplot(spec2[:, 0]))
        psp_vel = psp_vel.value
        i_+=1
        pos = axs[i_].contourf(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                              grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                              np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                              levels, cmap='jet', extend='both')
        axs[i_].contour(grid_vr[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[0] + psp_vel[0],
                       grid_vn[:, grid_vdf_max_ind[1], :] + vdf_max_vrtn[2] + psp_vel[2],
                       np.log10(grid_vdf[:, grid_vdf_max_ind[1], :]),
                       levels, colors='k',
                       linewidths=.5, linestyles='solid', negative_linestyles='solid')
        axs[i_].arrow(vdf_max_vrtn[0] + psp_vel[0],  # - 110 * rotmat_mfa_rtn[0, 0],
                     vdf_max_vrtn[2] + psp_vel[2],  # - 110 * rotmat_mfa_rtn[0, 2],
                     110 * rotmat_mfa_rtn[0, 0],
                     110 * rotmat_mfa_rtn[0, 2],
                     width=10., head_width=30., color='c')
        print(f"磁场：{rotmat_mfa_rtn[0, 0]},{rotmat_mfa_rtn[0, 2]}")
        axs[i_].set_xlabel('$V_R$ (km/s)', fontsize=ftsize)
        axs[i_].set_ylabel('$V_N$ (km/s)', fontsize=ftsize)
        # plt.colorbar(pos)
        axs[i_].set_aspect('equal', adjustable='box')
        axs[i_].set_xlim((0, 900))
        axs[i_].set_ylim((-450, 450))
        axs[i_].set_title('VDF-$t_4$(Outside HCS)', color='b', fontsize=25)
        axs[i_].tick_params(labelsize=ftsize)
        axs[i_].text(20,365,'(j)', fontsize=20,weight='semibold')

        cax = fig.add_subplot(spec2[:, 4])
        cbar=fig.colorbar(pos, cax=cax) 
        cbar.set_label(label='$\log_{10}$(VDF)',fontsize=25)
        cbar.ax.tick_params(labelsize=15)
        # plt.suptitle(fr"$v_r$={vp_r[tind_pmom]:.2f}km/s,$\Delta v={(vp_r[tind_pmom]-vp_r[-1])/va0:.2f}v_A$,|B|={Babs_epochp[tind_pmom]:.2f}nT, $\sqrt{{|Bt|^2+|Bn|^2}}$={np.sqrt(Babs_epochp[tind_pmom]**2-Br_epochp[tind_pmom]**2):.2f}nT,Br={Br_epochp[tind_pmom]:.2f}nT")

# %%
vdftime_full_list = [np.datetime64('2021-08-10T00:00:20') + np.timedelta64(60, 's') * n for n in range(10)]
EXPORT_PATH = 'plot_vdf_movie/E9_new/'
os.makedirs(EXPORT_PATH, exist_ok=True)
for i in range(180):
    # print(type(ttime.astype(datetime.datetime)))
    plot_vdf_frame(vdftime_full_list[i],6, frame='rtn')
    time_str = vdftime_full_list[i].tolist().strftime('%H:%M:%S')
    safe_time_str = time_str.replace(':', '_')
    plt.savefig(EXPORT_PATH + 'HCS1_' + safe_time_str + '_vdf_rtn.png')
    plt.close()
# %%
from utils import folder_to_movie

folder_to_movie(EXPORT_PATH, filename_pattern='HCS1_(.+)_vdf_rtn.png',
                time_format='%H:%M:%S', export_pathname=EXPORT_PATH + 'vdf_overview_rtn_HCS1',
                video_format='.mp4')

