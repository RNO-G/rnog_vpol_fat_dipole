import glob
import pickle 
import argparse
import os
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
import process_effective_height as pf
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

c = scipy.constants.c / 1e9
ZL = 50.0 # Impedance of coax / feed
Z0 = 120.0 * np.pi # Impedance of free space

def plot_gain(data, n, new_figure = True, color = "Purple", linestyle = "-", legend = True, labels = True):

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]
    azimuth_angles = data["azimuth_angles"]
    zenith_angles = data["zenith_angles"]

    nsamples = len(ts)
    fs = ts[1] - ts[0]
    freqs = np.fft.rfftfreq(len(h_phi[0][0]), fs)

    if(new_figure):
        plt.figure()

    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            if(zenith_angle != 90):
                continue

            h_fft_the = np.fft.rfft(h_the[i_azimuth_angle][i_zenith_angle], nsamples)
            h_fft_phi = np.fft.rfft(h_phi[i_azimuth_angle][i_zenith_angle], nsamples)
        
            # Convert to gains
            gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi))) * n / c, 2.0) * Z0 / ZL / n
            gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
            gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
            
            # Get rid of log of zero issues
            gain[0] = 1e-20
            gain_the[0] = 1e-20
            gain_phi[0] = 1e-20

            plt.plot(1000.0 * freqs, 
                     10.0 * np.log10(gain), 
                     color = color, 
                     linestyle = linestyle)
            #alpha = (azimuth_angle + 10.0) / 100.0)

    if(legend):
        plt.legend(loc = 'lower right', title = "Azimuth Angles")

    if(labels):
        plt.xlabel("Freqs. [MHz]")
        plt.ylabel("Realized Gain [dBi]")
        plt.ylim(-5., 5.0)
        plt.xlim(0.0, 1000.0)
        plt.grid()

def plot_effective_height(data, n, new_figure = True, color = "Purple", linestyle = "-", legend = True, labels = True):

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]
    azimuth_angles = data["azimuth_angles"]
    zenith_angles = data["zenith_angles"]

    nsamples = len(ts)
    fs = ts[1] - ts[0]
    freqs = np.fft.rfftfreq(len(h_phi[0][0]), fs)

    if(new_figure):
        plt.figure()

    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            if(zenith_angle != 60):
                continue
            #if(zenith_angle != 90):
            #    continue

            trace_ = np.fft.fftshift(h_the[i_azimuth_angle][i_zenith_angle])
            ts_ = ts - ts[np.argmin(trace_)]

            plt.plot(ts_, trace_, color = color, linestyle = linestyle)

    if(labels):

        if(legend):
            plt.legend(loc = 'lower right', title = "Azimuth Angles")

        plt.title("In-Ice Realized Effective Height at Boresight / $90^\circ$ Zenith of VPol v2 \n Air: n = 1.0, Ice: n = 1.75")
        plt.minorticks_on()
        plt.grid(which = "major")
        plt.grid(which = "minor", alpha = 0.25)
        plt.xlabel("Time [ns]")
        plt.ylabel("Effective Height [m]")
        plt.ylim(-0.02, 0.02)
        plt.xlim(0.0, 40.0)
        plt.grid()

def plot_gain_polar(file_name, n, freqs_oi):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    data = np.load(file_name)

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]
    azimuth_angles = data["azimuth_angles"]
    zenith_angles = data["zenith_angles"]

    nsamples = len(ts)
    fs = ts[1] - ts[0]
    freqs = np.fft.rfftfreq(len(h_phi[0][0]), fs)

    # Plot per freqs_oi
    gains_oi_the = [[] for i in range(len(freqs_oi))]
    gains_oi_phi = [[] for i in range(len(freqs_oi))]

    azimuth_angles = np.arange(0, 100, 10)
    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            if(azimuth_angle != 0):
                continue

            h_fft_the = np.fft.rfft(h_the[i_azimuth_angle][i_zenith_angle], nsamples)
            h_fft_phi = np.fft.rfft(h_phi[i_azimuth_angle][i_zenith_angle], nsamples)
        
            # Convert to gains
            gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi))) * n / c, 2.0) * Z0 / ZL / n
            gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
            gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
            
            # Get rid of log of zero issues
            gain_the[0] = 1e-20
            gain_phi[0] = 1e-20
            
            f_gain_the = scipy.interpolate.interp1d(freqs, gain_the, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
            f_gain_phi = scipy.interpolate.interp1d(freqs, gain_phi, kind = "cubic", bounds_error = False, fill_value = "extrapolate")

            for i_freq_oi, freq_oi in enumerate(freqs_oi):
                gains_oi_the[i_freq_oi] += [f_gain_the(freq_oi)]
                gains_oi_phi[i_freq_oi] += [f_gain_phi(freq_oi)]

    fig, ax = plt.subplots(nrows = 1, ncols = len(freqs_oi), subplot_kw = {'projection': 'polar'}, figsize = (3 * len(freqs_oi), 3))
    fig.suptitle("Realized Gain, Azimuth Beam Pattern at Boresight / $90^\circ$ Zenith", fontsize=14)

    for irow, row in enumerate(ax):
        row.set_title(str(np.round(freqs_oi[irow] * 1000.0, 0))+" MHz")
        row.plot(np.deg2rad(azimuth_angles), 10.0 * np.log10(gains_oi_phi[irow]), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 1.0 * np.pi / 2.0, 10.0 * np.log10(np.flip(gains_oi_phi[irow])), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 2.0 * np.pi / 2.0, 10.0 * np.log10(gains_oi_phi[irow]), color = "purple")
        row.plot(np.deg2rad(azimuth_angles) + 3.0 * np.pi / 2.0, 10.0 * np.log10(np.flip(gains_oi_phi[irow])), color = "purple")
        row.set_rmax(5.0)
        row.set_rmin(-5.0)        
        row.set_xticklabels(['', '$45^\circ$', '', '$135^\circ$', '', '$225^\circ$', '', '$315^\circ$'])

def plot_gain_polar_3d(file_name, n, freqs_oi):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    data = np.load(file_name)

    ts = data["ts"]
    h_the = data["h_the"]
    h_phi = data["h_phi"]
    azimuth_angles = data["azimuth_angles"]
    zenith_angles = data["zenith_angles"]

    nsamples = len(ts)
    fs = ts[1] - ts[0]
    freqs = np.fft.rfftfreq(len(h_phi[0][0]), fs)

    # Plot per freqs_oi
    gains_oi = [[] for i in range(len(freqs_oi))]
    gains_oi_the = [[] for i in range(len(freqs_oi))]
    gains_oi_phi = [[] for i in range(len(freqs_oi))]
    the = [[] for i in range(len(freqs_oi))]
    phi = [[] for i in range(len(freqs_oi))]

    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            h_fft_the = np.fft.rfft(h_the[i_azimuth_angle][i_zenith_angle], nsamples)
            h_fft_phi = np.fft.rfft(h_phi[i_azimuth_angle][i_zenith_angle], nsamples)
        
            # Convert to gains
            gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi))) * n / c, 2.0) * Z0 / ZL / n
            gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
            gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
            
            # Get rid of log of zero issues
            gain[0] = 1e-20
            gain_the[0] = 1e-20
            gain_phi[0] = 1e-20
            
            f_gain = scipy.interpolate.interp1d(freqs, gain, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
            f_gain_the = scipy.interpolate.interp1d(freqs, gain_the, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
            f_gain_phi = scipy.interpolate.interp1d(freqs, gain_phi, kind = "cubic", bounds_error = False, fill_value = "extrapolate")

            for i_freq_oi, freq_oi in enumerate(freqs_oi):
                gains_oi[i_freq_oi] += [10.0 * np.log10(f_gain(freq_oi))]
                gains_oi_the[i_freq_oi] += [10.0 * np.log10(f_gain_the(freq_oi))]
                gains_oi_phi[i_freq_oi] += [10.0 * np.log10(f_gain_phi(freq_oi))]
                the[i_freq_oi] += [azimuth_angle]
                phi[i_freq_oi] += [zenith_angle]

    phi = np.deg2rad(phi)
    the = np.deg2rad(the)

    gains_oi[0] = [gain_ if gain_ > 0 else 0.0 for gain_ in gains_oi[0]] # for plotting reasons

    Xs = gains_oi[0] * np.sin(phi[0]) * np.cos(the[0])
    Ys = gains_oi[0] * np.sin(phi[0]) * np.sin(the[0])
    Zs = gains_oi[0] * np.cos(phi[0])

    phis, thetas = np.mgrid[0.0:180.0:10j, 0.0:90.0:10j]
    phi = np.rad2deg(phi)
    the = np.rad2deg(the)

    x = np.zeros(phis.shape)
    y = np.zeros(phis.shape)
    z = np.zeros(phis.shape)

    for i in range(len(phis)):
        for j in range(len(phis[i])):
            phi_ = phis[i][j]
            theta_ = thetas[i][j]

            # now, need to find the index of the gain
            index_oi = -1
            for iii in range(len(phi[0])):
                if(np.round(phi[0][iii]) == np.round(phi_) and np.round(the[0][iii]) == np.round(theta_)):
                    index_oi = iii
                    break

            if(index_oi == -1):
                print("Didn't find, exiting")
                print(phi_, theta_)
                exit()

            x[i][j] = Xs[index_oi]
            y[i][j] = Ys[index_oi]
            z[i][j] = Zs[index_oi]
        
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)

    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    my_col = cm.jet(r / np.max(r))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(x, y, z,   rstride = 1, cstride = 1, cmap = cm.jet, facecolors = my_col, alpha = 1.0, linewidth = 0)
    ax.plot_surface(-x, y, z,  rstride = 1, cstride = 1, cmap = cm.jet, facecolors = my_col, alpha = 1.0, linewidth = 0)
    ax.plot_surface(x, -y, z,  rstride = 1, cstride = 1, cmap = cm.jet, facecolors = my_col, alpha = 1.0, linewidth = 0)
    ax.plot_surface(-x, -y, z, rstride = 1, cstride = 1, cmap = cm.jet, facecolors = my_col, alpha = 1.0, linewidth = 0)
    ax.set_xlim(-5., 5.)
    ax.set_ylim(-5., 5.)
    ax.set_zlim(-5., 5.)
    ax.set_aspect('equal')

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(r)
    fig.colorbar(m)

def plot_vswr(component_name, base_name, new_figure = True, color = "Purple", linestyle = "-", legend = True, labels = True):

    # Load up the data that has to do with the feed    
    data = np.genfromtxt(base_name+"/"+component_name+"-S-ParametersImag.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_i = np.array(data[:,1])
    
    data = np.genfromtxt(base_name+"/"+component_name+"-S-ParametersReal.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_r = np.array(data[:,1])

    s11 = s11_r + 1j * s11_i

    if(new_figure):
        plt.figure()

    plt.plot(1000.0 * s11_freqs, (1.0 + np.abs(s11)) / (1.0 - np.abs(s11)), color = color, linestyle = linestyle)

    if(labels):
        plt.title("In-Ice VSWR of one of two Horizontal VPols")
        plt.xlim(0.1, 1000.0)
        plt.ylim(1.0, 10.0)
        plt.minorticks_on()
        plt.grid(which = "major")
        plt.grid(which = "minor", alpha = 0.25)
        plt.xlabel("Freq. [MHz]")
        plt.ylabel("VSWR")

if __name__ == "__main__":

    # First, check if they have already been processed
    output_file_name_ice = "ice_processed_effective_height.npz"
    output_file_name_air = "air_processed_effective_height.npz"
    n_ice = 1.75
    n_air = 1.00
    r = 7.07106
    component_name = "Component"
    base_name_air = "vpol_xfdtd_simulation.xf/air_output"
    base_name_ice = "vpol_xfdtd_simulation.xf/ice_output"

    if not(os.path.isfile(output_file_name_ice)):
        base_names = [base_name_ice]
        
        pf.process_and_save(r = r, n = n_ice,                             
                            component_name = component_name,
                            base_names = base_names,
                            output_file_name = output_file_name_ice,
                            phase_shift = 0.0,
                            time_delay = 0.0
                        )

    if not(os.path.isfile(output_file_name_air)):
        base_names = [base_name_air]
    
        pf.process_and_save(r = r, n = n_air,                             
                            component_name = component_name,
                            base_names = base_names,
                            output_file_name = output_file_name_air,
                            phase_shift = 0.0,
                            time_delay = 0.0
                        )

    # now that is all done, got to load things up
    data_air = np.load(output_file_name_air)
    data_ice = np.load(output_file_name_ice)

    #####################
    # plot gain pattern #
    #####################

    plot_gain(data_air, n_air, legend = False, labels = False)
    plot_gain(data_ice, n_ice, new_figure = False, color = "blue", linestyle = "--", legend = False, labels = False)

    plt.plot([], [], color="purple", label = "In Ice")
    plt.plot([], [], color="blue", label = "In Air")

    plt.scatter([124.0], [-1], color = "purple")
    plt.scatter([144.0], [-1], color = "blue")

    plt.scatter([607.0], [-1], color = "purple")
    plt.scatter([578.0], [-1], color = "blue")

    plt.text(150, -1.5, "~145 MHz", color = "blue")
    plt.text(10, -0.6, "~125 MHz", color = "purple")

    plt.text(460, -1.5, "~580 MHz", color = "blue")
    plt.text(610, -0.6, "~610 MHz", color = "purple")

    plt.minorticks_on()
    plt.grid(which = "major")
    plt.grid(which = "minor", alpha = 0.25)
    plt.plot([0.0, 1000.], [-1.0, -1.0], color = "red", linestyle = "--")
    plt.yticks(np.arange(-5, 6))
    plt.xlim(0, 750.0)
    plt.ylim(-5, 5.0)
    plt.legend(loc = "upper right")
    plt.title("Realized Gain at Boresight / $90^\circ$ Zenith of VPol v2 \n Air: n = 1.0, Ice: n = 1.75")

    #############
    # plot vswr #
    #############

    plot_vswr(component_name, base_name_air, labels = False)
    plot_vswr(component_name, base_name_ice, new_figure = False, color = "blue", linestyle = "--", legend = False, labels = False)

    plt.plot([], [], color="purple", label = "In Ice")
    plt.plot([], [], color="blue", label = "In Air")

    plt.title("VSWR of VPol v2 \n Air: n = 1.0, Ice: n = 1.75")
    plt.xlim(0.0, 750.0)
    plt.ylim(1.0, 5.0)
    plt.minorticks_on()
    plt.grid(which = "major")
    plt.grid(which = "minor", alpha = 0.25)
    plt.xlabel("Freq. [MHz]")
    plt.ylabel("VSWR")
    plt.legend()

    #########################
    # plot effective height #
    #########################

    plot_effective_height(data_air, n_air, legend = False, labels = False)
    plot_effective_height(data_ice, n_ice, new_figure = False, color = "blue", linestyle = "--", legend = False, labels = False)

    plt.plot([], [], color="purple", label = "In Ice")
    plt.plot([], [], color="blue", label = "In Air")

    plt.title("Effective Height at Boresight / $90^\circ$ Zenith of VPol v2 \n Air: n = 1.0, Ice: n = 1.75")
    plt.legend()
    plt.xlabel("Time [ns]")
    plt.ylabel("Effective Height [m]")
    plt.ylim(-0.04, 0.04)
    plt.xlim(-5.0, 20.0)
    plt.grid()

    ########
    # show #
    ########

    plt.show()
