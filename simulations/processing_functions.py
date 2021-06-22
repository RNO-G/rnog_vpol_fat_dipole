import scipy.interpolate
import numpy as np
from scipy.signal import butter, lfilter, cheby1
import glob

def get_efield(filename):
    data = np.genfromtxt(filename, dtype = "float", delimiter = ",", skip_header = 1)
    ts = np.array(data[:,0])
    efield = np.array(data[:,1])
    return ts, efield

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band') #butter
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def phase_shift(efield, phase):
    nsamples = len(efield)
    efield_fft = np.fft.rfft(efield, nsamples)
    efield_fft_mag = np.abs(efield_fft)
    efield_fft_ang = np.unwrap(np.angle(efield_fft))
    efield_fft_ang += phase
    efield_fft = efield_fft_mag * (np.cos(efield_fft_ang) + 1j * np.sin(efield_fft_ang))
    efield = np.fft.irfft(efield_fft, nsamples)
    return efield

def time_delay(ts, efield, delay):
    f_efield = scipy.interpolate.interp1d(ts, efield, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
    efield = f_efield(np.array(ts) + delay)
    return efield

def process_and_save(r, n, component_name, base_names, output_file_name, phase_shift, time_delay):
    c = scipy.constants.c / 1e9
    ZL = 50.0 # Impedance of coax / feed
    Z0 = 120.0 * np.pi # Impedance of free space

    # Load up the data that has to do with the feed
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-Voltage (V).csv", dtype = "float", delimiter = ",", skip_header = 1)
    ts = np.array(data[:,0])
    ts -= ts[0]
    V_measured  = np.array(data[:,1])
    V_measured_func = scipy.interpolate.interp1d(ts, V_measured, kind = "cubic", bounds_error = False, fill_value = "extrapolate")
    
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-S-ParametersImag.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_i = np.array(data[:,1])
    
    data = np.genfromtxt(base_names[0]+"/"+component_name+"-S-ParametersReal.csv", dtype = "float", delimiter = ",", skip_header = 1)
    s11_freqs = np.array(data[:,0])
    s11_r = np.array(data[:,1])
    
    res11_func = scipy.interpolate.interp1d(s11_freqs, s11_r, kind = "cubic", bounds_error = False, fill_value = 0.0)
    ims11_func = scipy.interpolate.interp1d(s11_freqs, s11_i, kind = "cubic", bounds_error = False, fill_value = 0.0)

    file_names = glob.glob("./%s/PS_*X.csv" % base_names[0])
    zenith_angles = np.sort(np.unique(np.array([file_name.split("_")[-2] for file_name in file_names]).astype("float")))
    azimuth_angles = np.sort(np.unique(np.array([file_name.split("_")[-1].split("-")[0] for file_name in file_names]).astype("float")))

    ts, efield = get_efield(file_names[0])
    ts -= ts[0]
    nsamples = len(ts)

    effective_height_results_the = np.zeros((len(azimuth_angles), len(zenith_angles), nsamples))
    effective_height_results_phi = np.zeros((len(azimuth_angles), len(zenith_angles), nsamples))
    
    # Scan over azimuth angles
    for i_azimuth_angle, azimuth_angle in enumerate(azimuth_angles):
        for i_zenith_angle, zenith_angle in enumerate(zenith_angles):

            # Load up electric fields at this azimuth
            file_name_run1 = glob.glob("./%s/PS_%s_%s*X.csv" % (base_names[0], str(int(zenith_angle)), str(int(azimuth_angle))))[0]
            #file_name_run2 = glob.glob("./%s/PS_%s_%s*X.csv" % (base_names[1], str(int(zenith_angle)), str(int(azimuth_angle))))[0]

            ts_run1, efield_x_run1 = get_efield(file_name_run1)
            ts_run1, efield_y_run1 = get_efield(file_name_run1.replace("X", "Y"))
            ts_run1, efield_z_run1 = get_efield(file_name_run1.replace("X", "Z"))
        
            #ts_run2, efield_x_run2 = pf.get_efield(file_name_run2)
            #ts_run2, efield_y_run2 = pf.get_efield(file_name_run2.replace("X", "Y"))
            #ts_run2, efield_z_run2 = pf.get_efield(file_name_run2.replace("X", "Z"))
            
            ts_run1 -= ts_run1[0]
            #ts_run2 -= ts_run2[0]

            '''
            # Phase shift the second antenna before combining            
            efield_x_run2 = pf.phase_shift(efield_x_run2, phase_shift)
            efield_y_run2 = pf.phase_shift(efield_y_run2, phase_shift)
            efield_z_run2 = pf.phase_shift(efield_z_run2, phase_shift)
            
            # Signal delay as proxy for phase shift in the second antenna
            efield_x_run2 = pf.time_delay(ts_run2, efield_x_run2, time_delay)
            efield_y_run2 = pf.time_delay(ts_run2, efield_y_run2, time_delay)
            efield_z_run2 = pf.time_delay(ts_run2, efield_z_run2, time_delay)
            
            # Combine the two electric fields, only works if they have the same run time
            ts = ts_run2
            efield_x = (efield_x_run1 + efield_x_run2) 
            efield_y = (efield_y_run1 + efield_y_run2) 
            efield_z = (efield_z_run1 + efield_z_run2) 
            '''

            ts = ts_run1
            efield_x = (efield_x_run1)
            efield_y = (efield_y_run1)
            efield_z = (efield_z_run1)
            fs = ts[1] - ts[0]
            
            # Convert cartesian efields into spherical components
            the = np.deg2rad(zenith_angle)
            phi = np.pi/2.0 - np.deg2rad(azimuth_angle)
        
            r_matrix = np.array([[np.sin(the) * np.cos(phi), np.cos(the) * np.cos(phi), np.cos(the)],
                                 [np.cos(the) * np.cos(phi), np.cos(the) * np.sin(phi), -np.sin(the)],
                                 [-np.sin(phi), np.cos(phi), 0.0]])
        
            transform = r_matrix.dot([efield_x, efield_y, efield_z])

            # At this point, I assume the r component doesn"t exist. Only fair in true far field. At 5m, it is  ~20 dB down in power
            efield_the = transform[1,:]
            efield_phi = transform[2,:]
        
            freqs = np.fft.rfftfreq(len(efield_x), fs)
            efield_the_fft = np.fft.rfft(efield_the, nsamples)
            efield_phi_fft = np.fft.rfft(efield_phi, nsamples)            
            s11_ = res11_func(freqs) + 1j * ims11_func(freqs)

            # Input voltage at feed, corrected for matching
            V_input = V_measured_func(ts)
            V_straight_fft = np.fft.rfft(V_input)
            V_straight_fft /= (1.0 + s11_) 
        
            w = np.array(2.0 * np.pi * freqs)
            w[0] = 1e-20 # get rid of divide by zero issues
        
            h_fft_the = (2.0 * np.pi * r * c) / (1j * w) * (efield_the_fft / V_straight_fft) * (ZL / Z0) 
            h_fft_phi = (2.0 * np.pi * r * c) / (1j * w) * (efield_phi_fft / V_straight_fft) * (ZL / Z0) 
            
            h_the = np.fft.irfft(h_fft_the, nsamples)
            h_phi = np.fft.irfft(h_fft_phi, nsamples)

            # Bandpass out aphysical freqs
            h_the = butter_bandpass_filter(h_the, 0.01, 1.0, 1.0 / (fs), order=5)
            h_phi = butter_bandpass_filter(h_phi, 0.01, 1.0, 1.0 / (fs), order=5)
        
            # Align zero time, approximately. Sadly, I never automated this.
            h_the = np.roll(h_the, -100)
            h_phi = np.roll(h_phi, -100)
    
            h_fft_the = np.fft.rfft(h_the, nsamples)
            h_fft_phi = np.fft.rfft(h_phi, nsamples)

            effective_height_results_the[i_azimuth_angle][i_zenith_angle] = h_the
            effective_height_results_phi[i_azimuth_angle][i_zenith_angle] = h_phi
                
            # Convert to gains
            gain = 4.0 * np.pi * np.power(freqs * np.sqrt(np.square(np.abs(h_fft_the)) + np.square(np.abs(h_fft_phi)))* n / c, 2.0) * Z0 / ZL / n
            gain_the = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_the) * n / c, 2.0) * Z0 / ZL / n 
            gain_phi = 4.0 * np.pi * np.power(freqs * np.abs(h_fft_phi) * n / c, 2.0) * Z0 / ZL / n
        
            # Get rid of log of zero issues
            gain_the[0] = 1e-20
            gain_phi[0] = 1e-20

    effective_height_results_the = np.array(effective_height_results_the)
    effective_height_results_phi = np.array(effective_height_results_phi)

    np.savez(output_file_name, ts = ts, h_the = effective_height_results_the, h_phi = effective_height_results_phi)
