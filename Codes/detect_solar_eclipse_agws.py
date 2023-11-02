################### Necessary Libraries ###################

import numpy as np
import matplotlib.pyplot as plt
import glob
from waveletFunctions import wave_signif

import cmasher as cm
  
import matplotlib

import plottingfunctions
import datafunctions

plt.ion()

################### Plotting Properties ###################

tex_fonts = {
    "text.usetex": False,
    "font.family": 'DeJavu Serif',
    "font.serif": "Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "axes.formatter.use_mathtext": True,
    "lines.linewidth": 2,
    "ytick.minor.visible": True,
    "xtick.minor.visible": True,
    "figure.facecolor": "white",
    "pcolor.shading": "auto",
}

matplotlib.rcParams.update(tex_fonts)

################### Read Data ###################

# path to data
path = "/media/oana/Data1/Annular_Eclipse_Analysis/Data/"
path_to_save_figures = "/media/oana/Data1/Annular_Eclipse_Analysis/Figures/"

# Select all xls files that match
fname = glob.glob(path + "*end.xls")

file_nom = 0

# Read in dataset
dat = datafunctions.read_grawmet_profile(fname[file_nom])

# Extract several key parameters from dataset
(
    original_UTC_time,
    starting_time_for_flight,
    ending_time_for_flight,
    original_data_shape,
    original_altitude_grid,
    original_min_altitude,
    original_max_altitude,
) = datafunctions.grab_initial_grawmet_profile_parameters(dat)

print("\n")
print("Information about data:")
print("\n")
print("Time range: [%s, %s] UTC" % (starting_time_for_flight,ending_time_for_flight))
print("Altitude range: [%s, %s] m" % (original_min_altitude,original_max_altitude))
print("\n")

################### Preprocess and Interpolate Data ###################

# tropopause height [m]
tropopause_height = 12000 

# to give a rise rate of 5 m/s
spatial_resolution = 4

# Spatial height interpolation limit [m]
# Tells code to not interpolate if there are more than [interpolation_limit] consecutive rows of missing/ NaN values
# Interpolation starts anew after hitting the interpolation limit
interpolation_limit = 1000

# Cleaning up the data
dat = datafunctions.clean_data(dat, tropopause_height, original_data_shape)

# Interpolate the data
dat = datafunctions.interpolate_data(dat, interpolation_limit)

# Split dataset into multiple dataframes surrounding the gap of NaNs, if applicable
# As of now, code ignores the other sections if available
data_sections = datafunctions.check_data_for_interpolation_limit_set(
    dat, interpolation_limit
)

# Set the spatial resolution for the dataset
dat = datafunctions.set_spatial_resolution_for_data(data_sections, spatial_resolution)

# Choose which data_sections to use
# FUTURE: Discuss how to best tackle datasets that have a lot of consecutive NaNs
# Upcoming wavelet analysis fails if NaNs or 0s are present in data
choose_data_frame_analyze = dat[0]

# Add in a new time column with the date and time in UTC
choose_data_frame_analyze = datafunctions.convert_seconds_to_timestamp(choose_data_frame_analyze, starting_time_for_flight)

################### Zonal & Meridional Components of Wind Speed ###################

# Grab the wind speeds and temperatures needed for analysis
(
    u_zonal_speed,
    v_meridional_speed,
    temperature,
) = datafunctions.extract_wind_components_and_temperature(choose_data_frame_analyze)

# Grab the height array
height_array = choose_data_frame_analyze["Geopot [m]"]

################### Calculate First-Order Perturbations ###################

# Fit the wind speeds and the temperature 
v_meridional_fit = datafunctions.compute_polynomial_fits(
    choose_data_frame_analyze, v_meridional_speed, 2
)

u_zonal_fit = datafunctions.compute_polynomial_fits(
    choose_data_frame_analyze, u_zonal_speed, 2
)

temperature_fit = datafunctions.compute_polynomial_fits(
    choose_data_frame_analyze, temperature, 2
)

# Subtract the polynomial fits from the original vertical profiles to obtain the first-order perturbations
u_zonal_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, u_zonal_speed, u_zonal_fit
)

v_meridional_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, v_meridional_speed, v_meridional_fit
)

temperature_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, temperature, temperature_fit
)

# Quick Plot of First-Order Perturbations Vertical Profiles
plottingfunctions.plot_vertical_profiles_with_residual_perturbations(
    choose_data_frame_analyze["Geopot [m]"],
    u_zonal_speed,
    v_meridional_speed,
    temperature,
    v_meridional_fit,
    u_zonal_fit,
    temperature_fit,
    u_zonal_perturbations,
    v_meridional_perturbations,
    temperature_perturbations,
    starting_time_for_flight,
    path_to_save_figures + "/First-Order-Perturbations/",
    save_fig=False,
)

################### Wavelet Analysis ###################
# Wavelet Transform will isolate wave packets in wavenumber versus height space

# padding for the ends of the time series to avoid wrap around effects
padding = 1
# the spacing between discrete scales
dj = 0.01
# the amount of height between each recorded height value
dt = spatial_resolution
# The smallest sale of the wavelet
s0 = 2 * dt
# [Zink and Vincent, 2001] -- first order perturbations of a GW packet resembles sine wave ~ Morelet wavelet
mother_wavelet = "MORLET" 

# [Zink and Vincent, 2001] -- real part is data series band-pass filtered corresponding to scale a
# Imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
# Modulus is envelope of signal
u_coef, u_periods, u_scales, u_coi = datafunctions.compute_wavelet_components(
    u_zonal_perturbations, dj, dt, s0, mother_wavelet, spatial_resolution, padding
)
v_coef, v_periods, v_scales, v_coi = datafunctions.compute_wavelet_components(
    v_meridional_perturbations, dj, dt, s0, mother_wavelet, spatial_resolution, padding
)
t_coef, t_periods, t_scales, t_coi = datafunctions.compute_wavelet_components(
    temperature_perturbations, dj, dt, s0, mother_wavelet, spatial_resolution, padding
)

# [Koushik et al, 2019] -- Power surface is sum of squares of the U and V wavelet transformed surfaces
# S(a,z) = abs(U(a,z))^2  + abs(V(a,z))^2; a = vertical wavelength, z = height
power = abs(u_coef) ** 2 + abs(v_coef) ** 2  # [m^2/s^2]

# Calculate the significance of the wavelet coefficients; LAG 1 autocorrelation
alpha_u = datafunctions.acorr(
    u_zonal_perturbations.values, lags=range(len(u_zonal_perturbations.values))
)[1]

alpha_v = datafunctions.acorr(
    v_meridional_perturbations.values,
    lags=range(len(v_meridional_perturbations.values)),
)[1]

signif = wave_signif(
    u_zonal_perturbations,
    dt,
    u_scales,
    lag1=alpha_u,
) + wave_signif(
    v_meridional_perturbations,
    dt,
    v_scales,
    lag1=alpha_v,
)

# Match shape of power array
signif = np.ones([1, power.shape[1]]) * signif[:, None]
# Create boolean mask that is True where power is significant and False otherwise
signif = power > signif

# Match shape of power array
coiMask = np.array(
    [
        np.array(u_periods) <= (u_coi[i])
        for i in range(power.shape[1])
    ]
).T

################### Find Local Maxima & Extract Boundaries Around Gravity Wave Packet ###################

# Extract coordinates of the local maxima above a threshold and within the cone of influence and signifance levels
peaks = datafunctions.find_local_maxima(power, 0.011, coiMask, signif)

peak_nom = 3

peak_containers, boundary_rows, boundary_cols = datafunctions.extract_boundaries_around_peak(power, peaks, peak_nom)

associated_timestamps_range_of_boundary = choose_data_frame_analyze["Time [UTC]"].iloc[boundary_cols] # TimeStamps [UTC]
associated_height_range_of_boundary =  choose_data_frame_analyze["Geopot [m]"].iloc[boundary_cols] # m

associated_height_of_peak = choose_data_frame_analyze["Geopot [m]"].iloc[peaks[peak_nom][1]] # m
associated_time_of_peak = choose_data_frame_analyze["Time [UTC]"].iloc[peaks[peak_nom][1]] # TimeStamp [UTC]

z_index_of_max_local_power = peaks[peak_nom][1] # corresponds to index of the max height
a_index_of_max_local_power = peaks[peak_nom][0] # corresponds to the index of the max vertical wavelength

# Determine if other local maxima are found within the rectangualr boundary
peaks_within_boundaries = datafunctions.peaks_inside_rectangular_boundary(peaks, boundary_rows, boundary_cols)

################### Plot Power Surface ###################

colormap = cm.eclipse

plottingfunctions.plot_power_surface(
    choose_data_frame_analyze["Geopot [m]"]/1000,
    power,
    u_periods,
    peak_containers,
    signif,
    coiMask,
    peaks,
    colormap,
    starting_time_for_flight,
    path_to_save_figures + "/Power_Surfaces/",
    save_fig=False,
)

################### Inverse Wavelet Transform ###################

# [Zink and Vincent, 2001] -- Reconstruct perturbations associated with the gravity wave packet within the determined rectangular boundary
u_inverted_coeff = datafunctions.inverse_wavelet_transform(u_coef,peak_containers,u_scales,dj,dt)
v_inverted_coeff = datafunctions.inverse_wavelet_transform(v_coef,peak_containers,v_scales,dj,dt)
t_inverted_coeff = datafunctions.inverse_wavelet_transform(t_coef,peak_containers,t_scales,dj,dt)

# Calculate the horizontal wind variance [m^2/s^2]
horizontal_wind_variance = datafunctions.calculate_horizontal_wind_variance(u_inverted_coeff, v_inverted_coeff,peaks_within_boundaries,peaks,peak_nom)

# [Zink and Vincent, 2001] -- vertical extent of the gravity wave packet is associated with the FWHM of the horizontal wind variance
vertical_extent_coordx, vertical_extent_coordy, max_value_index, half_max  = datafunctions.wave_packet_FWHM_indices(horizontal_wind_variance)

# Quick plot of the horizontal wind variance
# plottingfunctions.plot_FWHM_wind_variance(horizontal_wind_variance,vertical_extent_coordx, vertical_extent_coordy,max_value_index,half_max)

# Only consider the perturbations associated with the vertical extent of the wave packet
iu_wave = (u_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
iv_wave = (v_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
it_wave = (t_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]

################### Hodograph Analysis ###################

plottingfunctions.plot_hodograph(iu_wave.real, iv_wave.real,choose_data_frame_analyze)
# plottingfunctions.winds_associated_with_dominant_vertical_wavelengths(iu_wave.real, iv_wave.real,(choose_data_frame_analyze["Geopot [m]"]/1000).iloc[vertical_extent_coordx:vertical_extent_coordy])

################### Extracting Wave Parameters ###################

# Calculate Stokes Parameters for the wave packet
# [Pfenninger et. al, 1999] -- Eqn. 9
# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
hilbert_v = iv_wave.imag

# Represent the total energy/variance
Stokes_I = np.mean(iu_wave.real**2) + np.mean(iv_wave.real**2)
# Variance difference/ Axial anisotropy
Stokes_D = np.mean(iu_wave.real**2) - np.mean(iv_wave.real**2)
# In-phase covariance; associated with linear polarization
Stokes_P = 2 * np.mean(iu_wave.real * iv_wave.real)
# Quadrature covariance; associated with circular polarization
Stokes_Q = 2 * np.mean(iu_wave.real * hilbert_v)

# [Pfenninger et. al, 1999] -- think of Stokes Q as the energy difference betwen propagating AGWs
# [Koushik et. al, 2019] -- Stokes  Q -- sense of rotation of the polarization ellipse
if Stokes_Q > 0:
    print("\n")
    print("Stokes Q is positive; Wave energy propagating upward")
    print("Clockwise rotation of the polarized ellipse")
else:
    print("\n")
    print("Stokes Q is negative; Wave energy propagating downwards")
    print("Anticlockwise rotation of the polarized ellipse")

# [Koushik et. al, 2019] -- Stokes P or Q less than threshold value might not be not agws
if np.abs(Stokes_P) < 0.05 or np.abs(Stokes_Q) < 0.05:
    print("\n")
    print("Might not be an AGW; representative of poor wave activity")


# degree of polarization -- stastical measure of the coherence of the wave field
polarization_factor = np.sqrt(Stokes_P**2 + Stokes_D**2 + Stokes_Q**2) / Stokes_I

# [Yoo et al, 2018] -- conditions for the wave packet
if 0.5 <= polarization_factor < 1:
    print("\n")
    print("d = %.2f -- Most likely an AGW"%polarization_factor)
elif polarization_factor == 1:
    print("\n")
    print("d = %.2f -- Monochromatic Wave"%polarization_factor)
elif polarization_factor == 0:
    print("\n")
    print("d = %.2f -- Not an AGW; no polarization relationship"%polarization_factor)
else:
    print("\n")
    print("d = %.2f -- Might not be an AGW. Polarization factor too low (d < 0.05) or unrealistic (d > 1) value"%polarization_factor)

# Pfenninger et. al, 1999] -- Eqn. 11
# Orientation of major axis - direction of propagation of gravity wave (deg clockwise from N)
# Measured anticlockwise from the x-axis
# 180 deg ambiguity
theta = np.arctan2(Stokes_P, Stokes_D) / 2

print("%.2f degree measured anticlockwise from x-axis"%np.rad2deg(theta))

# [Pfenninger et. al, 1999] -- Determine orientaiton of major axis and remove ambiguity
U_prime = iu_wave.real * np.cos(theta) + iv_wave.real * np.sin(theta)

# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted (Hilbert transform)
hilbert_t = it_wave.imag

# [Koushik et. al, 2019] -- sign determines direction of propagaton
# https://digital.library.adelaide.edu.au/dspace/bitstream/2440/19619/1/01front.pdf
sign = U_prime * hilbert_t

if np.sum(sign < 0, axis=0) > 0:
    theta = theta + np.pi
    print("\n")
    print("Counterclockwise rotation from x-axis")
    print("Propagation direction is in %.2f degree direction"% (np.rad2deg(theta)))

elif np.sum(sign > 0, axis=0)> 0:
    theta = theta
    print("\n")
    print("Clockwise rotation from x-axis")
    print("Propagation direction is in %.2f degree direction"% (np.rad2deg(theta)))
    theta = theta

# [Pfenninger et. al, 1999] -- Coriolis force causes wind vectors to rotate with height & form an ellipse
latitude_artesia = 32.842258  # latitude of location [degrees]
omega_Earth = 7.29e-5  # Rotation rate of earth [radians /seconds]
# [Fritts and Alexander, 2003] -- Coriolis Force
f_coriolis = 2 * omega_Earth * np.sin(np.deg2rad(latitude_artesia)) # [rad/s]

# [Koushik et. al, 2019] -- Eqn 8
# axial ratio -- ratio of the major axis to minor axis
eta = (0.5) * np.arcsin(
    Stokes_Q / np.sqrt(Stokes_D**2 + Stokes_P**2 + Stokes_Q**2)
)

# [Yoo et al, 2018] -- Eqn 6 & 7
# [Koushik et. al, 2019] -- typical values of inverse axial ratios [1.0, 3.5]; median: 1.4
axial_ratio = np.tan(eta)
inverse_axial_ratio = np.abs(1/ axial_ratio) # [rad/s]

# Eqn. 8 [Koushik et. al, 2019] -- intrinsic fequency: frequency observed in the reference frame moving with the background wind
intrinsic_frequency = f_coriolis * inverse_axial_ratio # [rad/s]

# intrisince period
intrinsic_period = 2*np.pi/intrinsic_frequency # [s]

# https://digital.library.adelaide.edu.au/dspace/bitstream/2440/19619/1/01front.pdf
wind_vectors = [iu_wave.real, iv_wave.real]
rotation_matrix = [  [np.cos(theta), np.sin(theta)], [-np.sin(theta),np.cos(theta)]    ]
u_parallel_v_perendicular = np.dot(rotation_matrix,wind_vectors)

# Phase difference between parallel zonal winds and temperature
# zonal and temperature perturbations should be 90 deg out of phase
gamma = np.mean(u_parallel_v_perendicular[0]*np.conj(it_wave))/ np.sqrt( np.mean( np.abs(u_parallel_v_perendicular[0])**2) * np.mean( np.abs(it_wave)**2)   )
phase_difference_gamma_in_degrees = np.angle(gamma,deg=True)

# if phase_difference_gamma_in_degrees < 0:
#         theta = theta + np.pi
        
# Constants
grav_constant = 9.81  # gravity [m/s^2]
ps = 1000  # standard pressure [hPa] -- equal to 1 millibar
kappa = 2 / 7 # Poisson constant for dry air 
celsius_to_kelvin_conversion = 273.15 # 0 deg Celsius = 273.15 K

# Convert temperature array from Celsium to Kelvin
temperature_K =  (choose_data_frame_analyze["T [°C]"] + celsius_to_kelvin_conversion)

# Potential temperature -- temperature a parcel of air would have if moved adiabatically (no heat exchange)
# Eqn. 1 from [Pfenninger et. al, 1999]
potential_temperature = (
    (temperature_K)
    * (ps / choose_data_frame_analyze["P [hPa]"]) ** kappa
)  # [K]

# Mean buoyancy frequency -- describs the stability of the region
# Eqn. 4 from [Pfenninger et. al, 1999]
mean_buoyancy_frequency = np.sqrt(
    (grav_constant/ potential_temperature) * np.gradient(potential_temperature, spatial_resolution)
) # [Hz] = [rad/s]

# average over boundary which wave was detected
# Upper frequency for the gravity waves
mean_buoyancy_frequency = mean_buoyancy_frequency[vertical_extent_coordx:vertical_extent_coordy].mean(skipna=True) # [Hz] = [rad/s]

# Mean buoyancy period
mean_buoyancy_period = (2 * np.pi) / mean_buoyancy_frequency # [s]

## other checck
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/97JD03325
#limit
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014JD022448 figure 1b
#mean_buoyancy_frequency > intrinsic_frequency > intrinsic_frequency

if not mean_buoyancy_frequency > intrinsic_frequency > f_coriolis:
    print("Not physical")

height_range_over_vertical_extent = choose_data_frame_analyze["Geopot [m]"][vertical_extent_coordx:vertical_extent_coordy]
u_zonal_perturbations_over_vertical_extent = u_zonal_perturbations[vertical_extent_coordx:vertical_extent_coordy]
v_meridional_perturbations_over_vertical_extent = v_meridional_perturbations[vertical_extent_coordx:vertical_extent_coordy]

# dynamic shear instability -- Richardson Number
richardson_number = mean_buoyancy_frequency**2 /( np.gradient(u_zonal_perturbations_over_vertical_extent,height_range_over_vertical_extent)**2 + np.gradient(v_meridional_perturbations_over_vertical_extent,height_range_over_vertical_extent)**2   )
richardson_number = np.mean(richardson_number)

# [Pfenninger et. al, 1999] -
# if N^2 and R < 0
if mean_buoyancy_frequency**2 < 0 and richardson_number < 0:
    print("\n")
    print("Convectively unstable")
elif 0 < richardson_number < 0.25:
    print("\n")
    print("Dynamically unstable")

# vertical wavenumber and wavelength
vertical_wavenumber = (2 * np.pi) / u_periods[a_index_of_max_local_power] # [1/m]
vertical_wavelength = u_periods[a_index_of_max_local_power] # [m]

# horizontal wavenumber and wavelength [Murphy et al, 2014] -- Eqn B2
horizontal_wavenumber = (vertical_wavenumber/mean_buoyancy_frequency)*np.sqrt(intrinsic_frequency**2 - f_coriolis**2) # [1/m]
horizontal_wavelength = (2 * np.pi) / horizontal_wavenumber # [m]

# dispersion relation for low/medium-freqyuency gravity waves
# should be equal to the intrisince frequency
omega_squared = mean_buoyancy_frequency**2 * (horizontal_wavenumber**2 / vertical_wavenumber**2) + f_coriolis**2

if intrinsic_frequency != np.sqrt(omega_squared):
    print("Something isn't right")

# [Murphy et al, 2014] -- Eqn 2
# averages done over vertical span of the wave
vertical_extent_of_zonal_perturbation = np.mean((iu_wave.real)**2)
vertical_extent_of_meridional_perturbation = np.mean((iv_wave.real)**2)
vertical_extent_of_temperature = np.mean((it_wave.real / choose_data_frame_analyze["T [°C]"].iloc[vertical_extent_coordx:vertical_extent_coordy])**2)

# Eqn. 14 & 15 [Koushik et. al, 2019]
#  gravity wave kinetic/potential energy density [J/Kg]
kinetic_energy = (0.5) * (vertical_extent_of_zonal_perturbation + vertical_extent_of_meridional_perturbation)
potential_energy = (0.5) * (grav_constant**2/mean_buoyancy_frequency**2 )* vertical_extent_of_temperature

total_energy_of_packet =  kinetic_energy + potential_energy

# intrinsic vertical group velocity [m/s] 
# [Murphy et al, 2014] -- Eqn B5
cgz = -1 * ((intrinsic_frequency**2 - f_coriolis**2)/(intrinsic_frequency*vertical_wavenumber))

# zonal wavenumber -- [Murphy et al, 2014] -- Table 2
k = horizontal_wavenumber*np.sin(theta)

# meridional wavenumber -- [Murphy et al, 2014] -- Table 2
l = horizontal_wavenumber*np.cos(theta)

# intrinsice vertical phase speed [m/s] -- [Murphy et al, 2014] -- Table 2
cz = intrinsic_frequency/vertical_wavenumber

# intrinsic horizontal phase speed [m/s] -- [Murphy et al, 2014] -- Table 2
c_hor = intrinsic_frequency/horizontal_wavenumber

# intrinsic zonal phase speed [m/s] -- [Murphy et al, 2014] -- Table 2
cx = intrinsic_frequency/k

# intrinsice meridional phase speed [m/s] -- [Murphy et al, 2014] -- Table 2
cy = intrinsic_frequency/l

# intrinsic zonal group velocity [m/s] -- [Murphy et al, 2014] -- Table 2
cgx = k*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# intrinsic meridional group velocity [m/s] -- [Murphy et al, 2014] -- Table 2
cgy = l*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# intrinsic horizontal group velocity [m/s] -- [Murphy et al, 2014] -- Table 2
cgh = np.sqrt(cgx**2 + cgy**2)

# Ideal gas law
# specific gas constant of dry air
Rconst = 287 # J/kg/K
rho = (choose_data_frame_analyze["P [hPa]"]**100)/(Rconst *temperature_K) # density [kg/m^3]

# momentum flux [Pa] - [m^2/s^2]
zonal_momentum_flux = -rho * (intrinsic_frequency*grav_constant/mean_buoyancy_frequency**2)*np.mean(iu_wave.real*it_wave.imag/ choose_data_frame_analyze["T [°C]"].iloc[vertical_extent_coordx:vertical_extent_coordy])
meridional_momentum_flux = -rho * (intrinsic_frequency*grav_constant/mean_buoyancy_frequency**2)*np.mean(iv_wave.real*it_wave.imag/ choose_data_frame_analyze["T [°C]"].iloc[vertical_extent_coordx:vertical_extent_coordy])


data_dictionary = {}


# Fitting an ellipse
coeffs = datafunctions.fit_ellipse(iu_wave.real,iv_wave.real)
print('Fitted parameters:')
print('a, b, c, d, e, f =', coeffs)
x0, y0, ap, bp, e, phi = datafunctions.cart_to_pol(coeffs)
print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)

plt.plot(iu_wave.real,iv_wave.real, 'x')     # given points
xn, yn = datafunctions.get_ellipse_pts((x0, y0, ap, bp, e, phi))
plt.plot(xn, yn)
plt.plot(x0,y0,color='r',marker='x')
plt.axhline(y=y0,linestyle='--', color='k')
plt.axvline(x=x0,linestyle='--', color='k')

# mag = axial_ratio
# U = mag * np.cos( np.deg2rad(phase_difference_gamma_in_degrees ) )
# V = mag * np.sin( np.deg2rad(phase_difference_gamma_in_degrees ) )

# plt.quiver(x0,y0, U, V, color='r', width=0.003)

# mag = inverse_axial_ratio
# U = mag * np.cos( theta ) 
# V = mag * np.sin( theta ) 
# plt.quiver(x0,y0, U, V, color='k', width=0.003)



plt.show()