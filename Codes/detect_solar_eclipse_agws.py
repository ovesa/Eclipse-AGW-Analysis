################### Necessary Libraries ###################

import numpy as np
import glob
import cmasher as cm
import matplotlib
import plottingfunctions
import datafunctions
from waveletFunctions import wave_signif
import pandas as pd
from tqdm import tqdm
# plt.ion()

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
path_to_save_wave_results = "/media/oana/Data1/Annular_Eclipse_Analysis/"

# Select all xls files that match
fname = glob.glob(path + "*end.xls")

file_nom = 6

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
print("\n")
print("Altitude range: [%s, %s] m" % (original_min_altitude,original_max_altitude))
print("\n")

################### Preprocess and Interpolate Data ###################

# tropopause height [m]
tropopause_height = 13000 

# to give a rise rate of 5 m/s
spatial_resolution = 5

# Spatial height interpolation limit [m]
# Tells code to not interpolate if there are more than [interpolation_limit] consecutive rows of missing/ NaN values
# Interpolation starts anew after hitting the interpolation limit
interpolation_limit = 1000

# Cleaning up the data
dat = datafunctions.clean_data(dat, tropopause_height, original_data_shape)

# Linearly interpolate the data to create an evenly spaced spatial grid [m]
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
# By only choosing the first half of the dataset before the NaNs, this avoids the problem
choose_data_frame_analyze = dat[0]

# New time column with the date and time in UTC
# 10 columns to 11 columns
choose_data_frame_analyze = datafunctions.convert_seconds_to_timestamp(choose_data_frame_analyze, starting_time_for_flight)

################### Zonal & Meridional Components of Wind Speed ###################

# Grab the wind speeds and temperatures needed for analysis
(
    u_zonal_speed_array,
    v_meridional_speed_array,
    temperature_C,
) = datafunctions.extract_wind_components_and_temperature(choose_data_frame_analyze)

# Grab the height array in km
height_km = choose_data_frame_analyze["Geopot [m]"]/1000 # [km]

# Grab time array in UTC
time_UTC = choose_data_frame_analyze["Time [UTC]"]

################### Calculate First-Order Perturbations ###################

# Fit the wind speeds and the temperature 
v_meridional_fit = datafunctions.compute_polynomial_fits(
    height_km, v_meridional_speed_array, 2
)

u_zonal_fit = datafunctions.compute_polynomial_fits(
    height_km, u_zonal_speed_array, 2
)

temperature_fit = datafunctions.compute_polynomial_fits(
    height_km, temperature_C, 2
)

# Subtract the polynomial fits from the original vertical profiles to obtain the first-order perturbations
u_zonal_perturbations = datafunctions.derive_first_order_perturbations(
    height_km, u_zonal_speed_array, u_zonal_fit
)

v_meridional_perturbations = datafunctions.derive_first_order_perturbations(
    height_km, v_meridional_speed_array, v_meridional_fit
)

temperature_perturbations = datafunctions.derive_first_order_perturbations(
    height_km, temperature_C, temperature_fit
)

# Quick Plot of First-Order Perturbations Vertical Profiles
plottingfunctions.plot_vertical_profiles_with_residual_perturbations(
    height_km,
    u_zonal_speed_array,
    v_meridional_speed_array,
    temperature_C,
    v_meridional_fit,
    u_zonal_fit,
    temperature_fit,
    u_zonal_perturbations,
    v_meridional_perturbations,
    temperature_perturbations,
    time_UTC.iloc[0],
    path_to_save_figures + "First_Order_Perturbations",save_fig=True,)

################### Wavelet Analysis ###################
# Wavelet Transform will isolate wave packets in wavenumber vs height space

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
    u_zonal_perturbations, dj, dt, s0, mother_wavelet, padding
)
v_coef, v_periods, v_scales, v_coi = datafunctions.compute_wavelet_components(
    v_meridional_perturbations, dj, dt, s0, mother_wavelet, padding
)
t_coef, t_periods, t_scales, t_coi = datafunctions.compute_wavelet_components(
    temperature_perturbations, dj, dt, s0, mother_wavelet, padding
)

# [Koushik et al., 2019] -- Power surface is sum of squares of the U and V wavelet transformed surfaces
# S(a,z) = abs(U(a,z))^2  + abs(V(a,z))^2; a = vertical wavelength, z = height
power = abs(u_coef) ** 2 + abs(v_coef) ** 2  # [m^2/s^2]

# Calculate the significance of the wavelet coefficients; LAG 1 autocorrelation
alpha_u = datafunctions.acorr(
    u_zonal_perturbations, lags=range(len(u_zonal_perturbations))
)[1]

alpha_v = datafunctions.acorr(
    v_meridional_perturbations,
    lags=range(len(v_meridional_perturbations)),
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
# True where power is significant and False otherwise
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
local_maxima_coords = datafunctions.find_local_maxima(power, 0.011, coiMask, signif)

# Loop through all local_maxima_coords
# for nom in tqdm(range(0,1)):
wave_nom = 1 #nom

local_max = local_maxima_coords[wave_nom]

# Create a rectangular boundary around the local maxima that fulfills the required criteria
rectangular_boundary_container, boundary_rows, boundary_cols = datafunctions.extract_boundaries_around_peak(power, local_maxima_coords, wave_nom)

# Index of the max height
z_index_of_max_local_power = local_maxima_coords[wave_nom][1]
# Index of the max scale/vertical wavelength
a_index_of_max_local_power = local_maxima_coords[wave_nom][0] 

# Determine if other local maxima are present within the rectangualr boundary
local_max_within_boundaries = datafunctions.peaks_inside_rectangular_boundary(local_maxima_coords, [boundary_rows[0],boundary_rows[-1]], [boundary_cols[0],boundary_cols[-1]])

################### Plot Power Surface ###################

colormap = cm.eclipse

plottingfunctions.plot_power_surface(
    height_km,
    power,
    u_periods,
    rectangular_boundary_container,
    signif,
    coiMask,
    local_maxima_coords,
    colormap,
    starting_time_for_flight, wave_nom,
    path_to_save_figures + "Power_Surfaces",
    save_fig=True,
)

################### Inverse Wavelet Transform ###################

# [Zink and Vincent, 2001] -- Reconstruct perturbations associated with the gravity wave packet within the determined rectangular boundary
u_inverted_coeff = datafunctions.inverse_wavelet_transform(u_coef,rectangular_boundary_container,u_scales,dj,dt)
v_inverted_coeff = datafunctions.inverse_wavelet_transform(v_coef,rectangular_boundary_container,v_scales,dj,dt)
t_inverted_coeff = datafunctions.inverse_wavelet_transform(t_coef,rectangular_boundary_container,t_scales,dj,dt)

# Calculate the horizontal wind variance [m^2/s^2]
horizontal_wind_variance = datafunctions.calculate_horizontal_wind_variance(u_inverted_coeff, v_inverted_coeff,local_max_within_boundaries,local_maxima_coords,wave_nom)

# [Zink and Vincent, 2001] -- vertical extent of the gravity wave packet is associated with the FWHM of the horizontal wind variance
vertical_extent_coordx, vertical_extent_coordy, max_value_index, half_max  = datafunctions.wave_packet_FWHM_indices(horizontal_wind_variance)

# Quick plot of the horizontal wind variance
plottingfunctions.plot_FWHM_wind_variance(horizontal_wind_variance,vertical_extent_coordx, vertical_extent_coordy,max_value_index,half_max,starting_time_for_flight,wave_nom,path_to_save_figures + "Wind_Variance",save_fig=True)

# Only consider the perturbations associated with the vertical extent of the wave packet
iu_wave = (u_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
iv_wave = (v_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
it_wave = (t_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]

# The height array within the vertical bounds of the wave packet
height_km_vertical_extent = height_km.iloc[vertical_extent_coordx:vertical_extent_coordy]

# The time array within the vertical bounds of the wave packet
time_UTC_vertical_extent = time_UTC.iloc[vertical_extent_coordx:vertical_extent_coordy]

# The zonal and meridional perturbations within the vertical bounds of the wave packet
u_zonal_perturbations_vertical_extent = u_zonal_perturbations.iloc[vertical_extent_coordx:vertical_extent_coordy]
v_meridional_perturbations_vertical_extent = v_meridional_perturbations.iloc[vertical_extent_coordx:vertical_extent_coordy]

# Pressure array and pressure array within vertical bounds of wave packet
pressure_array = choose_data_frame_analyze["P [hPa]"]
pressure_vertical_extent = pressure_array.iloc[vertical_extent_coordx:vertical_extent_coordy]

################### Extracting Wave Parameters ###################

# Constants
grav_constant = 9.81  # gravity [m/s^2]
ps = 1000  # standard pressure [hPa] -- equal to 1 millibar
kappa = 2 / 7 # Poisson constant for dry air 
celsius_to_kelvin_conversion = 273.15 # 0 deg Celsius = 273.15 K

# Convert temperature array from Celsius to Kelvin
temperature_K =  temperature_C.apply(lambda x: x + celsius_to_kelvin_conversion)

# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
hilbert_v = iv_wave.imag
hilbert_t = it_wave.imag

# Calculate Stokes Parameters for the wave packet
# [Pfenninger et al., 1999] -- Eqn. 9
# Represent the total energy/variance
Stokes_I = np.mean(iu_wave.real**2) + np.mean(iv_wave.real**2)
# Variance difference/ Axial anisotropy
Stokes_D = np.mean(iu_wave.real**2) - np.mean(iv_wave.real**2)
# In-phase covariance; associated with linear polarization
Stokes_P = 2 * np.mean(iu_wave.real * iv_wave.real)
# Quadrature covariance; associated with circular polarization
Stokes_Q = 2 * np.mean(iu_wave.real * hilbert_v)

# [Pfenninger et al., 1999] -- think of Stokes Q as the energy difference betwen propagating AGWs
# [Koushik et al., 2019] -- Stokes  Q -- sense of rotation of the polarization ellipse
if Stokes_Q > 0:
    print("\n")
    print("Stokes Q is positive; Wave energy propagating upward")
    print("\n")
    print("Clockwise rotation of the polarized ellipse")
    print("\n")
    ellipse_rotation = "Clockwise Rotation"
else:
    print("\n")
    print("Stokes Q is negative; Wave energy propagating downwards")
    print("\n")
    print("Anticlockwise rotation of the polarized ellipse")
    print("\n")
    ellipse_rotation = "Anticlockwise Rotation"

# [Koushik et al., 2019] -- Stokes P or Q less than threshold value might not be not agws
# [Murphy et al., 2014] -- Stokes P and Q show the covariance between u parallel and v perpendicular and u parallel and phase shifted v perpendicular
if np.abs(Stokes_P) < 0.05 or np.abs(Stokes_Q) < 0.05:
    print("\n")
    print("Might not be an AGW; representative of poor wave activity")
    print("\n")
    condition2 = "Fail"
else:
    print("\n")
    condition2 = "Pass"

# degree of polarization -- stastical measure of the coherence of the wave field
polarization_factor = np.sqrt(Stokes_P**2 + Stokes_D**2 + Stokes_Q**2) / Stokes_I

# [Yoo et al., 2018] -- conditions for the wave packet
if 0.5 <= polarization_factor < 1:
    print("\n")
    print("d = %.4f -- Most likely an AGW"%polarization_factor)
    print("\n")
    condition3 = "Gravity Wave"
elif polarization_factor == 1:
    print("\n")
    print("d = %.2f -- Monochromatic Wave"%polarization_factor)
    print("\n")
    condition3 = "Monochromatic Wave"
elif polarization_factor == 0:
    print("\n")
    print("d = %.2f -- Not an AGW; no polarization relationship"%polarization_factor)
    print("\n")
    condition3 = "Not a Wave"
else:
    print("\n")
    print("d = %.4f -- Might not be an AGW. Polarization factor too low (d < 0.05) or unrealistic (d > 1) value"%polarization_factor)
    print("\n")
    condition3 = "Might not be a Wave"

# Pfenninger et al., 1999] -- Eqn. 11
# Orientation of major axis - direction of propagation of gravity wave (deg clockwise from N)
# Measured anticlockwise from the x-axis (180 deg ambiguity)
theta = np.arctan2(Stokes_P, Stokes_D) / 2

# [Zink 2000] -- phase difference between u|| and T' resolves direction ambiguity (90 deg out of phase)
# [Zink 2000] -- Eqn 3.15 -- velocities parallel and perpendicular to the GW alignment
wind_vectors = [iu_wave.real, iv_wave.real]
rotation_matrix = [  [np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)] ]
transformed_wind_vectors = np.dot(rotation_matrix,wind_vectors)

# [Zink 2000] -- Eqn 3.17
phase_difference_Uparallel_T = np.mean(transformed_wind_vectors[0] * np.conj(it_wave) )/np.sqrt(  np.mean(np.abs(transformed_wind_vectors[0])**2) * np.mean( np.abs(it_wave**2)) ) 
angle_Uparallel_T = np.angle(phase_difference_Uparallel_T,deg=True)

# [Zink 2000]
if angle_Uparallel_T > 0:
    theta = theta
    print("\n")
    print("Propagation direction is in %.2f degree measured anticlockwise from x-axis"% (np.rad2deg(theta)))
    print("\n")
else:
    theta = theta + np.pi
    print("\n")
    print("Propagation direction is in %.2f degree measured anticlockwise from x-axis"% (np.rad2deg(theta)))
    print("\n")

# [Pfenninger et al., 1999] -- Coriolis force causes wind vectors to rotate with height & form an ellipse
latitude_artesia = 32.842258  # latitude of location [degrees]
omega_Earth = 7.29e-5  # Rotation rate of earth [radians /seconds]
# [Fritts and Alexander, 2003] -- Coriolis Force
f_coriolis = 2 * omega_Earth * np.sin(np.deg2rad(latitude_artesia)) # [rad/s]

# [Koushik et al., 2019] -- Eqn 8
# axial ratio -- ratio of the major axis to minor axis
eta = (0.5) * np.arcsin(Stokes_Q / np.sqrt(Stokes_D**2 + Stokes_P**2 + Stokes_Q**2))

# [Yoo et al., 2018] -- Eqn 6 & 7
# [Koushik et al., 2019] -- typical values of inverse axial ratios 1.0--3.5; median: 1.4
axial_ratio = np.tan(eta)
inverse_axial_ratio = np.abs(1 / axial_ratio) # [rad/s]

# Eqn. 8 [Koushik et al., 2019] -- intrinsic fequency: frequency observed in the reference frame moving with the background wind
intrinsic_frequency = f_coriolis * inverse_axial_ratio # [rad/s]

# Intrinsic period [s]
intrinsic_period = 2*np.pi/intrinsic_frequency # [s]

# Potential temperature -- temperature a parcel of air would have if moved adiabatically (no heat exchange)
# [Pfenninger et al., 1999] -- Eqn. 1 
potential_temperature = (
    (temperature_K)
    * (ps / pressure_array) ** kappa
)  # [K]

# Mean buoyancy frequency -- describs the stability of the region
# [Pfenninger et al., 1999] -- Eqn. 4 
# Ignore the error - some NaNs will be present
buoyancy_frequency = np.sqrt(
    (grav_constant/ potential_temperature) * np.gradient(potential_temperature, spatial_resolution)
) # [Hz] = [rad/s]

# Average over vertical extent of the potential gravity wave
# Upper frequency boundary for the gravity waves
mean_buoyancy_frequency = buoyancy_frequency.iloc[vertical_extent_coordx:vertical_extent_coordy].mean(skipna=True) # [Hz] = [rad/s]

# Mean buoyancy period [s]
mean_buoyancy_period = (2 * np.pi) / mean_buoyancy_frequency

# [Murphy et al., 2014] -- Physical boundary check for frequency
if not mean_buoyancy_frequency > intrinsic_frequency > f_coriolis:
    print("\n")
    print("Not physical frequency")
    condition4 = "Not Physical"
else:
    print("\n")
    print("Physical frequency")
    condition4 = "Physical"

# [Pfenninger et al., 1999] -- Eqn 15 -- Richardson Number 
# Dynamic shear instability function of wind shear and temperature gradient
richardson_number = buoyancy_frequency**2 /( np.gradient(u_zonal_perturbations,height_km/1000)**2 + np.gradient(v_meridional_perturbations,height_km/1000)**2   )
richardson_number = richardson_number.iloc[vertical_extent_coordx:vertical_extent_coordy].mean(skipna=True)

# Conditions from [Pfenninger et al., 1999] 
# [Galperin et al., 2007]
if mean_buoyancy_frequency**2 < 0 and richardson_number < 0:
    print("\n")
    print("Convectively unstable")
elif 0 < richardson_number < 0.25:
    print("\n")
    print("Atmospheric turbulence detected")

# Vertical wavenumber and wavelength
vertical_wavenumber = (2 * np.pi) / u_periods[a_index_of_max_local_power] # [1/m]
vertical_wavelength = u_periods[a_index_of_max_local_power] # [m]

# Horizontal wavenumber and wavelength [Murphy et al., 2014] -- Eqn B2
horizontal_wavenumber = (vertical_wavenumber/mean_buoyancy_frequency)*np.sqrt(intrinsic_frequency**2 - f_coriolis**2) # [1/m]
horizontal_wavelength = (2 * np.pi) / horizontal_wavenumber # [m]

# Dispersion relation for low/medium-freqyuency gravity waves
# Should be equal to the intrisinc frequency
omega_squared = mean_buoyancy_frequency**2 * (horizontal_wavenumber**2 / vertical_wavenumber**2) + f_coriolis**2

if intrinsic_frequency != np.sqrt(omega_squared):
    print("\n")
    print("Something isn't right. Intrinsic frequency doesn't equal omega")

# [Murphy et al., 2014] -- Eqn 2
# Averages done over vertical span of the wave
vertical_extent_of_zonal_perturbation = np.mean((iu_wave.real)**2)
vertical_extent_of_meridional_perturbation = np.mean((iv_wave.real)**2)
vertical_extent_of_temperature = np.mean((it_wave.real/temperature_C.iloc[vertical_extent_coordx:vertical_extent_coordy] )**2)

# Eqn. 14 & 15 [Koushik et al., 2019]
# Gravity wave kinetic [m^2/s^2] / potential energy density [m^2/s^2]
kinetic_energy = (0.5) * (vertical_extent_of_zonal_perturbation + vertical_extent_of_meridional_perturbation)
potential_energy = (0.5) * (grav_constant**2/mean_buoyancy_frequency**2 )* vertical_extent_of_temperature
total_energy_of_packet =  kinetic_energy + potential_energy

# Zonal wavenumber -- [Murphy et al., 2014] -- Table 2
k = horizontal_wavenumber*np.sin(theta)

# Meridional wavenumber -- [Murphy et al., 2014] -- Table 2
l = horizontal_wavenumber*np.cos(theta)

# Intrinsic vertical phase speed [m/s] -- [Murphy et al., 2014] -- Table 2
cz = intrinsic_frequency/vertical_wavenumber

# Intrinsic horizontal phase speed [m/s] -- [Murphy et al., 2014] -- Table 2
c_hor = intrinsic_frequency/horizontal_wavenumber

# Intrinsic zonal phase speed [m/s] -- [Murphy et al., 2014] -- Table 2
cx = intrinsic_frequency/k

# Intrinsic meridional phase speed [m/s] -- [Murphy et al., 2014] -- Table 2
cy = intrinsic_frequency/l

# [Murphy et al., 2014] -- Eqn B5 -- intrinsic vertical group velocity [m/s] 
cgz = -1 * ((intrinsic_frequency**2 - f_coriolis**2)/(intrinsic_frequency*vertical_wavenumber))

# Intrinsic zonal group velocity [m/s] -- [Murphy et al., 2014] -- Table 2
cgx = k*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# Intrinsic meridional group velocity [m/s] -- [Murphy et al., 2014] -- Table 2
cgy = l*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# Intrinsic horizontal group velocity [m/s] -- [Murphy et al., 2014] -- Table 2
cgh = np.sqrt(cgx**2 + cgy**2)

# Specific gas constant of dry air
Rconst = 287 # [J/kg/K]
# Pressure unit conversion [hPa -- J]
pressure_Joules = pressure_vertical_extent*100
# Ideal gas law
rho = (pressure_Joules)/(Rconst * temperature_K.iloc[vertical_extent_coordx:vertical_extent_coordy]) # density [kg/m^3]

# Momentum flux [Pa] -- [Murphy et al., 2014] -- Table 3
zonal_momentum_flux = -rho * (intrinsic_frequency*grav_constant/ buoyancy_frequency.iloc[vertical_extent_coordx:vertical_extent_coordy]**2)*np.mean(iu_wave.real*it_wave.imag/ temperature_C.iloc[vertical_extent_coordx:vertical_extent_coordy])
zonal_momentum_flux = zonal_momentum_flux[z_index_of_max_local_power]

meridional_momentum_flux = -rho * (intrinsic_frequency*grav_constant/ buoyancy_frequency[vertical_extent_coordx:vertical_extent_coordy]**2)*np.mean(iv_wave.real*it_wave.imag/ temperature_C.iloc[vertical_extent_coordx:vertical_extent_coordy])
meridional_momentum_flux = meridional_momentum_flux[z_index_of_max_local_power]

# Amplitude [m^2/s^2]
amplitude = power[wave_nom][z_index_of_max_local_power]


################### Hodograph Analysis ###################

plottingfunctions.perturbations_associated_with_dominant_vertical_wavelengths(iu_wave.real, iv_wave.real,it_wave.real,height_km_vertical_extent, starting_time_for_flight,wave_nom, path_to_save_figures + "Dominant_Vertical_Perturbations",save_fig=True)

# plottingfunctions.plot_hodograph(iu_wave.real, iv_wave.real,height_km_vertical_bounds, starting_time_for_flight,wave_nom, path_to_save_figures + "/Simple_Hodograph_Plot/",save_fig=False)

# Fitting an ellipse
coeffs = datafunctions.fit_ellipse(iu_wave.real,iv_wave.real)
print("\n")
print("Fitted parameters:")
# print('a, b, c, d, e, f =', coeffs)
x0, y0, ap, bp, e, phi = datafunctions.cart_to_pol(coeffs)
print('x0, y0, ap, bp, e, phi = ', x0, y0, ap, bp, e, phi)
fit_u, fit_v = datafunctions.get_ellipse_pts((x0, y0, ap, bp, e, phi))

fig = plottingfunctions.plot_hodograph_with_fitted_ellipse(iu_wave.real, iv_wave.real,height_km_vertical_extent, fit_u,fit_v,x0,y0, ellipse_rotation,inverse_axial_ratio,theta, starting_time_for_flight, wave_nom,  path_to_save_figures + "Hodograph_Analysis",
    save_fig=True)



extracted_wave_paramters = {"Wave": wave_nom+1, "Rotation": ellipse_rotation, "Condition 2": condition2, "Condition 3": condition3, "Condition 4": condition4, "Detection Height [km]": height_km.iloc[z_index_of_max_local_power], "Time Detected [UTC]": str(time_UTC.iloc[z_index_of_max_local_power]), "Amplitude [m^2/s^2]": amplitude, "Stokes I": Stokes_I, "Stokes D": Stokes_D, "Stokes Q": Stokes_Q, "Stokes P": Stokes_P, "polarization_factor": polarization_factor, "theta": theta, "omega [rad/s]": intrinsic_frequency, "T [min]": intrinsic_period/60, "N [rad/s]": mean_buoyancy_frequency, "N_period [min]": mean_buoyancy_period, "Richardson Number": richardson_number, "vertical wavenumber [1/m]": vertical_wavenumber, "vertical wavelength [m]": vertical_wavelength, "horizontal wavenumber [1/m]": horizontal_wavenumber, "horizontal wavelength [m]": horizontal_wavelength, "kinetic energy [m^2/s^2]": kinetic_energy, "potential_energy [m^2/s^2]": potential_energy,"Total Energy [m^2/s^2]": total_energy_of_packet, "Vertical Group Velocity [m/s]": cgz, "Horizontal Group Velocity [m/s]": cgh,"Vertical Phase Speed [m/s]": cz, "Horizontal Phase Speed [m/s]": c_hor, "Zonal Momentum Flux [Pa]": zonal_momentum_flux,"Meridional Momentum Flux [Pa]" :meridional_momentum_flux}

if wave_nom == 0:
    # Set up intial dataframe; only at the very beginning
    final_results = pd.DataFrame(columns=extracted_wave_paramters.keys())
    
wave_results = pd.DataFrame(extracted_wave_paramters, index=[0])  # the `index` argument is important 
# Append a row corresponding to the current wave packet being analyzed
final_results = pd.DataFrame(pd.concat([final_results,wave_results]))


## loop ends here

# Reset Index
final_results = final_results.reset_index(drop=True)

date_string = str(starting_time_for_flight.date())
date_string = date_string.replace("-", "")
time_string = str(starting_time_for_flight.time())
time_string = time_string.replace(":", "")

# Save wave results to xls file
print("\n")
print("Saving results for %s to xlsx format"%starting_time_for_flight)
final_results.to_excel(path_to_save_wave_results + "%s_%s_UTC_wave_results.xlsx"%(date_string,time_string))

