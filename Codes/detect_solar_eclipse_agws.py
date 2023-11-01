################### Necessary Libraries ###################

import numpy as np
import matplotlib.pyplot as plt
import glob
from waveletFunctions import wave_signif
import copy

import cmasher as cm
  
import matplotlib

import plottingfunctions
import datafunctions
import toml
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

file_nom = 5

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
spatial_resolution = 5 

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

# Quick Plot
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
# Isolate wave packets in wavenumber versus height space

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
# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
# [Zink and Vincent, 2001] -- modulus is envelope of signal
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

# Calculate the significance of the wavelet coefficients; LAG1- autocorrelation
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

# Turn 1D array into a 2D array matching shape of power surface array for direct comparison
signif = np.ones([1, choose_data_frame_analyze.shape[0]]) * signif[:, None]
# Create boolean mask that is True where power is significant and False otherwise
signif = power > signif

# Turn 1D array into a 2D array matching shape of power surface array for direct comparison
# I assume that the cone of influence should match the wave significance array in using both zonal and meridional wavelet coefficient perturbations
coiMask = np.array(
    [
        np.array(u_periods) <= (u_coi[i])
        for i in range(len(choose_data_frame_analyze["Geopot [m]"]))
    ]
).T

coi_1d =  v_coi 

################### Find Local Maxima & Extract Boundaries Around Gravity Wave Packet ###################

# Extract coordinates of the local maxima above a threshold and within the cone of influence and signifance levels
peaks = datafunctions.find_local_maxima(power, 0.011, coiMask, signif)

peak_nom = 0
peak_containers, boundary_rows, boundary_cols = datafunctions.extract_boundaries_around_peak(power, peaks, peak_nom)



associated_timestamps_range_of_boundary = choose_data_frame_analyze["Time [UTC]"].iloc[boundary_cols] # TimeStamps [UTC]
associated_height_range_of_boundary =  choose_data_frame_analyze["Geopot [m]"].iloc[boundary_cols] # m

associated_height_of_peak = choose_data_frame_analyze["Geopot [m]"].iloc[peaks[peak_nom][1]] # m
associated_time_of_peak = choose_data_frame_analyze["Time [UTC]"].iloc[peaks[peak_nom][1]] # TimeStamp [UTC]


z_index_of_max_local_power = peaks[peak_nom][1] # corresponds to the height
a_index_of_max_local_power = peaks[peak_nom][0] # corresponds to the vertical wavelength

################### Plot Power Surface ###################

colormap = cm.eclipse

plottingfunctions.plot_power_surface(
    choose_data_frame_analyze["Geopot [m]"]/1000,
    power,
    u_periods,
    peak_containers,
    signif,
    coiMask,coi_1d,
    peaks,
    colormap,
    starting_time_for_flight,
    path_to_save_figures + "/Power_Surfaces/",
    save_fig=False,
)

################### Inverse Wavelet Transform ###################

# [Zink and Vincent, 2001] -- Reconstruct zonal and meridional perturbations associated with the gravity wave packet 
# by using the inverse wavelet transform of the wavelet coefficients centered within the boundary
# Make everything outside of the rectangular boundary 0

u_inverted_coeff = copy.deepcopy(u_coef)
u_inverted_coeff = u_inverted_coeff*peak_containers

v_inverted_coeff = copy.deepcopy(v_coef)
v_inverted_coeff  = v_inverted_coeff*peak_containers

t_inverted_coeff = copy.deepcopy(t_coef)
t_inverted_coeff = t_inverted_coeff*peak_containers


# Inverse wavelet transform
# Want to use the exact parameters used in the initial calculation of the wavelet coefficients
# [Torrence and Compo, 1998] Eqn 11

u_div_scale = np.divide(u_inverted_coeff.T,np.sqrt(u_scales))
v_div_scale= np.divide(v_inverted_coeff.T,np.sqrt(v_scales))
t_div_scale = np.divide(t_inverted_coeff.T,np.sqrt(t_scales))

# [Torrence and Compo, 1998] Table 2
C_delta_morlet = 0.776 #  reconstruction factor
psi0_morlet = np.pi**(1/4) # to remove energy scaling
wavelet_constant = dj*np.sqrt(dt)/ (C_delta_morlet*psi0_morlet)

u_inverted_coeff = np.multiply(u_div_scale.sum(axis=0),wavelet_constant)
v_inverted_coeff = np.multiply(v_div_scale.sum(axis=0),wavelet_constant)
t_inverted_coeff = np.multiply(t_div_scale.sum(axis=0),wavelet_constant)

# [Zink and Vincent, 2001] -- vertical extent: the FWHM of the horizontal wind variance
# wind variance - the sum of the reconstructed u and v wavelet coefficients
horizontal_wind_variance = np.abs(u_inverted_coeff) ** 2 + np.abs(v_inverted_coeff) ** 2


x1, y1, w, h = boundary_rows[0], boundary_cols[0], boundary_rows[1] - boundary_rows[0],  boundary_cols[1]  - boundary_cols[0]
x2,y2 = x1+w,y1+h

peaks_within_boundary = []
## Find peaks within rectangular boundary:
for nom in peaks:
    if (x1 < nom[0] and nom[0] < x2):
            if (y1 < nom[1] and nom[1] < y2):
                peaks_within_boundary.append(nom)
                
                

# If peaks is equal to itself essentially
if len(peaks_within_boundary)==1 and np.array(peaks_within_boundary[0] == peaks[peak_nom]).all():
    peaks_within_boundary = peaks[peak_nom]
    horizontal_wind_variance = horizontal_wind_variance
else:
    horizontal_wind_variance = horizontal_wind_variance/2
    
    
# zink if boundary of reconstructed wavelets overlap, horizontal wind variance is divided in equal parts



# https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
# Find the maximum value
max_value = np.max(horizontal_wind_variance)
max_value_index = np.argmax(horizontal_wind_variance)

# Find the half maximum
half_max = max_value/2
# https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
# Find the indices where the values are closest to half the maximum on both sides of the peak
left_index = next( ( i for i in range(max_value_index,-1,-1) if horizontal_wind_variance[i] <= half_max), 0)
right_index = next((i for i in range(max_value_index, len(horizontal_wind_variance)) if horizontal_wind_variance[i] <= half_max), len(horizontal_wind_variance) - 1)

plt.figure()
plt.plot(np.arange(len(horizontal_wind_variance)), horizontal_wind_variance, color='k',zorder=0,)
plt.scatter(left_index, horizontal_wind_variance[left_index], s= 30, color='red', edgecolor='k',zorder=1)
plt.scatter(right_index, horizontal_wind_variance[right_index], s=30,  color='red', edgecolor='k',zorder=1)
plt.scatter(max_value_index, horizontal_wind_variance[max_value_index], s=30,  color='gold', edgecolor='k',zorder=1)
plt.axhline(y=half_max, linestyle='--', color='navy')
plt.xlim([max_value_index-100,max_value_index+100])
plt.ylabel(r"Horizontal Wind Variance [m$^2$/s$^2$]")
plt.xlabel("Arb")
plt.tight_layout()
plt.show()




 
vertical_extent_coordx, vertical_extent_coordy = left_index,right_index
# The reconstructed wind and temperature paramters based on the full width half max
iu_wave = (u_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
iv_wave = (v_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]
it_wave = (t_inverted_coeff)[vertical_extent_coordx:vertical_extent_coordy]









################### Hodograph Analysis ###################

plottingfunctions.plot_hodograph(iu_wave.real, iv_wave.real,choose_data_frame_analyze)
plottingfunctions.winds_associated_with_dominant_vertical_wavelengths(iu_wave.real, iv_wave.real,(choose_data_frame_analyze["Geopot [m]"]/1000).iloc[vertical_extent_coordx:vertical_extent_coordy])

################### Extracting Wave Parameters ###################

# Constants
grav_constant = 9.81  # gravity [m/s^2]
ps = 1000  # standard pressure [hPa] -- equal to 1 millibar
kappa = 2 / 7 # Poisson constant for dry air 
celsius_to_kelvin_conversion = 273.15 # 0 deg Celsius == 273.15 K

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


## Stokes parameters for gravity waves
# Eqn. 9 from [Pfenninger et. al, 1999]
# Hilbert transform of the meridional wind component - adds a 90 deg phase shift to vertical profile
# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
hilbert_v = iv_wave.imag

# Represent the total energy
Stokes_I = np.mean(iu_wave.real**2) + np.mean(iv_wave.real**2)
# variance difference/ linear polarization measure
Stokes_D = np.mean(iu_wave.real**2) - np.mean(iv_wave.real**2)
# P and Q are the in phase and quadrature components between wave components
Stokes_P = 2 * np.mean(iu_wave.real * iv_wave.real)
# measure of the circular polarization
# Sense rotation of the polarized ellipse
Stokes_Q = 2 * np.mean(iu_wave.real * hilbert_v)

# [Pfenninger et. al, 1999] -- think of Stokes Q as the energy difference betwen propagating AGWs
# [Koushik et. al, 2019] -- Stokes  Q -- sense of rotation of th[boundary_cols[0]:boundary_cols[1]]e polarizaition ellipse
if Stokes_Q > 0:
    print("\n")
    print("Wave energy propagating upward")
    print("Clockwise rotation of the polarized ellipse")
    print("Stokes Q is positive")
else:
    print("\n")
    print("Wave energy propagating downwards")
    print("Anticlockwise rotation of the polarized ellipse")
    print("Stokes Q is negative")

# [Koushik et. al, 2019]  -- stokes p or q less than threshold value might not be not agws
if np.abs(Stokes_P) < 0.05 or np.abs(Stokes_Q) < 0.05:
    print("\n")
    print("Might not be an AGW; representative of poor wave activity")


# stastical measure of the coherence of the wave field
# degree of polarization
polarization_factor = np.sqrt(Stokes_P**2 + Stokes_D**2 + Stokes_Q**2) / Stokes_I

# [Yoo et al, 2018]
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
    print("d = %.2f -- Might not be an AGW. Polarization factor too low or unrealistic value"%polarization_factor)


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

# Eqn. 11 from [Pfenninger et. al, 1999]
# Orientation of major axis of ellipse
# direction of the horizontal component of the wave vector
# oriented parallel to GW vector
# theta is measured anticlockwise from the x-axis
# 180 deg ambiguity
# horizontal direction of propagation of GW (deg clockwise from N)
theta = np.arctan2(Stokes_P, Stokes_D) / 2

# phase difference
# [Eckerman and Vincent, 1989] -- Eqn 7
phase_difference = np.arctan2(Stokes_Q,Stokes_P)

# Coriolis Force
# Omega is rotation rate of earth
# [Pfenninger et. al, 1999] -- Coriolis forced causes wind vectors to rotate with height & form an ellipse
latitude_artesia = 32.842258  # degrees
omega_Earth = 7.29e-5  # [radians /seconds]
# [Fritts and Alexander, 2003]
f_coriolis = 2 * omega_Earth * np.sin(np.deg2rad(latitude_artesia)) # [rad/s]


# remove ambiguity
# wind found in the major axis direction
# [Pfenninger et. al, 1999]
U_prime = iu_wave.real * np.cos(theta) + iv_wave.real * np.sin(theta)

# Hilbert transform of the temperature perturbation
# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)
hilbert_t = it_wave.imag

# sign determines direction of propagaton
sign = U_prime * hilbert_t

# [Koushik et. al, 2019]
if np.sum(sign < 0, axis=0) > 0:
    print("\n")
    print("sign of direction is negative")
    theta = np.deg2rad(np.rad2deg(theta) + 180 )

elif np.sum(sign > 0, axis=0)> 0:
    print("\n")
    print("sign of direction is positive")
    theta = theta


# coherencey, correlated power between U and V
C = np.sqrt((Stokes_P**2 + Stokes_Q**2) / (Stokes_I**2 - Stokes_D**2))

# axial ratio -- ratio of the minor axis to the maxjor axis; aspect ratio of the polarization ellipse
# Eqn. 8 [Koushik et. al, 2019]
eta = (0.5) * np.arcsin(
    Stokes_Q / np.sqrt(Stokes_D**2 + Stokes_P**2 + Stokes_Q**2)
)
# [Yoo et al, 2018] -- Eqn 6 & 7
# [Koushik et. al, 2019] -- typical values of inverse axial ratios [1.0, 3.5]; median: 1.4
axial_ratio = np.tan(eta)
inverse_axialratio = np.abs(1/ axial_ratio) # [rad/s]

# Eqn. 8 [Koushik et. al, 2019] -- intrinsic fequency -- frequency observed in the reference frame moving with the background wind
intrinsic_frequency = f_coriolis * inverse_axialratio # [rad/s]

intrinsic_period = 2*np.pi/intrinsic_frequency # [s]

## other checck
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/97JD03325
#limit
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014JD022448 figure 1b
#mean_buoyancy_frequency > intrinsic_frequency > intrinsic_frequency

if intrinsic_frequency > mean_buoyancy_frequency:
    print("Intrinsic frequency of wave packet greater than the Brunt-Vaisala frequency...not possible")

# vertical wavenumber and wavelength
vertical_wavenumber = (2 * np.pi) / u_periods[a_index_of_max_local_power] # [1/m]
vertical_wavelength = u_periods[a_index_of_max_local_power] # [m]


# horizontal wavenumber and wavelength [Murphy et al, 2014] -- Eqn B2
horizontal_wavenumber = (vertical_wavenumber/mean_buoyancy_frequency)*np.sqrt(intrinsic_frequency**2 - f_coriolis**2) # [1/m]
horizontal_wavelength = (2 * np.pi) / horizontal_wavenumber # [m]



## dispersion relation for low/medium-freqyuency gravity waves
omega_squared = mean_buoyancy_frequency**2 * (horizontal_wavenumber**2 / vertical_wavenumber**2) + f_coriolis**2


# [Murphy et al, 2014] -- Eqn 2
# averages done over vertical span of the wave
vertical_extent_of_zonal_perturbation = np.mean((iu_wave.real)**2)
vertical_extent_of_meridional_perturbation = np.mean((iv_wave.real)**2)
vertical_extent_of_temperature = np.mean((it_wave.real / choose_data_frame_analyze["T [°C]"].iloc[FWHM_variance])**2)

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

# momentum flux [Pa]
zonal_momentum_flux = -rho * (intrinsic_frequency*grav_constant/mean_buoyancy_frequency**2)*np.mean(iu_wave.real*it_wave.imag/ choose_data_frame_analyze["T [°C]"].iloc[FWHM_variance])

meridional_momentum_flux = -rho * (intrinsic_frequency*grav_constant/mean_buoyancy_frequency**2)*np.mean(iv_wave.real*it_wave.imag/ choose_data_frame_analyze["T [°C]"].iloc[FWHM_variance])


data_dictionary = {}

