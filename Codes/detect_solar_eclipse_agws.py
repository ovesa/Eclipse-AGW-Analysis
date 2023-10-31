################### Necessary Libraries ###################

import numpy as np
import matplotlib.pyplot as plt
import glob
from waveletFunctions import wave_signif
import copy

import pycwt as wavelet_method2
import cmasher as cm

plt.ion()
  
import matplotlib

import plottingfunctions
import datafunctions

################### Plotting Properties ###################

tex_fonts = {
    "text.usetex": False,
    "font.family": "serif",
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
# Select all xls files that match
fname = glob.glob(path + "*end.xls")

file_nom = -1

dat = datafunctions.read_grawmet_profile(fname[file_nom])

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
print("Information about datasheet:")
print("Time range: [%s, %s] UTC" % (starting_time_for_flight,ending_time_for_flight))
print("Altitude range: [%s, %s] m" % (original_min_altitude,original_max_altitude))
print("\n")

################### Preprocess and Interpolate Data ###################

# tropopause height [m]
tropopause_height = 12000 

# to give a rise rate of 5 m/s
spatial_resolution = 5 

# spatial height interpolation limit [m]
# Tells code to not interpolate if there are more than [interpolation_limit] consecutive rows of missing/ NaN values
# Interpolation starts anew after hitting the interpolation limit
interpolation_limit = 1000

dat = datafunctions.clean_data(dat, tropopause_height, original_data_shape)

dat = datafunctions.interpolate_data(dat, interpolation_limit)

data_sections = datafunctions.check_data_for_interpolation_limit_set(
    dat, interpolation_limit
)

dat = datafunctions.set_spatial_resolution_for_data(data_sections, spatial_resolution)

choose_data_frame_analyze = dat[0]

choose_data_frame_analyze = datafunctions.convert_seconds_to_timestamp(choose_data_frame_analyze, starting_time_for_flight)

################### Zonal & Meridional Components of Wind Speed ###################

(
    u_zonal_speed,
    v_meridional_speed,
    temperature,
) = datafunctions.extract_wind_components_and_temperature(choose_data_frame_analyze)


################### Calculate First-Order Perturbations ###################

v_meridional_fit = datafunctions.compute_second_order_polynomial_fits(
    choose_data_frame_analyze, v_meridional_speed, 2
)

u_zonal_fit = datafunctions.compute_second_order_polynomial_fits(
    choose_data_frame_analyze, u_zonal_speed, 2
)

temperature_fit = datafunctions.compute_second_order_polynomial_fits(
    choose_data_frame_analyze, temperature, 2
)

u_zonal_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, u_zonal_speed, u_zonal_fit
)

v_meridional_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, v_meridional_speed, v_meridional_fit
)

temperature_perturbations = datafunctions.derive_first_order_perturbations(
    choose_data_frame_analyze, temperature, temperature_fit
)


plottingfunctions.plot_vertical_profiles_with_residual_perturbations(
    choose_data_frame_analyze,
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
    "/media/oana/Data1/Annular_Eclipse_Analysis/Figures/First-Order-Perturbations/",
    save_fig=False,
)

################### Wavelet Analysis ###################

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

# Calculate the significance of the wavelet coefficients
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
        np.array(u_periods + v_periods) <= (u_coi[i] + v_coi[i])
        for i in range(len(choose_data_frame_analyze["Geopot [m]"]))
    ]
).T

################### Find Local Maxima & Extract Boundaries Around Gravity Wave Packet ###################

peaks = datafunctions.find_local_maxima(power, 0.011, coiMask, signif)

peak_nom = 8
peak_containers, boundary_rows, boundary_cols = datafunctions.extract_boundaries_around_peak(power, peaks, peak_nom)

associated_timestamps_range_of_boundary = choose_data_frame_analyze["Time [UTC]"].iloc[boundary_cols] # TimeStamps [UTC]
associated_height_range_of_boundary =  choose_data_frame_analyze["Geopot [m]"].iloc[boundary_cols] # m

associated_height_of_peak = choose_data_frame_analyze["Geopot [m]"].iloc[peaks[peak_nom][1]] # m
associated_time_of_peak = choose_data_frame_analyze["Time [UTC]"].iloc[peaks[peak_nom][1]] # TimeStamp [UTC]

################### Plot Power Surface ###################

colormap = cm.chroma

plottingfunctions.plot_power_surface(
    choose_data_frame_analyze,
    power,
    u_periods,
    peak_containers,
    signif,
    coiMask,
    peaks,
    colormap,
    starting_time_for_flight,
    "/media/oana/Data1/Annular_Eclipse_Analysis/Figures/Power_Surfaces",
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
horizonatal_windvariance = np.abs(u_inverted_coeff) ** 2 + np.abs(v_inverted_coeff) ** 2
max_horizonatal_windvariance = np.max(horizonatal_windvariance)
FWHM_variance = np.where(horizonatal_windvariance >= 0.5*max_horizonatal_windvariance)[0]

vertical_extent_coordx, vertical_extent_coordy = FWHM_variance[0],FWHM_variance[-1]

iu_wave = (u_inverted_coeff)[FWHM_variance]
iv_wave = (v_inverted_coeff)[FWHM_variance]
it_wave = (t_inverted_coeff)[FWHM_variance]

################### Hodograph Analysis ###################

plottingfunctions.plot_hodograph(iu_wave.real, iv_wave.real,choose_data_frame_analyze)
# plottingfunctions.winds_associated_with_dominant_vertical_wavelengths(iu_wave, iv_wave,choose_data_frame_analyze)

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
) # [Hz]

# average over boundary which wave was detected
# Upper frequency for the gravity waves
mean_buoyancy_frequency = mean_buoyancy_frequency[vertical_extent_coordx:vertical_extent_coordy].mean(skipna=True) # [Hz]

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
    print("Wave energy propagating upward")
    print("Clockwise rotation of the polarized ellipse")
    print("Stokes Q is positive")
else:
    print("Wave energy propagating downwards")
    print("Anticlockwise rotation of the polarized ellipse")
    print("Stokes Q is negative")



# stastical measure of the coherence of the wave field
# degree of polarization
polarization_factor = np.sqrt(Stokes_P**2 + Stokes_D**2 + Stokes_Q**2) / Stokes_I


# [Yoo et al, 2018]
if 0.5 <= polarization_factor < 1:
    print("d=%.2f -- Most likely an AGW"%polarization_factor)
elif polarization_factor == 1:
    print("d=%.2f -- Monochromatic Wave"%polarization_factor)
elif polarization_factor == 0:
    print("d=%.2f -- Not an AGW; no polarization relationship"%polarization_factor)
else:
    print("d=%.2f -- Might not be an AGW. Polarization factor too low or unrealistic value"%polarization_factor)

# [Koushik et. al, 2019]  -- stokes p and q less than threshold value might not be not agws
if np.abs(Stokes_P) < 0.05 or np.abs(Stokes_Q) < 0.05:
    print("Might not be an AGW; representative of poor wave activity")



# dynamic shear instability -- Richardson Number
richardson_number = mean_buoyancy_frequency**2 /( np.gradient(u_zonal_perturbations[vertical_extent_coordx:vertical_extent_coordy],choose_data_frame_analyze["Geopot [m]"][vertical_extent_coordx:vertical_extent_coordy])**2 + np.gradient(v_meridional_perturbations[vertical_extent_coordx:vertical_extent_coordy],choose_data_frame_analyze["Geopot [m]"][vertical_extent_coordx:vertical_extent_coordy])**2   )
richardson_number = np.mean(richardson_number)
# [Pfenninger et. al, 1999] -
# if N^2 and R < 0
if mean_buoyancy_frequency**2 < 0 and richardson_number < 0:
    print("Convectively unstable")
elif 0 < richardson_number < 0.25:
    print("Dynamically unstable")

# Eqn. 11 from [Pfenninger et. al, 1999]
# Orientation of major axis of ellipse
# direction of the horizontal component of the wave vector
# oriented parallel to GW vector
# theta is measured anticlockwise from the x-axis
# 180 deg ambiguity
# horizontal direction of propagation of GW (deg clockwise from N)
theta = np.arctan2(Stokes_P, Stokes_D) / 2


# Coriolis Force
# Omega is rotation rate of earth
# [https://www.teos-10.org/pubs/gsw/pdf/f.pdf]
# [Pfenninger et. al, 1999] -- Coriolis forced causes wind vectors to rotate with height & form an ellipse
latitude_artesia = 32.842258  # degrees
omega_Earth = 7.29e-5  # [radians /seconds]
## [Fritts and Alexander: MIDDLE ATMOSPHERE GRAVITY WAVE DYNAMIC]
f_coriolis = 2 * omega_Earth * np.sin(np.deg2rad(latitude_artesia))


# remove ambiguity
# wind found in the major axis direction
# [Pfenninger et. al, 1999]
U_prime = iu_wave.real * np.cos(theta) + iv_wave.real * np.sin(theta)

# Hilbert transform of the temperature perturbation
# [Zink and Vincent, 2001] -- imaginary part is 90 degree phase shifted version (Hilbert transformed versions)

hilbert_t = it_wave.imag

# sign determines direction of propagaton
sign = U_prime * hilbert_t
len(np.unique(np.sign(sign))) == 1
theta_deg = np.rad2deg(theta)

# positive, or negative sign == theta +/- 180 deg


# coherencey, correlated power between U and V
C = np.sqrt((Stokes_P**2 + Stokes_Q**2) / (Stokes_I**2 - Stokes_D**2))

# # https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
# rotation_matrix = np.array(
#     ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
# )
# wind_matrix = np.array([iu_wave.real, iv_wave.real])
# wind_matrix = np.dot(rotation_matrix, wind_matrix)
# axial_ration2 = np.linalg.norm(wind_matrix[0]) / np.linalg.norm(wind_matrix[1])


# axial ratio -- ratio of the minor axis to the maxjor axis; aspect ratio of the polarization ellipse
# Eqn. 8 [Koushik et. al, 2019]
eta = (0.5) * np.arcsin(
    Stokes_Q / np.sqrt(Stokes_D**2 + Stokes_P**2 + Stokes_Q**2)
)
# [Yoo et al, 2018] -- Eqn 6 & 7
# [Koushik et. al, 2019] -- typical values of axial ratios [1.0, 3.5]; median: 1.4
axial_ratio = np.tan(eta)
inverse_axialratio = 1/ axial_ratio



# shear in wind compnonet transcverse to propaghation direcction will change axial ratio
# corrrected_axial_ration = np.abs()

# Eckermann, S. D., & Vincent, R. A. (1989). Falling sphere observations of anisotropic gravity wave motions in the upper stratosphere over Australia. Pure and Applied Geophysics PAGEOPH, 130(2-3), 509–532. doi:10.1007/bf00874472

# vertical wavenumber m
vertical_wavenumber = (2 * np.pi) / u_periods
vertical_wavelength = u_periods

# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014JD022448
# Eqn. 8 [Koushik et. al, 2019] -- intrinsic fequency -- frequency observed in the reference frame moving with the background wind
# Typical values are from 1.0 to 3.5
intrinsic_frequency = f_coriolis * axial_ratio


# [Murphy et al, 2014] -- Eqn 82
horizontal_wavenumber = (vertical_wavenumber/mean_buoyancy_frequency)*np.sqrt(intrinsic_frequency**2 - f_coriolis**2)
horizontal_wavelength = (2 * np.pi) / horizontal_wavenumber



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



# intrinsic group velocities 

# intrinsic vertical group velocity
# [Murphy et al, 2014] -- Eqn B5
cgz = -1 * ((intrinsic_frequency**2 - f_coriolis**2)/(intrinsic_frequency*vertical_wavenumber))

# zonal wavenumber -- [Murphy et al, 2014] -- Table 2
k = horizontal_wavenumber*np.sin(theta)

# meridional wavenumber -- [Murphy et al, 2014] -- Table 2
l = horizontal_wavenumber*np.cos(theta)

# intrinsice vertical phase speed -- [Murphy et al, 2014] -- Table 2
cz = intrinsic_frequency/vertical_wavenumber

# intrinsic horizontal phase speed -- [Murphy et al, 2014] -- Table 2
c_hor = intrinsic_frequency/horizontal_wavenumber

# intrinsic zonal phase speed -- [Murphy et al, 2014] -- Table 2
cx = intrinsic_frequency/k

# intrinsice meridional phase speed -- [Murphy et al, 2014] -- Table 2
cy = intrinsic_frequency/l

# intrinsic zonal group velocity -- [Murphy et al, 2014] -- Table 2
cgx = k*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# intrinsic meridional group velocity -- [Murphy et al, 2014] -- Table 2
cgy = l*mean_buoyancy_frequency**2/(intrinsic_frequency*vertical_wavenumber**2)

# intrinsic horizontal group velocity -- [Murphy et al, 2014] -- Table 2
cgh = np.sqrt(cgx**2 + cgy**2)

