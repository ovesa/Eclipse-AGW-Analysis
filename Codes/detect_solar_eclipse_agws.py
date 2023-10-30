################### Necessary Libraries ###################

import numpy as np
import matplotlib.pyplot as plt
import glob
from waveletFunctions import wave_signif
import copy

import pycwt as wavelet_method2
import cmasher as cm
from scipy.signal import hilbert

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

################### Zonal & Meridional Components of Wind Speed ###################

choose_data_frame_analyze = dat[0]

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

padding = 1
dj = 0.01
dt = spatial_resolution
s0 = 2 * dt
mother_wavelet = "MORLET"

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

peak_nom = 0
peak_containers = datafunctions.extract_boundaries_around_peak(power, peaks, peak_nom)

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

u_invert = copy.deepcopy(u_coef)
u_invert[~(peak_containers)] = 0

v_invert = copy.deepcopy(v_coef)
v_invert[~(peak_containers)] = 0

t_invert = copy.deepcopy(t_coef)
t_invert[~(peak_containers)] = 0


# Inverse wavelet transform
# Want to use the exact parameters used in the initial calculation of the wavelet coefficients
iu_wave = wavelet_method2.icwt(u_invert, u_scales, dt, dj, wavelet_method2.Morlet(6))
iv_wave = wavelet_method2.icwt(v_invert, v_scales, dt, dj, wavelet_method2.Morlet(6))
it_wave = wavelet_method2.icwt(t_invert, t_scales, dt, dj, wavelet_method2.Morlet(6))

################### Hodograph Analysis ###################

fig = plt.figure(figsize=[5, 4])

plt.plot(iu_wave.real, iv_wave.real, color="k", linewidth=1.5,zorder=0)

plt.scatter(
    iu_wave.real[0],
    iv_wave.real[0],
    color="r",
    marker="o",
    s=35,
    zorder=1, edgecolor='k',
)

plt.annotate(
    "%.1f km" % (choose_data_frame_analyze["Geopot [m]"].iloc[0] / 1000),
    (iu_wave.real[0], iv_wave.real[0]),
)

plt.scatter(
    iu_wave.real[-1],
    iv_wave.real[-1],
    color="gold",
    marker="o",
    s=35,
    zorder=1, edgecolor='k',
)

plt.annotate(
    "%.1f km" % (choose_data_frame_analyze["Geopot [m]"].iloc[-1] / 1000),
    (iu_wave.real[-1], iv_wave.real[-1]),
)

plt.xlabel("Zonal Wind Speed [m/s]")
plt.ylabel("Meridional Wind Speed [m/s]")

plt.tight_layout()
plt.show()


################### Extracting Wave Parameters ###################

# Constants
grav_constant = 9.81  # gravity [m/s^2]
ps = 1000  # standard pressure [hPa] -- equal to 1 millibar
kappa = 2 / 7 # Poisson constant for dry air 
celsius_to_kelvin_conversion = 273.15 # 0 Celsium == 273.15 K

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
    grav_constant * potential_temperature * np.gradient(potential_temperature, spatial_resolution)
)
# need to average over the vertical extent of the wave

mean_buoyancy_period = (2 * np.pi) / mean_buoyancy_frequency


## Stokes parameters for gravity waves
# Eqn. 9 from [Pfenninger et. al, 1999]
# Hilbert transform of the meridional wind component - adds a 90 deg phase shift to vertical profile
hilbert_v = np.imag(hilbert(iv_wave.real))

# Represent the total energy
Stokes_I = np.mean(iu_wave.real**2) + np.mean(iv_wave.real**2)
# variance difference/ linear polarization measure
Stokes_D = np.mean(iu_wave.real**2) - np.mean(iv_wave.real**2)
# P and Q are the in phase and quadrature components between wave components
Stokes_P = 2 * np.mean(iu_wave.real * iv_wave.real)
# measure of the circular polarization
# Determines the rotation of the polarized ellipse
Stokes_Q = 2 * np.mean(iu_wave.real * hilbert_v)

# [Pfenninger et. al, 1999] -- think of Stokes Q as the energy difference betwen propagating AGWs
if Stokes_Q >= 0:
    print("Wave energy propagating upward")
    print("Clockwise rotation of the polarized ellipse")
else:
    print("Wave energy propagating downwards")
    print("Anticlockwise rotation of the polarized ellipse")


# stastical measure of the coherence of the wave field
polarization_factor = np.sqrt(Stokes_P**2 + Stokes_D**2 + Stokes_Q**2) / Stokes_I


# [Yoo et al, 2018: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JD029164]
if polarization_factor == 1:
    print("Monochromatic Wave")
elif polarization_factor == 0:
    print("Not an AGW; no polarization relationship")

# [Yoo et al, 2018: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JD029164]
if 0.5 <= polarization_factor < 1:
    print("Most likely an AGW")
else:
    print("Might not be an AGW. Polarization factor too low")

# dynamic shear instability -- Richardson Number
# richardson_number = mean_buoyancy_frequency**2 / ((du / dz) ** 2 + (dv / dz) ** 2)

# [Pfenninger et. al, 1999] -
# if N^2 and R < 0
if mean_buoyancy_frequency**2 < 0:
    print("Convectively unstable")
# dynamic instability --- 0 < R < 0.25

# Eqn. 11 from [Pfenninger et. al, 1999]
# Orientation of major axis of ellipse
# direction of the horizontal component of the wave vector
# oriented parallel to GW vector
# theta is measured anticlockwise from the x-axis
# 180 deg ambiguity
# horizontal direction of propagation of GW (deg clockwise from N)
theta = np.arctan2(Stokes_P, Stokes_D) / 2


# [Koushik et. al, 2019]  -- stokes p and q less than threshodl value are not agws
if np.abs(Stokes_P) and np.abs(Stokes_Q) < 0.5:
    print("Might not be an AGW")


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
hilbert_t = np.imag(hilbert(it_wave.real))

# sign determines direction of propagaton
sign = U_prime * hilbert_t
len(np.unique(np.sign(sign))) == 1
theta_deg = np.rad2deg(theta)

# positive, or negative sign == theta +/- 180 deg


# coherencey, correlated power between U and V
C = np.sqrt((Stokes_P**2 + Stokes_Q**2) / (Stokes_I**2 - Stokes_D**2))

# https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
rotation_matrix = np.array(
    ((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta)))
)
wind_matrix = np.array([iu_wave, iv_wave])
wind_matrix = np.dot(rotation_matrix, wind_matrix)
axial_ration2 = np.linalg.norm(wind_matrix[0]) / np.linalg.norm(wind_matrix[1])


# axial ratio -- ratio of the minor axis to the maxjor axis; aspect ratio of the polarization ellipse
# Eqn. 8 [Koushik et. al, 2019]
eta = (0.5) * np.arcsin(
    Stokes_Q / np.sqrt(Stokes_D**2 + Stokes_P**2 + Stokes_Q**2)
)
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2018JD029164
axial_ratio = 1 / np.tan(eta)

# shear in wind compnonet transcverse to propaghation direcction will change axial ratio
# corrrected_axial_ration = np.abs()

# Eckermann, S. D., & Vincent, R. A. (1989). Falling sphere observations of anisotropic gravity wave motions in the upper stratosphere over Australia. Pure and Applied Geophysics PAGEOPH, 130(2-3), 509–532. doi:10.1007/bf00874472


vertical_wavenumber = (2 * np.pi) / u_periods


# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2014JD022448
# Eqn. 8 [Koushik et. al, 2019] -- intrinsic fequency -- frequency observed in the reference frame moving with the background wind
# Typical values are from 1.0 to 3.5
intrinsic_frequency = f_coriolis * axial_ratio

## dispersion relation for low/medium-freqyuency gravity waves
omega_squared = N**2 * (kh**2 / m**2) + f_coriolis**2


# Eqn. 814 [Koushik et. al, 2019]
#  gravity wave kinetic energy [J/Kg]
kinetic_energy = (0.5) * (np.mean(iu_wave.real**2) + np.mean(iv_wave.real**2))

##########################################

windVariance = np.abs(iu_wave) ** 2 + np.abs(iv_wave) ** 2

uR = iu_wave.copy()[windVariance >= 0.5 * np.max(windVariance)]
vR = iv_wave.copy()[windVariance >= 0.5 * np.max(windVariance)]

uR = uR.real
vR = vR.real


plt.figure()
plt.plot(uR, vR)
plt.show()


plt.figure()
plt.plot(iu_wave.real, iv_wave.real)
plt.show()


# ##############
# vHilbert = iv_wave.copy().imag
# ucg = iu_wave.real
# vcg = iv_wave.real
# # Stokes parameters from Murphy (2014) appendix A and Eckerman (1996) equations 1-5
# I = np.mean(ucg**2) + np.mean(vcg**2)
# D = np.mean(ucg**2) - np.mean(vcg**2)
# P = np.mean(2 * ucg * vcg)
# Q = np.mean(2 * ucg * vHilbert)
# degPolar = np.sqrt(D**2 + P**2 + Q**2) / I


# theta = 0.5 * np.arctan2(
#     P, D
# )  # arctan2 has a range of [-pi, pi], as opposed to arctan's range of [-pi/2, pi/2]

# uvMatrix = [
#     ucg.copy(),
#     vcg.copy(),
# ]  # Combine into a matrix for easy rotation along propagation direction

# # Rotate by -theta so that u and v components of 'uvMatrix' are parallel/perpendicular to propogation direction
# rotate = [
#     [np.cos(theta), np.sin(theta)],
#     [-np.sin(theta), np.cos(theta)],
# ]  # Inverse of the rotation matrix
# uvMatrix = np.dot(rotate, uvMatrix)

# # From Murphy (2014) table 1, and Zink & Vincent (2001) equation A10
# axialRatio = np.linalg.norm(uvMatrix[0]) / np.linalg.norm(uvMatrix[1])


# # From Murphy (2014), citing Zink (2000) equation 3.17
# # This is the coherence function, measuring the coherence of U|| and T
# gamma = np.mean(uvMatrix[0] * np.conj(it_wave.real)) / np.sqrt(
#     np.mean(np.abs(uvMatrix[0]) ** 2) * np.mean(np.abs(it_wave.real) ** 2)
# )


# # Vertical wavenumber [1/m]
# m = 2 * np.pi / u_periods
