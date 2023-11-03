# Eclipse-AGW-Analysis

Analysis for detecting Atmospheric Gravity Waves (AGWs) using radiosonde data taken during a solar eclipse. 

This code reads in the GRAWMET profile for a particular launch and preprocesses it. Preprocessing steps include cleaning the data to remove bad values, linearly interpolating the data onto an evenly spaced spatial grid, and setting the desired spatial resolution.

This script uses mainly the zonal wind speed, meridional wind speed, and temperature vertical profiles acquired from the radiosonde attached to the weather balloon. From these vertical profiles, the first-order perturbations, which are believed to be perturbations directly caused by AGWs, are calculated by subtracting the least squares polynomial fit [Moffat-Griffin et. al, 2011].

A Wavelet Transform with the Morlet function [Torrence and Campo, 1989] that isolates the wave packets in wavenumber vs height space is performed for each of the mentioned parameters. From the zonal and meridional wavelet transformed coefficients, we can compute the power surface.

The next steps follow the procedure listed in [Zink and Vincent, 2001]. The power surface is scanned for local maxima above a certain threshold and within the cone of influence and wave significance determined by the Wavelet Transform. These local maxima correspond to potential AGWs. For each local maxima, we determine a rectangular boundary around the value where the power surface falls to 1/4 its value or rises again. Using only the values in the wavelet-transformed perturbation coefficients that fall within that rectangular boundary, we reconstruct the perturbations with an Inverse Wavelet Transform. The vertical extent of the potential AGW is determined by the full width half max (FWHM) of the horizontal wind variance.

From here, we can compute the Stokes Parameters to derive wave parameters such as the vertical wavelength, horizontal wavelength, propagation direction, phase and group speed, etc.

