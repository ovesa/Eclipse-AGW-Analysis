# Eclipse-AGW-Analysis

Analysis pipeline for detecting Atmospheric Gravity Waves (AGWs) using radiosonde data collected during solar eclipses.

## Overview

This code reads in a GRAWMET profile for a given radiosonde launch and preprocesses it for AGW analysis. Preprocessing includes removing bad values, linearly interpolating onto an evenly spaced spatial grid, and setting the desired spatial resolution.

## Methodology

The pipeline uses vertical profiles of zonal wind speed, meridional wind speed, and temperature from the radiosonde. First-order perturbations — believed to be directly caused by AGWs are extracted by subtracting a least-squares polynomial fit ([Moffat-Griffin et al., 2011](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015349)).

A Morlet Wavelet Transform ([Torrence & Compo, 1998](https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf)) is applied to each parameter to isolate wave packets in wavenumber–height space. The zonal and meridional wavelet coefficients are combined to compute a power surface.

The power surface is then scanned for local maxima following the procedure of [Zink & Vincent, 2001](https://digital.library.adelaide.edu.au/dspace/bitstream/2440/12560/1/hdl_12560.pdf). Candidate AGWs are identified as maxima above a significance threshold and within the cone of influence. For each candidate, a rectangular boundary is defined where the power surface falls to one-quarter of its peak value (or begins to rise again), and an Inverse Wavelet Transform reconstructs the perturbations within that region. The vertical extent of each potential AGW is determined from the full-width at half-maximum (FWHM) of the horizontal wind variance.

Finally, Stokes Parameters ([Eckermann, 1996](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96JD01578)) are computed to derive wave properties including vertical and horizontal wavelength, propagation direction, and phase and group speed.

## Output

Results are saved to a `.xlsx` file in a `Results/` folder. Figures are saved to a `Figures/` folder and include:

- Dominant vertical perturbations
- First-order perturbations
- Hodograph analysis
- Power surface
- Wind variance FWHM


## Publications

This code was used in the following peer-reviewed publications:

- Shetye et al. (2024). *Characterization of Atmospheric Gravity Waves Observed During a Total Solar Eclipse in Granbury, Texas.* Bulletin of the AAS. DOI:10.3847/25c2cfeb.af234821. [Link](https://baas.aas.org/pub/2024n9i038/release/1)
- Vesa et al. (2024). *Revealing the Dynamics of Atmospheric Gravity Waves: Insights from an Annular Solar Eclipse Event at Artesia Science Center, NM.* Bulletin of the AAS. DOI: 10.3847/25c2cfeb.5ceea9d5. [Link](https://baas.aas.org/pub/2024n9i043/release/2)
