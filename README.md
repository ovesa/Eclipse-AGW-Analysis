# Eclipse-AGW-Analysis

An end-to-end data pipeline for detecting and characterizing Atmospheric Gravity Waves (AGWs) using radiosonde data collected during solar eclipses. The pipeline covers the full data lifecycle: ingestion and cleaning of raw instrument data, signal processing, automated feature detection, wave parameter extraction, and structured result export.

---

## Pipeline Overview

```
Raw GRAWMET Profile (.xls)
        │
        ▼
 Data Ingestion & Cleaning        ← pandas, removes bad values
        │
        ▼
 Preprocessing & Interpolation    ← evenly spaced spatial grid, resolution setting
        │
        ▼
 Signal Extraction                ← least-squares polynomial fit subtraction
        │
        ▼
 Wavelet Transform                ← Morlet wavelet, wavenumber–height space
        │
        ▼
 Automated Feature Detection      ← power surface scanning, local maxima, significance thresholding
        │
        ▼
 Wave Reconstruction              ← Inverse Wavelet Transform, FWHM vertical extent
        │
        ▼
 Parameter Extraction             ← Stokes Parameters: wavelength, direction, phase/group speed
        │
        ▼
 Structured Output                ← .xlsx results + diagnostic figures
```

---

## Technical Stack

- **Data ingestion:** `pandas` to read raw GRAWMET radiosonde profiles (`.xls`), handle missing values, and structure data for downstream processing
- **Numerical processing:** `numpy`, `scipy` for interpolation, least-squares fitting, signal processing
- **Wavelet analysis:** Morlet Wavelet Transform and Inverse Wavelet Transform for time-frequency decomposition
- **Automated detection:** power surface scanning with configurable significance thresholds and cone-of-influence masking
- **Output:** results exported to `.xlsx` via `pandas`; diagnostic figures generated with `matplotlib`

---


## Methodology

### 1. Data Ingestion and Preprocessing
Raw GRAWMET radiosonde profiles are read in as `.xls` files using `pandas`. Preprocessing includes removing bad or unphysical values, linearly interpolating onto an evenly spaced spatial grid, and setting the desired vertical resolution. The pipeline uses vertical profiles of zonal wind speed, meridional wind speed, and temperature.

### 2. Perturbation Extraction
First-order perturbations, believed to be directly caused by AGWs, are extracted by subtracting a least-squares polynomial fit from each vertical profile ([Moffat-Griffin et al., 2011](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2010JD015349)).

### 3. Wavelet Transform
A Morlet Wavelet Transform ([Torrence & Compo, 1998](https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf)) is applied to each parameter to isolate wave packets in wavenumber–height space. The zonal and meridional wavelet coefficients are combined to compute a power surface.

### 4. Automated AGW Detection
The power surface is scanned for local maxima following the procedure of [Zink & Vincent, 2001](https://digital.library.adelaide.edu.au/dspace/bitstream/2440/12560/1/hdl_12560.pdf). Candidate AGWs are identified as maxima above a significance threshold and within the cone of influence. For each candidate, a rectangular boundary is defined where the power surface falls to one-quarter of its peak value (or begins to rise again), and an Inverse Wavelet Transform reconstructs the perturbations within that boundary. The vertical extent of each potential AGW is determined from the full-width at half-maximum (FWHM) of the horizontal wind variance.

### 5. Wave Parameter Extraction
Stokes Parameters ([Eckermann, 1996](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96JD01578)) are computed to derive wave properties including vertical and horizontal wavelength, propagation direction, and phase and group speed.

---

## Output

Results are saved to a `.xlsx` file in a `Results/` folder. Figures are saved to a `Figures/` folder and include:

- Dominant vertical perturbations
- First-order perturbations
- Hodograph analysis
- Power surface
- Wind variance FWHM

---

## Publications

This pipeline was used in the following peer-reviewed publications:

- Shetye et al. (2024). *Characterization of Atmospheric Gravity Waves Observed During a Total Solar Eclipse in Granbury, Texas.* Bulletin of the AAS. DOI:10.3847/25c2cfeb.af234821. [Link](https://baas.aas.org/pub/2024n9i038/release/1)
- Vesa et al. (2024). *Revealing the Dynamics of Atmospheric Gravity Waves: Insights from an Annular Solar Eclipse Event at Artesia Science Center, NM.* Bulletin of the AAS. DOI: 10.3847/25c2cfeb.5ceea9d5. [Link](https://baas.aas.org/pub/2024n9i043/release/2)
