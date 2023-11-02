################### Necessary Libraries ###################

import pandas as pd
import numpy as np
import tqdm as tqdm
import copy
from scipy.signal import argrelmin
from waveletFunctions import wavelet
from skimage.feature import peak_local_max

###########################################################


def read_grawmet_profile(file):
    """
    Read in the GRAWMET profile associated with the input string. Entire code currently assumes the
    file is in .xls format.

    Arguments:
        file -- Path to file and/or just filename [string].

    Returns:
        The entire GRAWMET profile in a Pandas DataFrame.
    """
    try:
        data = pd.read_excel(file)

        print("\n")
        print("Reading in %s" % (file))
        print("\n")
        print("Shape of dataframe: ", data.shape)
        print("\n")
        print("Columns Read In : ", data.columns)
        print("\n")

    except FileNotFoundError:
        print("\n")
        print("File does not exist. Double check path and/or filename")
        print("\n")

    return data


def grab_initial_grawmet_profile_parameters(dataframe):
    """
    Extracts several key GRAWMET profile parameters from the data.

    Arguments:
        dataframe -- The Pandas DataFrame from which to extract the parameters below.

    Returns:
        The time grid [UTC], the height grid [meters], the starting and ending times for the radiosonde
        launch [UTC], the starting and ending heights for the radiosonde launch [m].
    """

    # Check for if this particular column exists
    if "Time (UTC)" not in dataframe.columns:
        print("Time grid not found. Check column headings")
        UTC_time_grid, starting_time_for_flight, ending_time_for_flight = 0, 0, 0

    else:
        UTC_time_grid = dataframe["Time (UTC)"]
        starting_time_for_flight = UTC_time_grid.iloc[0]
        ending_time_for_flight = UTC_time_grid.iloc[-1]

    data_shape = dataframe.shape

    altitude_grid = dataframe["Geopot [m]"]
    min_altitude = altitude_grid.iloc[0]
    max_altitude = altitude_grid.iloc[-1]

    return (
        UTC_time_grid,
        starting_time_for_flight,
        ending_time_for_flight,
        data_shape,
        altitude_grid,
        min_altitude,
        max_altitude,
    )


def clean_data(dataframe, tropopause_height, original_data_shape):
    """
    Clean and preprocess the data by removing rows with bad values, rows where the height is not incrementally
    increasing (radiosonde isn't ascending anymore), and rows with values that correspond below the tropopause.

    Arguments:
        dataframe -- The Pandas DataFrame to perform cleaning and preproccessing procedures.
        tropopause_height -- The height at which the tropopause begins [m].
        original_data_shape -- The shape of the DataFrame prior to cleaning and preprocessing the data.

    Returns:
        The data [Pandas DataFrame] after it has been preprocessed and cleaned to fulfill the necessary conditions.
    """

    # We need to check if the values in the height column are strictly increasing
    # We ignore the first or zeroth row
    condition1 = (dataframe["Geopot [m]"].diff().gt(0)) | (dataframe.index == 0)
    # If rows do not match this condition, then we remove/drop them
    # Reset the DataFrame index to ensure counting starts back at 0
    dataframe = dataframe.drop(dataframe[condition1 == False].index).reset_index(
        drop=True
    )

    # We need to check if any rows contain bad values (NaNs)
    # GRAWMET sometimes prints out the value 999999.0, which I assume to be NaN equivalent
    # OR equal to 0 anywhere (meaning that the data didn't get recorded properly)
    condition2 = dataframe.eq(999999.0).any(1) | dataframe.eq(0).any(1)
    # We will remove rows that match this condition
    # Reset the DataFrame index to ensure counting starts back at 0
    dataframe = dataframe.drop(dataframe[condition2 == True].index).reset_index(
        drop=True
    )
    

    # We need to remove data below the tropopause
    # Steep changes in temperature below the tropopause will affect identifying atmospheric gravity waves
    tropopause_height = tropopause_height  # height [m]
    condition3 = dataframe["Geopot [m]"] >= tropopause_height

    # Keep rows matching this condition
    # Reset the DataFrame index to ensure counting starts back at 0
    dataframe = dataframe[condition3].reset_index(drop=True)

    # Check for if this particular column exists
    if "Time (UTC)" not in dataframe.columns:
        print("Time grid not found. Check column headings")
    else:
        # Drop the UTC Time column for now as it becomes difficult to deal with during the interpolation stage
        dataframe = dataframe.drop(columns=["Time (UTC)"])

    # Basic checks
    if dataframe.empty:
        print("Empty DataFrame; Double Check Data")
    else:
        print("Removed %s rows" % (original_data_shape[0] - dataframe.shape[0]))
        print("Shape of dataframe after cleaning up rows: ", dataframe.shape)

    return dataframe


def interpolate_data(dataframe, interpolation_limit):
    """
    Interpolate the data onto an evenly spaced spatial grid. This is a necessary step for the wavelet transform.

    Arguments:
        dataframe -- The Pandas DataFrame after it has been cleaned and preprocessed.
        interpolation_limit -- The interpolation limit for the maximum number of NaNs allowed to be interpolated
                                over. If there are rows greater than this limit, they will remain as NaNs. As we
                                are interpolating over the spatial/height grid, this value is represented as
                                a height value [m].

    Returns:
        An interpolated DataFrame on an evenly spaced spatial grid.
    """

    # We need to create a uniform spatial (height) grid [Colligan et. al, 2020]
    # Because the the reported height intervals from the radiosondes ascending are irregular
    height_grid = pd.DataFrame(
        {
            "Geopot [m]": np.arange(
                min(dataframe["Geopot [m]"]), max(dataframe["Geopot [m]"]) + 1
            )
        }
    )

    # We will now merge the new DataFrame containing the evenly spaced spatial grid to the main DataFrame
    # This code ensures that all the rows and values from the main DataFrame are maintained
    # We will merge on the height column to accomplish this
    # Reset the DataFrame index to ensure counting starts at 0
    dataframe = (
        pd.merge(dataframe, height_grid, how="outer", on="Geopot [m]")
        .sort_values(by=["Geopot [m]"])
        .reset_index(drop=True)
    )

    print("New shape of dataframe: ", dataframe.shape)

    # Now, we need to interpolate the data as to fill in the missing values in all the columns
    # If missing data in the height column hits the interpolation_limit variable, we don't interpolate and
    # leave those values as NaNs
    # Regular Pandas interpolation misses some values
    # [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html]
    # [https://stackoverflow.com/questions/30533021/interpolate-or-extrapolate-only-small-gaps-in-pandas-dataframe]
    # Solution: interpolate the data and then apply a mask to reset the interpolation limit gap to NaN values

    # Create a mask of the orginal DataFrame
    interpolation_mask = copy.deepcopy(dataframe)

    # New DataFrame where every value ascends by 1
    intermediate_mask = (
        interpolation_mask.notnull() != interpolation_mask.shift().notnull()
    ).cumsum()
    # New column that is filled with the value "1"
    intermediate_mask["ones"] = 1

    # Interpolation Mask is now True/False if condition is met
    for col in dataframe.columns:
        interpolation_mask[col] = (
            intermediate_mask.groupby(col)["ones"].transform("count")
            < interpolation_limit
        ) | dataframe[col].notnull()

    # Linearly interpolate the data in the backward direction [bfill()]
    # Bfill() fills in NaN values using the next valid value
    # Apply the interpolation mask to reset NaN values that match the interpolation limit gap

    dataframe = dataframe.interpolate().bfill()[interpolation_mask]

    return dataframe


def check_data_for_interpolation_limit_set(dataframe, interpolation_limit):
    """
    Checks interpolated DataFrame for where the interpolation limit was met and split the data before and after this
    group of NaNs.

    Arguments:
        dataframe -- The interpolated DataFrame.
        interpolation_limit -- The interpolation limit for the maximum number of NaNs allowed to be interpolated
                                over. If there are rows greater than this limit, they will remain as NaNs. As we
                                are interpolating over the spatial/height grid, this value is represented as
                                a height value [m].

    Returns:
        Returns a list of the split DataFrame(s) before and after the block of NaNs.
    """

    # We check all the rows in the DataFrame for groups of NaN values
    nan_indices = dataframe.index[dataframe.isnull().any(1)].to_list()
    # Increment list with list-comprehension
    # The idea here is to double your list of NaN indices to successfully split the sections surrounding the group of
    # of NaNs
    nan_indices = nan_indices + [val + 1 for val in nan_indices]

    # Split DataFrame
    # Split DataFrames are now in a list
    dataframe = [
        data
        for data in np.split(dataframe, nan_indices)
        if data.shape[0] > interpolation_limit and ~data.isnull().values.any()
    ]

    # Check if the DataFrame has been split
    if len(dataframe) == 1:
        print("There is 1 section of data")
    elif len(dataframe) == 0:
        print("No data to analyze. Double check dataframe")
    else:
        print("There are %s sections of data" % (len(dataframe)))

    return dataframe


def set_spatial_resolution_for_data(dataframe, spatial_resolution=5):
    """
    Set the spatial resolution over which to analyze atmospheric gravity waves.

    Arguments:
        dataframe -- The interpolated Pandas DataFrame for which to set the spatial resolution
        spatial_resolution -- The desired spatial resolution [m]. Resolution should be based on radiosonde data
                              [Colligan et. al, 2020] uses a spatial resolution that gives an average rise rate
                              of 5 m/s [m/s].

    Returns:
        The DataFrame set to the desired spatial resolution.
    """

    # Perform for each of the split DataFrames
    dataframe = [data.reset_index(drop=True) for data in dataframe]
    dataframe = [
        data.iloc[
            np.arange(0, len(data["Geopot [m]"]), spatial_resolution), :
        ].reset_index(drop=True)
        for data in dataframe
    ]

    return dataframe


def compute_zonal_and_meridional_speeds(
    dataframe,
    dataframe_wind_speed_column,
    dataframe_wind_direction_column,
    dataframe_temperature_column,
):
    """
    Calculate the zonal and meridional wind speeds from the total wind speed. Grab the temperature array, as well.
    Function not currently in use as GRAWMET already calculated these values.
    [http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv].

    Arguments:
        dataframe -- The Pandas DataFrame
        dataframe_wind_speed_column -- Column name corresponding to the total wind speed column [string].
        dataframe_wind_direction_column -- Column name corresponding to the wind direction column [string].
        dataframe_temperature_column -- Column name corresponding to the temperature column [string].

    Returns:
        The temperature array [C] and the zonal and meridional wind speeds [m/s] derived from the total wind
        speed [m/s] and wind direction [deg] columns.
    """

    # Zonal wind speed: East-West wind component
    # Meridional wind speed: North-South wind component
    u_zonal_speed = -dataframe[dataframe_wind_speed_column] * np.sin(
        dataframe[dataframe_wind_direction_column] * np.pi / 180
    )
    v_meridional_speed = -dataframe[dataframe_wind_speed_column] * np.cos(
        dataframe[dataframe_wind_direction_column] * np.pi / 180
    )
    temp_arr = dataframe[dataframe_temperature_column]

    return u_zonal_speed, v_meridional_speed, temp_arr


def extract_wind_components_and_temperature(dataframe):
    """
    Extracts the temperature array, and the zonal and meridional wind speeds from the DataFrame.
    [http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv].

    Arguments:
        dataframe -- The Pandas DataFrame.

    Returns:
         The temperature array [C], and the zonal and meridional wind speeds [m/s].
    """

    # Zonal wind speed: East-West wind component [m/s]
    # Meridional wind speed: North-South wind component [m/s]
    # Total vertical wind speed = sqrt(meridional_wind^2 + zonal_wind^2) [m/s]
    u_zonal_speed = dataframe["Ws U [m/s]"]
    v_meridional_speed = dataframe["Ws V [m/s]"]
    temperature = dataframe["T [°C]"]
    return u_zonal_speed, v_meridional_speed, temperature


def compute_polynomial_fits(dataframe, array, order=2):
    """
    Calculate a least squares polynomial fit to the necessary variable. This step is necessary to
    remove the background state from the main vertical profile of the variable.

    Arguments:
        dataframe -- The Pandas height series.
        array -- The array to perform the least squares polynomial fit.

    Keyword Arguments:
        order -- The degree of the fitting polynomial (default: {2}).

    Returns:
        A vector of the polynomial coefficients that minimizes the squared error.
    """
    idx = np.isfinite(dataframe) & np.isfinite(array)
    array_fit = np.polyfit(dataframe[idx], array[idx], order)

    return array_fit

def derive_first_order_perturbations(dataframe, perturbations, polynomial_fit):
    """
    Derive the first-order perturbations, which are thought to caused by atmospheric gravity waves
    [Moffat-Griffin et. al, 2011]. After computing the least squares polynomial fit for the
    perturbation (specifically the zonal and meridional wind perturbations), subtract it from the
    original vertical profiles. This removes the background pertubations [Moffat-Griffin et. al, 2011]
    & [Vincent and Alexander, 2000].

    Arguments:
        dataframe -- The Pandas height series.
        perturbations -- The perturbation array to compute the polynomial fit on.
        polynomial_fit -- A vector of the polynomial coefficients that minimizes the squared error.

    Returns:
        The first-order perturbations that are assumed to be due to atmospheric gravity waves.
    """

    # Apply polynomia fit then subtract from main vertical profile to remove background states
    # Don't need to create a new array because we are using the recently created evenly spaced height grid
    first_order_perturbations = perturbations - (
        polynomial_fit[0] * dataframe ** 2
        + polynomial_fit[1] * dataframe
        + polynomial_fit[2] * np.ones(len(perturbations))
    )

    return first_order_perturbations


def acorr(array, lags):
    """
    Performs the autocorrelation on a signal. The autocorrelation is normalized on the [-1,1] interval
    [https://stackoverflow.com/questions/71493065/what-is-the-difference-between-the-autocorrelation-functions-provided-by-statsmo].

    Arguments:
        array -- The wavelet coefficient array to calculate the statistical autocorrelation.
        lags -- The length of the wavelet coefficient array.

    Returns:
        The autocorrelation function normalized on the [-1,1] interval. The first value is lag 1.
    """

    # x_demeaned = array - array.mean()
    corr = np.correlate(array, array, "full")[len(array) - 1 :] / (
        np.var(array) * len(array)
    )

    return corr[: len(lags)]


def get_boundaries(
    signal,
    power_array_shape_for_signal,
    coord,
    threshold,
):
    """
    Obtain the boundaries along the local maxima where the value of the power surface
    falls to one-fourth its value or rises again.

    Arguments:
        signal -- The row or column where the local maxima is.
        power_array_shape_for_signal -- The shape of the power surface array.
        coord -- The coordinate associated with the local maxima.
        threshold -- The power surface threshold.

    Returns:
        The boundaries around each row and column where the power surface value falls to one-fourth
        its value or rises again.
    """
    # For each row and column where the local maxima is, we need to find the local minima around it
    # In other words, we need to find the first minima on either side of the local max
    minima = np.array(argrelmin(signal))

    # Find where the row or column is less than the power surface limit
    mask = np.where(signal <= threshold)
    minima = np.append(minima, mask)

    # Boundaries of the AGW packet
    # if near edge, add the shape of the power
    minima = np.sort(np.append(minima, [0, coord, power_array_shape_for_signal - 1]))

    coords = np.arange(
        minima[np.where(minima == coord)[0] - 1][0],
        minima[np.where(minima == coord)[0] + 1][0] + 1,
    )
    return coords




def compute_wavelet_components(array, dj,dt, s0,mother_wavelet, padding):
    """
    Calculate the wavelet coefficients of the first-order perturbations [Zink and Vincent, 2001]. 
    Uses the wavelet transform described in [Torrence and Campo, 1989]. 

    Arguments:
        array -- The first order perturbations.
        dj -- Resolution of the wavelet transform.
        mother_wavelet -- The wavelet family. Typically, a morlet wavelength is chosen.
                            Resembles the gravity wave packets [Koushik et. al, 2019].

    Returns:
        The wavelet coefficients, periods, scales, and cone of influence for the wavelet
        transformed first order perturbations. This isolates AGWs in altitude and vertical
        wavelength space [Moffet at el, 2011].
    """

    # Set padding
    padding = 1

    wavelet_coef, coef_periods, coef_scales, coef_coi = wavelet(
        array,
        dt=dt,
        pad=padding,
        dj=dj,
        s0=s0,
        mother=mother_wavelet,
    )

    return wavelet_coef, coef_periods, coef_scales, coef_coi


def find_local_maxima(power_array, threshold_val, coi, sig):
    """
    Find the local maxima of a 2D array greater than some threshold value and filtered to be within
    the cone of influence and wave significance.

    Arguments:
        power_array -- The 2D power surface [m^2/s^2].
        threshold_val -- The threshold value. Anything less than this value is considered
                            noise [m^2/s^2].
        coi -- The cone of influence array.
        sig -- The wave significance array.

    Returns:
        The local maxima that are greater than some threshold noise and filtered to be within
        the wavelet coefficient's cone of influence and significance.
    """
   
    # The coordinates of the local maxima greather than some threshold value
    # [Zink and Vincent, 2001] -- Discard values less than 0.01 m^2/s^2 as these are usually noise
    # peak_local_max does the threshold as > 0.01 m^2/s^2, so use 0.011 m^2/s^2 to ignore values equal to 0.01 m^2/s^2
    # Scan the power surface (vertical wavelength-height) for the local maxima
    peaks = peak_local_max(power_array, threshold_abs=threshold_val)

    # filter local maxima within cone of influence and significance interval
    coi_mask = copy.deepcopy(coi)
    peaks = peaks[[coi_mask[tuple(x)] for x in peaks]]

    sig_mask = copy.deepcopy(sig)
    peaks = peaks[[sig_mask[tuple(x)] for x in peaks]]

    print("\n")
    print("Found %s local maxima within cone of influence & significance" % (peaks.shape[0]))
    print("\n")

    return peaks


def extract_boundaries_around_peak(power_array, peaks, peak_nom):
    """
    Determine the boundaries of the AGW packets in vertical wavelength-height space. Scan the 
    power surface (vertical wavelength-height) for the boundaries pertaining to when the value of the power
    surface at the local maxima falls to 1/4 of its value in all directions or when it rises again. 
    Based on [Zink and Vincent, 2001].

    Arguments:
        power_array -- The power surface [m^2/s^2].
        peaks -- The coordinates for the local maxima of the power surface.
        peak_nom -- The chosen coordinates to analyze.

    Returns:
        The recorded boundaries are defined in a 2D numpy array that matches the size of the power surface
    """
    peak_containers = np.zeros(power_array.shape, dtype=bool)

    chosen_peak_coordinate = peaks[peak_nom]
    row_peak, col_peak = tuple(chosen_peak_coordinate)

    # The power surface limit
    power_surface_limit = 0.25 * (power_array[row_peak, col_peak])

    # Find boundaries around the local maxima
    # Extract the row and column associated with the local maximum
    search_around_peak_row = power_array[row_peak, :]
    search_around_peak_col = power_array[:, col_peak]

    # Boundaries for the rows and columns associated with the local maxima
    col_coords = get_boundaries(
        search_around_peak_row, power_array.shape[1], col_peak, power_surface_limit
    )

    row_coords = get_boundaries(
        search_around_peak_col, power_array.shape[0], row_peak, power_surface_limit
    )

    # The area pertaining to the boundary of the identified AGW packet is set as True.
    peak_containers[row_coords[:, np.newaxis], col_coords] = True

    return peak_containers,row_coords, col_coords



def convert_seconds_to_timestamp(dataframe, initial_timestamp_for_flight):
    """
    Convert remaining time column in seconds to a timestamp with date and time in UTC.

    Arguments:
        dataframe -- The Pandas DataFrame
        initial_timestamp_for_flight -- The intial timestamp for the flight [Timestamp]

    Returns:
        A new time column in UTC. Moved new column to be the 2nd column in DataFrame.
    """    
    
    dataframe["Time [UTC]"] = pd.to_datetime(dataframe["Time [sec]"], unit='s', origin=str(initial_timestamp_for_flight))
    column_to_move =  dataframe.pop("Time [UTC]")
    dataframe.insert(1, "Time [UTC]", column_to_move )
    
    return dataframe



def peaks_inside_rectangular_boundary(peaks, boundaries_for_rows, boundaries_for_cols):
    """
    Determine if other local maxima are located within the rectangular boundary in question.
    
    Arguments:
        peaks -- The list of all local maxima known.
        boundaries_for_rows -- The indices associated with the length of the rectangular boundary.
        boundaries_for_cols -- The indices associated with the height of the rectangular boundary.

    Returns:
        The list of indices associated with the local maxima located inside the rectangular boundary.
    """    
    x1, x2 =  boundaries_for_rows
    y1, y2 =  boundaries_for_cols
    
    # list to store lolca maxima found inside rectangular boundary
    peaks_within_boundary = []

    for coords in peaks:
        if (x1 < coords[0] and coords[0] < x2):
                if (y1 < coords[1] and coords[1] < y2):
                    peaks_within_boundary.append(coords)
        
    return peaks_within_boundary

def calculate_horizontal_wind_variance(inverted_u_coeff, inverted_v_coeff,peaks_within_boundary_list,peaks,peak_nom):
    """
    Calculate the horizontal wind variance, which is the sum of the reconstructed zonal and meridional wind perturbation
    wavelet coefficients. According to [Zink and Vincent, 2001], if the rectangular boundaries of the reconstructed wavelet
    coefficients overlap, the horizontal wind variance must be divided in equal parts among them.

    Arguments:
        inverted_u_coeff -- The reconstructed zonal wind perturbation.
        inverted_v_coeff -- he reconstructed meridional wind perturbation.
        peaks_within_boundary_list -- List of all the local maxima present within current rectangular boundary.
        peaks -- The list of all known local maxima.
        peak_nom -- The index of the local maxima being investigated.

    Returns:
        The horizontal wind variance [m^2/s^2].
    """        
    # [Zink and Vincent, 2001] -- vertical extent: the FWHM of the horizontal wind variance
    # wind variance -- the sum of the reconstructed u and v wavelet coefficients
    horizontal_wind_variance = np.abs(inverted_u_coeff) ** 2 + np.abs(inverted_v_coeff) ** 2
    
    # If peaks is equal to itself, leave it be
    # If multiple peaks found inside boundary, divide horizontal wind variance among all the peaks equally
    if len(peaks_within_boundary_list)==1 and np.array(peaks_within_boundary_list[0] == peaks[peak_nom]).all():
        peaks_within_boundary_list = peaks[peak_nom]
        horizontal_wind_variance = horizontal_wind_variance
    else:
        horizontal_wind_variance = horizontal_wind_variance/len(peaks_within_boundary_list)
        
    return horizontal_wind_variance


def inverse_wavelet_transform(wavelet_coef,peak_containers,wavelet_scales,dj,dt):
    """
    Reconstruct the perturbations associated with the potential gravity wave packet 
    through the inverse wavelet transform of the wavelet coefficients centered within the 
    pre-determined rectangular boundary [Zink and Vincent, 2001].

    Arguments:
        wavelet_coef -- The wavelet coefficients computed from the wavelet transform for the perturbations.
        peak_containers -- The array that contains the rectangular boundary around the wave packet.
        wavelet_scales -- The scales computed from the wavelet transform.
        dj -- The spacing between discrete scales.
        dt -- The spatial resolution [m].

    Returns:
        The reconstructed perturbations associated with the wave packet at the local maxima.
    """    
    
    # Constants for the inverse wavelet transform
    # [Torrence and Compo, 1998] -- Table 2
    C_delta_morlet = 0.776 #  reconstruction factor
    psi0_morlet = np.pi**(1/4) # to remove energy scaling
    # Want the exact parameters used in the initial calculation of the wavelet coefficients
    wavelet_constant = dj*np.sqrt(dt)/ (C_delta_morlet*psi0_morlet)
    
    copied_wavelet_coef = copy.deepcopy(wavelet_coef)
    
    # All points outside of this rectangular boundary are 0
    copied_wavelet_coef = copied_wavelet_coef*peak_containers
    wavelet_div_scale = np.divide(copied_wavelet_coef.T,np.sqrt(wavelet_scales))
    
    # Sum over all scales -- NOTE order of dimensions swamped
    copied_wavelet_coef = np.multiply(wavelet_div_scale.sum(axis=1),wavelet_constant)
    
    return copied_wavelet_coef


def wave_packet_FWHM_indices(horizontal_wind_variance):
    """
    The vertical extent of the wave packet is determined by the Full Width Half Max (FWHM) of 
    the horizontal wind variance [Zink and Vincent, 2001].

    Arguments:
        horizontal_wind_variance -- The sum of the reconstructed wind perturbation wavelet coefficients [m^2/s^2].

    Returns:
        The indices representing the vertical extent of the wave packet, the index associated with the maximum value, and the value of the half maximum.
    """    
    # https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    # Find the maximum value and index of maximum value of the horizontal wind variance
    max_value = np.max(horizontal_wind_variance)
    max_value_index = np.argmax(horizontal_wind_variance)
    
    # Find the half maximum of the horizontal wind variance
    half_max = max_value/2

    # Find indices closest to the half maximum on either side of the local maximum.
    vertical_extent_coordx = next( ( i for i in range(max_value_index,-1,-1) if horizontal_wind_variance[i] <= half_max), 0)
    vertical_extent_coordy = next((i for i in range(max_value_index, len(horizontal_wind_variance)) if horizontal_wind_variance[i] <= half_max), len(horizontal_wind_variance) - 1)
    
    return vertical_extent_coordx, vertical_extent_coordy, max_value_index, half_max


# Following equations are copied from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """
    # Following equations are copied from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """
    # Following equations are copied from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """
    # Following equations are copied from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y
