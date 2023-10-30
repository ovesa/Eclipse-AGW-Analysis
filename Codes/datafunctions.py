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
    condition2 = dataframe.eq(999999.0).any(1)
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


def compute_second_order_polynomial_fits(dataframe, array, order):
    """
    Calculate a least squares polynomial fit to the necessary variable. This step is necessary to
    remove the background state from the main vertical profile of the variable.

    Arguments:
        dataframe -- The Pandas DataFrame.
        array -- The array to perform the least squares polynomial fit.
        order -- The degree of the fitting polynomial.

    Returns:
        A vector of the polynomial coefficients that minimizes the squared error.
    """

    array_fit = np.polyfit(dataframe["Geopot [m]"], array, order)

    return array_fit


def derive_first_order_perturbations(dataframe, perturbations, polynomial_fit):
    """
    Derive the first-order perturbations, which are thought to caused by atmospheric gravity waves
    [Moffat‐Griffin et. al, 2011]. After computing the least squares polynomial fit for the
    perturbation (specifically the zonal and meridional wind perturbations), subtract it from the
    original vertical profiles. This removes the background pertubations [Moffat‐Griffin et. al, 2011]
    & [Vincent and Alexander, 2000].

    Arguments:
        dataframe -- The Pandas DataFrame.
        perturbations -- The perturbation array to compute the polynomial fit on.
        polynomial_fit -- A vector of the polynomial coefficients that minimizes the squared error.

    Returns:
        The first-order perturbations that are assumed to be due to atmospheric gravity waves.
    """

    # Apply polynomia fit then subtract from main vertical profile to remove background states
    # Don't need to create a new array because we are using the recently created evenly spaced height grid
    first_order_perturbations = perturbations - (
        polynomial_fit[0] * dataframe["Geopot [m]"] ** 2
        + polynomial_fit[1] * dataframe["Geopot [m]"]
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


def compute_wavelet_components(array, dj,dt, s0,mother_wavelet, spatial_resolution, padding):
    """
    Calculate the wavelet coefficients of the first-order perturbations [Zink and Vincent, 2001]. 
    Uses the wavelet transform described in [Torrence and Campo, 1989]. 

    Arguments:
        array -- The first order perturbations.
        dj -- Resolution of the wavelet transform.
        mother_wavelet -- The wavelet family. Typically, a morlet wavelength is chosen.
                            Resembles the gravity wave packets [Koushik et. al, 2019].
        spatial_resolution -- The spatial resolution of the data.

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
    coi_mask = copy.deepcopycoi
    peaks = peaks[[coi_mask[tuple(x)] for x in peaks]]

    sig_mask = copy.deepcopy(sig)
    peaks = peaks[[sig_mask[tuple(x)] for x in peaks]]

    print("Found %s peaks within cone of influence & siugnificance" % (peaks.shape[0]))
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
    return peak_containers



def convert_seconds_to_timestamp(dataframe, initial_timestamp_for_flight):
    """
    Convert remaining time column in seconds to a timestamp with date and time in UTC.

    Arguments:
        dataframe -- The Pandas DataFrame
        initial_timestamp_for_flight -- The intial timestamp for the flight [Timestamp]

    Returns:
        A new time column in UTC.
    """    
    dataframe["Time [UTC]"] = pd.to_datetime(dataframe["Time [sec]"], unit='s', origin=str(initial_timestamp_for_flight))
    return dataframe