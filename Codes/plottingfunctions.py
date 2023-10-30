import numpy as np
import matplotlib.pyplot as plt


def plot_vertical_profiles_with_residual_perturbations(
    dataframe,
    u_zonal_speed,
    v_meridional_speed,
    temperature,
    v_meridional_fit,
    u_zonal_fit,
    temperature_fit,
    u_zonal_perturbations,
    v_meridional_perturbations,
    temperature_perturbations,
    time,
    path_to_save_figure,
    save_fig=False,
):
    """
    Plot the original vertical temperature, zonal wind speed, and meridional wind speed profiles
    and the first-order perturbations of these variables side by side.

    Arguments:
        dataframe -- The Pandas DataFrame.
        u_zonal_speed -- The original zonal wind speed perturbations [m/s].
        v_meridional_speed -- The original meridional wind speed perturbations [m/s].
        temperature -- The original temperature perturbations [C].
        v_meridional_fit -- A vector of the polynomial coefficients that minimizes the squared error
                            of the meridional wind speed perturbations.
        u_zonal_fit -- A vector of the polynomial coefficients that minimizes the squared error
                        of the zonal wind speed perturbations.
        temperature_fit -- A vector of the polynomial coefficients that minimizes the squared error
                            of the temperature perturbations.
        u_zonal_perturbations -- The first-order perturbations for the zonal wind [m/s].
        v_meridional_perturbations -- The first-order perturbations for the meridional wind [m/s].
        temperature_perturbations -- The first-order perturbations for the temperature [C].
        time -- The starting time of the radiosonde launch [Pandas Timestamp].
        path_to_save_figure -- Path to where to save the figure [string].

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False}).

    Returns:
        Plots the original vertical profiles for the temperature, zonal, and meridional wind speeds and
        the computed first-order perturbations for the same variables.
    """
    fig, axs = plt.subplots(3, 2, figsize=[8, 10], sharey=True)

    fig.suptitle(str(time) + " UTC")

    axs[0, 0].plot(u_zonal_speed, dataframe["Geopot [m]"], color="navy", linewidth=1)

    axs[0, 0].plot(
        u_zonal_fit[0] * dataframe["Geopot [m]"] ** 2
        + u_zonal_fit[1] * dataframe["Geopot [m]"]
        + u_zonal_fit[2] * np.ones(len(u_zonal_speed)),
        dataframe["Geopot [m]"],
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[0, 0].set_title("Radiosonde Measurements")
    axs[0, 0].set_xlabel("Zonal Speed [m/s]")

    axs[0, 1].plot(
        u_zonal_perturbations, dataframe["Geopot [m]"], color="navy", linewidth=1
    )

    axs[0, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)

    axs[0, 1].set_xlabel("Residual [m/s]")
    axs[0, 1].set_title("Residual Perturbations")

    axs[1, 0].plot(
        v_meridional_speed, dataframe["Geopot [m]"], color="navy", linewidth=1
    )

    axs[1, 0].plot(
        (
            v_meridional_fit[0] * dataframe["Geopot [m]"] ** 2
            + v_meridional_fit[1] * dataframe["Geopot [m]"]
            + v_meridional_fit[2] * np.ones(len(v_meridional_speed))
        ),
        dataframe["Geopot [m]"],
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[1, 0].set_xlabel("Meridional Speed [m/s]")

    axs[1, 1].plot(
        v_meridional_perturbations, dataframe["Geopot [m]"], color="navy", linewidth=1
    )
    axs[1, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)
    axs[1, 1].set_xlabel("Residual [m/s]")

    axs[2, 0].plot(temperature, dataframe["Geopot [m]"], color="navy", linewidth=1)

    axs[2, 0].plot(
        (
            temperature_fit[0] * dataframe["Geopot [m]"] ** 2
            + temperature_fit[1] * dataframe["Geopot [m]"]
            + temperature_fit[2] * np.ones(len(temperature))
        ),
        dataframe["Geopot [m]"],
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[2, 0].set_xlabel("Temperature [C]")

    axs[2, 1].plot(
        temperature_perturbations, dataframe["Geopot [m]"], color="navy", linewidth=1
    )

    axs[2, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)

    axs[2, 1].set_xlabel("Residual [s]")

    for ax in [axs[0, 0], axs[1, 0], axs[2, 0]]:
        ax.set(ylabel="Altitude [m]")

    plt.tight_layout()

    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")
        time_string = str(time.time())
        time_string = time_string.replace(":", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_%s_perturbations.png" % (date_string, time_string),
            bbox_inches="tight",
        )

    return fig


def plot_power_surface(
    dataframe,
    power_array,
    periods,
    peak_container,
    signif,
    coi,
    peaks,
    colormap,
    time,
    path_to_save_figure,
    save_fig=False,
):
    """
    Plot the power surface.

    Arguments:
        dataframe -- The Pandas DataFrame.
        power_array -- The power surface [m^2/s^2].
        periods -- The wavelet coefficient scales; corresponds to the vertical wavelength [m].
        peak_container -- The true/false array marking the boundaries of the AGW wave packet.
        signif -- The wave significance array.
        coi -- _The cone of influence significane array.
        peaks -- _The coordinates of the local maxima.
        colormap -- The colormap for the power surface.
        time -- The starting time of the radiosonde launch [Pandas Timestamp].
        path_to_save_figure -- Path to where to save the figure [string].

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        Plots the power surface with the local maxima identified.
    """
    power_array = np.log(power_array.copy())
    power_array[power_array < 0] = 0

    fig = plt.figure(figsize=[8, 6])
    plt.contourf(
        dataframe["Geopot [m]"],
        (periods) / 1000,
        (power_array),
        levels=200,
        cmap=colormap,
    )
    cb = plt.colorbar()
    cb.set_label("Power [m^2/s^2]")

    plt.contour(
        dataframe["Geopot [m]"],
        periods / 1000,
        peak_container,
        colors="r",
        levels=[0.5],
    )

    plt.contour(
        dataframe["Geopot [m]"],
        periods / 1000,
        signif,
        colors="black",
        levels=[0.5],
    )

    plt.contour(
        dataframe["Geopot [m]"],
        periods / 1000,
        coi,
        colors="black",
        levels=[0.5],
    )
    plt.scatter(
        dataframe["Geopot [m]"][peaks.T[1]],
        periods[peaks.T[0]] / 1000,
        c="red",
        marker=".",
    )

    plt.yscale("log")

    plt.xlim(
        [
            dataframe["Geopot [m]"].min(),
            dataframe["Geopot [m]"].max(),
        ]
    )
    #

    plt.ylabel("Vertical Wavelength [km]")
    plt.xlabel("Altitude [m]")

    plt.title("Power Surface at " + str(time) + " UTC")
    plt.tight_layout()

    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")
        time_string = str(time.time())
        time_string = time_string.replace(":", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_%s_power_surface.png" % (date_string, time_string),
            bbox_inches="tight",
        )
    return fig



def plot_potential_temperature_vs_pressure(potential_temperature_array, temperature_array, pressure_array):
    """
    Creates a plot of the potential temperature anf the climatological atmospheric temperature versus atmospheric pressure.

    Arguments:
        potential_temperature_array -- Potential temperature array [K].
        temperature_array -- Atmospheric temperature array [K]
        pressure_array -- Atmospheric pressure array [K]

    Returns:
        A figure
    """
    
    fig = plt.figure()
    plt.plot(potential_temperature_array, pressure_array, label="Potential Temperature [K]")
    plt.plot(temperature_array, pressure_array, label="Temperature [K]")

    plt.gca().invert_yaxis()
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [hPa]")
    plt.legend(loc ='best',fancybox=True )
    return fig
