import numpy as np
import matplotlib.pyplot as plt
import copy

def plot_vertical_profiles_with_residual_perturbations(
    height_km,
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
        height_km -- Height array [km].
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
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        path_to_save_figure -- Path to where to save the figure [string].

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False}).

    Returns:
        Plots the original vertical profiles for the temperature, zonal, and meridional wind speeds and
        the computed first-order perturbations for the same variables.
    """
    fig, axs = plt.subplots(3, 2, figsize=[8, 10], sharey=True)

    fig.suptitle(str(time) + " UTC")

    axs[0, 0].plot(u_zonal_speed,height_km, color="navy", linewidth=1)

    axs[0, 0].plot(
        u_zonal_fit[0] * height_km ** 2
        + u_zonal_fit[1] * height_km
        + u_zonal_fit[2] * np.ones(len(u_zonal_speed)),
        height_km,
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[0, 0].set_title("Radiosonde Measurements")
    axs[0, 0].set_xlabel("Zonal Speed [m/s]")

    axs[0, 1].plot(
        u_zonal_perturbations, height_km, color="navy", linewidth=1
    )

    axs[0, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)

    axs[0, 1].set_xlabel("Residual [m/s]")
    axs[0, 1].set_title("Residual Perturbations")

    axs[1, 0].plot(
        v_meridional_speed, height_km, color="navy", linewidth=1
    )

    axs[1, 0].plot(
        (
            v_meridional_fit[0] * height_km ** 2
            + v_meridional_fit[1] * height_km
            + v_meridional_fit[2] * np.ones(len(v_meridional_speed))
        ),
        height_km,
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[1, 0].set_xlabel("Meridional Speed [m/s]")

    axs[1, 1].plot(
        v_meridional_perturbations, height_km, color="navy", linewidth=1
    )
    axs[1, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)
    axs[1, 1].set_xlabel("Residual [m/s]")

    axs[2, 0].plot(temperature, height_km, color="navy", linewidth=1)

    axs[2, 0].plot(
        (
            temperature_fit[0] * height_km ** 2
            + temperature_fit[1] * height_km
            + temperature_fit[2] * np.ones(len(temperature))
        ),
        height_km,
        color="k",
        linestyle="--",
        linewidth=1.5,
    )

    axs[2, 0].set_xlabel("Temperature [C]")

    axs[2, 1].plot(
        temperature_perturbations, height_km, color="navy", linewidth=1
    )

    axs[2, 1].axvline(x=0, color="k", linestyle="--", linewidth=1.5)

    axs[2, 1].set_xlabel("Residual [C]")

    for ax in [axs[0, 0], axs[1, 0], axs[2, 0]]:
        ax.set(ylabel="Altitude [km]")
        ax.yaxis.get_ticklocs(minor=True)
        ax.xaxis.get_ticklocs(minor=True)
        ax.minorticks_on()
        ax.set_ylim([height_km.min(),height_km.max()])


    fig.tight_layout()

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
    height_km,
    power_array,
    periods,
    peak_container,
    signif,
    coi,
    peaks,
    colormap,
    time, nom,
    path_to_save_figure,
    save_fig=False,
):
    """
    Plot the power surface.

    Arguments:
        height_km -- Height array [km]
        power_array -- The power surface [m^2/s^2].
        periods -- The wavelet coefficient scales; corresponds to the vertical wavelength [m].
        peak_container -- The true/false array marking the boundaries of the AGW wave packet.
        signif -- The wave significance array.
        coi -- _The cone of influence significane array.
        peaks -- _The coordinates of the local maxima.
        colormap -- The colormap for the power surface.
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        nom -- The index of the local maximum being investigated ~ the potential GW.
        path_to_save_figure -- Path to where to save the figure [string].

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        Plots the power surface with the local maxima identified.
    """
    copied_power_array = np.log(copy.deepcopy(power_array))
    copied_power_array[copied_power_array < 0] = 0

    fig, ax = plt.subplots(1,1,figsize=[8, 6])
    
    im = ax.contourf(
        height_km,
        (periods) / 1000,
        (copied_power_array),
        levels=200,
        cmap=colormap,vmin=0,vmax=9
    )
    
    ax.yaxis.get_ticklocs(minor=True)
    ax.xaxis.get_ticklocs(minor=True)

    ax.minorticks_on()
    
    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel("Power [m$^2$/s$^2$]")


    ax.contour(
        height_km,
        periods / 1000,
        signif,
        colors="black",
        levels=[0.5],
    )

    ax.contour(
        height_km,
        periods / 1000,
        coi,
        colors="black",
        levels=[0.5],
    )
    
    ax.contourf(
        height_km,
        periods / 1000,
        ~coi,
        colors="none",
        levels=[0.5,1],hatches = ['x']
    )

    ax.scatter(
        height_km[peaks.T[1]],
        periods[peaks.T[0]] / 1000,
        c="red",s=50,
        marker=".", edgecolor='k'
    )
    
    ax.contour(
        height_km,
        periods / 1000,
        peak_container,
        colors="r",
        levels=[0.5],
    )


    ax.set_yscale("log")

    ax.set_xlim(
        [
            height_km.min(),
            height_km.max(),
        ]
    )

    ax.set_ylabel("Vertical Wavelength [km]")
    ax.set_xlabel("Altitude [km]")

    ax.set_title("Power Surface at " + str(time) + " UTC")
    plt.tight_layout()

    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")
        time_string = str(time.time())
        time_string = time_string.replace(":", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_%s_wave%s_power_surface.png" % (date_string, time_string, nom+1),
            bbox_inches="tight",
        )
    return fig


def plot_potential_temperature_vs_pressure(potential_temperature_array, temperature_array, pressure_array, time, path_to_save_figure,save_fig=False,):
    """
    Creates a plot of the potential temperature anf the climatological atmospheric temperature versus atmospheric pressure.
    A
    rguments:
        potential_temperature_array --  Potential temperature array [K].
        temperature_array -- tmospheric temperature array [K].
        pressure_array -- Atmospheric pressure array [hPa].
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        path_to_save_figure -- Path to where to save the figure [string]

    Keyword Arguments:
        save_fig --  Keyword to save the figure (default: {False})

    Returns:
        A figure.
    """
    
    fig, ax = plt.subplots(1,1,figsize=[8, 6])    
    ax.plot(potential_temperature_array, pressure_array, color='k', label="Potential Temperature [K]")
    ax.plot(temperature_array, pressure_array, color='k', linestyle='--', label="Temperature [K]")

    ax.invert_yaxis()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Pressure [hPa]")
    ax.legend(loc ='upper left',fancybox=True)
    ax.yaxis.get_ticklocs(minor=True)
    ax.xaxis.get_ticklocs(minor=True)

    ax.minorticks_on()
    fig.tight_layout()

    if save_fig:
            date_string = str(time.date())
            date_string = date_string.replace("-", "")
    
            fig.savefig(
                path_to_save_figure
                + "/%s_potential_temperature_vs_pressure.png" % (date_string,),
                bbox_inches="tight",
        )

    return fig


def perturbations_associated_with_dominant_vertical_wavelengths(zonal_wind_perturbation, meridional_wind_perturbation,temperature_perturbations,height_km, time, nom,path_to_save_figure,save_fig=False,):
    """
    Figure of the zonal winds, meridional winds, and temperature associated with the gravity wave packet.


    Arguments:
        zonal_wind_perturbation -- Zonal wind perturbation corresponding to the gravity wave packet [m/s].
        meridional_wind_perturbation -- Meridional wind perturbation corresponding to the gravity wave packet [m/s].
        temperature_perturbations -- Temperature perturbations corresponding to the gravity wave packet [C].
        height_km -- Height array [km]
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        nom -- The index of the local maximum being investigated ~ the potential GW.
        path_to_save_figure -- Path to where to save the figure [string]

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        A figure.
    """
    
    fig, ax = plt.subplots(1,1,figsize=[8, 6])  
    ax.set_title("Dominant Vertical Wavelengths \n Associated with Wave")
    ax.plot(zonal_wind_perturbation,height_km, color="k", linewidth=1.5, zorder=0, label="Zonal Wind Speed [m/s]")
    ax.plot(meridional_wind_perturbation,height_km, color="red", linewidth=1.5, zorder=0,label="Meridional Wind Speed [m/s]")
    ax.plot(temperature_perturbations,height_km, color="blue", linewidth=1.5, linestyle='--', zorder=0,label="Temperature [C]")
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel("Perturbations")
    ax.legend(loc ='lower center',fancybox=True,ncol=3,prop={'size':9})
    ax.yaxis.get_ticklocs(minor=True)
    ax.xaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    fig.tight_layout()
    
    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_wave%s_dominant_perturbations.png" % (date_string, nom+1),
            bbox_inches="tight",
        )


    return fig

def plot_hodograph(zonal_wind_perturbation, meridional_wind_perturbation,height_km,time, nom,path_to_save_figure,save_fig=False,):
    """
    Figure of the zonal versus meridional wind perturbations associated with the gravity wave packet -- hodograph.
    
    Arguments:
        zonal_wind_perturbation -- Zonal wind perturbation corresponding to the gravity wave packet [m/s].
        meridional_wind_perturbation -- Meridional wind perturbation corresponding to the gravity wave packet [m/s].
        height_km -- Height array [km]
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        nom -- The index of the local maximum being investigated ~ the potential GW.
        path_to_save_figure -- _description_

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        A figure.
    """
    
    fig, ax = plt.subplots(1,1,figsize=[8, 6])  
    ax.plot(zonal_wind_perturbation, meridional_wind_perturbation, color="k", linewidth=1.5, linestyle='--',zorder=0)
    
    ax.set_title("Hodograph Analysis")

    ax.scatter(
        zonal_wind_perturbation[0],
        meridional_wind_perturbation[0],
        color="r",
        marker="D",
        s=35,lw=1.5,
        zorder=1, edgecolor='k', label="%.2f km" % (height_km.iloc[0])
    )

    # ax.annotate(
    #     "%.2f km" % (height_km.iloc[0]), 
    #    xy= (zonal_wind_perturbation[0], meridional_wind_perturbation[0]), xycoords='data',xytext=(3, 1), textcoords='offset points',
    # )
    
    middle_x_point, middle_y_point = len(zonal_wind_perturbation)//2, len(meridional_wind_perturbation)//2
    ax.scatter(
        zonal_wind_perturbation[middle_x_point],
        meridional_wind_perturbation[middle_y_point],
        color="pink",
        marker="o",
        s=35,lw=1.5,
        zorder=1, edgecolor='k', label="%.2f km" % (height_km.iloc[middle_x_point])
    )

    # ax.annotate(
    #     "%.2f km" % (height_km.iloc[middle_x_point]), 
    #    xy= (zonal_wind_perturbation[middle_x_point], meridional_wind_perturbation[middle_x_point]), xycoords='data',xytext=(3, 1), textcoords='offset points',
    # )

    ax.scatter(
        zonal_wind_perturbation[-1],
        meridional_wind_perturbation[-1],
        color="gold",
        marker="s",
        s=35,lw=1.5,
        zorder=1, edgecolor='k',label="%.2f km" % (height_km.iloc[-1])
    )

    # ax.annotate(
    #     "%.2f km" % (height_km.iloc[-1]),
    #     xy=(zonal_wind_perturbation[-1], meridional_wind_perturbation[-1]) , xycoords='data',xytext=(3, 1), textcoords='offset points',
    # )

    ax.set_xlabel("Zonal Wind Speed [m/s]")
    ax.set_ylabel("Meridional Wind Speed [m/s]")
    
    ax.yaxis.get_ticklocs(minor=True)
    ax.xaxis.get_ticklocs(minor=True)

    ax.minorticks_on()
    ax.legend(loc ='upper center',fancybox=True,ncol=3,prop={'size':10})


    fig.tight_layout()
    
    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_wave%s_simple_hodograph_plot.png" % (date_string, nom+1),
            bbox_inches="tight",
        )

    return fig

def plot_FWHM_wind_variance(horizontal_wind_variance,vertical_extent_coordx, vertical_extent_coordy,max_value_index,half_max,time,nom, path_to_save_figure,save_fig=False,):
    """
    Plot the horizontal wind variance, identifying the peak, the FWHM, and the points associated with the FWHM.

    Arguments:
        horizontal_wind_variance -- The horizontal wind variance [m^2/s^2].
        vertical_extent_coordx -- The left bound index of the FWHM.
        vertical_extent_coordy -- The right bound index of the FWHM.
        max_value_index -- The index corresponding to the maximum value of the horizontal wind variance.
        half_max -- The value of the half maximum of the horizontal wind variance [m^2/s^2].
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        nom -- The index of the local maximum being investigated ~ the potential GW.
        path_to_save_figure -- _description_

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        A figure.
    """
    
    fig, ax = plt.subplots(1,1,figsize=[8, 6])  
    ax.set_title("Detect FWHM")
    ax.plot(np.arange(len(horizontal_wind_variance)), horizontal_wind_variance, color='k',zorder=0,)
    ax.scatter(max_value_index, horizontal_wind_variance[max_value_index], s=40,  color='gold', edgecolor='k',zorder=1)
    ax.axhline(y=half_max, linestyle='--', color='k',linewidth=0.8,zorder=0)
    ax.axvspan(vertical_extent_coordx,vertical_extent_coordy, facecolor='pink', alpha=0.5)
    ax.scatter(vertical_extent_coordx, horizontal_wind_variance[vertical_extent_coordx], s=40, color='red', edgecolor='k',zorder=1)
    ax.scatter(vertical_extent_coordy, horizontal_wind_variance[vertical_extent_coordy], s=40,  color='red', edgecolor='k',zorder=1)


    ax.set_xlim([vertical_extent_coordx-100,vertical_extent_coordy+100])
    ax.set_ylabel(r"Horizontal Wind Variance [m$^2$/s$^2$]")
    ax.set_xlabel("Vertical Extent [indices]")
    fig.tight_layout()
    
    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")

        fig.savefig(
            path_to_save_figure
            + "/%s_wave%s_FWHM.png" % (date_string, nom+1),
            bbox_inches="tight",
        )

    return fig


def plot_hodograph_with_fitted_ellipse(zonal_wind_perturbation, meridional_wind_perturbation,height_km, fitted_zonal_comps,fitted_meridional_comps,centerx,centery, title,magnitude,theta, time,nom,path_to_save_figure,save_fig=False,):
    """
    Plot a hodograph of the zonal versus meridional wind perturbations and fit an ellipse to the points.

    Arguments:
        zonal_wind_perturbation -- The zonal wind perturbations associated with the gravity wave packet [m/s].
        meridional_wind_perturbation -- The meridional wind perturbations associated with the gravity wave packet [m/s].
        height_km -- The height array [km]
        fitted_zonal_comps -- The array corresponding to the fitted parameters of the zonal wind perturbations.
        fitted_meridional_comps -- The array corresponding to the fitted parameters of the meridional wind perturbations.
        centerx -- The center x value of the ellipse.
        centery -- _The center y value of the ellipse.
        title -- The title of the plot.
        magnitude -- The inverse axial ratio [rad/s].
        theta -- The direction of propagation of the GW; parallel to the semi-major axis of the ellipse [radians].
        time -- The starting time & date of the radiosonde launch [Pandas Timestamp].
        nom -- The index of the local maximum being investigated ~ the potential GW.
        path_to_save_figure -- _description_

    Keyword Arguments:
        save_fig -- Keyword to save the figure (default: {False})

    Returns:
        A figure.
    """    
    fig, ax = plt.subplots(1,1,figsize=[8, 6])  
    ax.set_title(title)


    ax.scatter(zonal_wind_perturbation, meridional_wind_perturbation, color='k', s=15, marker='x', zorder=1,label="Data")
    ax.axhline(y=centery,linestyle='--',  color='b', linewidth=0.5, zorder=1)
    ax.axvline(x=centerx,linestyle='--', color='b', linewidth=0.5, zorder=1)
    # ax.scatter(centerx,centery,color='k',marker='x',s=30,zorder=1)
    ax.plot(fitted_zonal_comps, fitted_meridional_comps, color='pink', linewidth=3,zorder=0, label="Ellipse Fit")

    U = magnitude*np.cos(theta)
    V = magnitude*np.sin(theta)
    plt.quiver(centerx,centery,U,V, width=0.003)



   
    ax.scatter(
        zonal_wind_perturbation[0],
        meridional_wind_perturbation[0],
        color="r",
        marker="D",
        s=35,lw=1.5,
        zorder=1, edgecolor='k', label="%.2f km" % (height_km.iloc[0])
    )

    # ax.annotate(
    #     "%.2f km" % (dataframe.iloc[0]), 
    #    xy= (zonal_wind_perturbation[0], meridional_wind_perturbation[0]), xycoords='data',xytext=(3, 1), textcoords='offset points',
    # )
    
    middle_x_point, middle_y_point = len(zonal_wind_perturbation)//2, len(meridional_wind_perturbation)//2
    ax.scatter(
        zonal_wind_perturbation[middle_x_point],
        meridional_wind_perturbation[middle_y_point],
        color="pink",
        marker="o",
        s=35,lw=1.5,
        zorder=1, edgecolor='k', label="%.2f km" % (height_km.iloc[middle_x_point])
    )

    # ax.annotate(
    #     "%.2f km" % (dataframe.iloc[middle_x_point]), 
    #    xy= (zonal_wind_perturbation[middle_x_point], meridional_wind_perturbation[middle_x_point]), xycoords='data',xytext=(3, 1), textcoords='offset points',
    # )

    ax.scatter(
        zonal_wind_perturbation[-1],
        meridional_wind_perturbation[-1],
        color="gold",
        marker="s",
        s=35,lw=1.5,
        zorder=1, edgecolor='k',label="%.2f km" % (height_km.iloc[-1])
    )

    ax.set_xlabel("Zonal Wind Speed [m/s]")
    ax.set_ylabel("Meridional Wind Speed [m/s]")
    
    ax.yaxis.get_ticklocs(minor=True)
    ax.xaxis.get_ticklocs(minor=True)

    ax.minorticks_on()
    ax.legend(loc='upper left', fancybox=True)

    fig.tight_layout()
    
    if save_fig:
        date_string = str(time.date())
        date_string = date_string.replace("-", "")
 

        fig.savefig(
            path_to_save_figure
            + "/%s_wave%s_hodograph_analysis.png" % (date_string, nom+1),
            bbox_inches="tight",
        )

    return fig


