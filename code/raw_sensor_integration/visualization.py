"""Echogram visualization utilities."""

import matplotlib.pyplot as plt


# Canonical colormaps for echogram display.
#
# ECHOGRAM_CMAP  — use when showing the echogram on its own.
# ECHOGRAM_MASK_CMAP — use when a bottom detection line or mask is overlaid.
ECHOGRAM_CMAP = "viridis"

ECHOGRAM_MASK_CMAP = "viridis"


def plot_echogram(
    data,
    title="Echogram",
    vmin=-80,
    vmax=-20,
    cmap=ECHOGRAM_CMAP,
    ax=None,
    figsize=(14, 6),
    ylabel="Range (m)",
    range_array=None,
):
    """
    Plot echogram with standard acoustic visualization settings.

    Parameters
    ----------
    data        : 2D array (n_samples, n_pings)
    title       : plot title
    vmin, vmax  : colour scale limits (dB)
    cmap        : matplotlib colormap name
    ax          : existing Axes to plot into (creates new figure if None)
    figsize     : figure size when creating a new figure
    ylabel      : y-axis label
    range_array : optional 1D array of range values in meters;
                  when provided the y-axis shows depth in meters

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extent = None
    if range_array is not None:
        n_pings = data.shape[1]
        extent = [0, n_pings, range_array[-1], range_array[0]]

    im = ax.imshow(
        data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", extent=extent
    )
    ax.set_xlabel("Ping number")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ({data.shape[0]} × {data.shape[1]})")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Power (dB)")

    return fig, ax
