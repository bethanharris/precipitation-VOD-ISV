"""
Functions to aid plotting of data.

Bethan Harris, UKCEH, 04/01/2021
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase


def binned_cmap(levels, colormap_key, fix_colours=[], bad_colour='white', extend='neither'):
    """
    Create a discretised colourmap based on given contour levels.
    Parameters:
    levels (list or numpy array): contour levels at which colour should change.
    colormap_key (str): key to a matplotlib colormap, e.g. 'viridis'.
    fix_colours (list of (int, str) tuples): colours to override, e.g. [(0, 'white')]
                                             will change first colour to white regardless of colormap chosen.
                                             Default no overrides.
    bad_colour (str): colour to use for invalid data. Default white.
    extend (str): determines whether to include colours for extension triangles at either end of colourbar.
                  May be 'min', 'max', 'both' or (default) 'neither'.
    Returns:
    (matplotlib colormap): discretised colormap with number of colours to match contour levels.
    (matplotlib norm): norm to map data to correct colours based on contour levels.
    """
    # find number of colours required for colourbar
    number_colors = len(levels) - 1
    # extra colours required if colourbar has extension triangles
    if extend == 'both':
        number_colors += 2
    elif extend == 'max':
        number_colors += 1
    elif extend == 'min':
        number_colors += 1
    # get this number of colours evenly spaced along desired colormap
    colormap = cm.get_cmap(colormap_key, number_colors)
    colors = list(colormap(np.arange(number_colors)))
    for (idx, colour) in fix_colours: # override any specified colours
        if extend in ['min', 'both']:
            idx += 1
        colors[idx] = colour
    # create colourmap out of these colours
    if extend == 'both':
        colormap = mpl.colors.ListedColormap(colors[1:-1], "")
        colormap.set_under(colors[0])
        colormap.set_over(colors[-1])
    elif extend == 'max':
        colormap = mpl.colors.ListedColormap(colors[:-1], "")
        colormap.set_over(colors[-1])
    elif extend == 'min':
        colormap = mpl.colors.ListedColormap(colors[1:], "")
        colormap.set_under(colors[0])
    else:
        colormap = mpl.colors.ListedColormap(colors, "")
    colormap.set_bad(bad_colour)
    # create norm to map to colourmap
    norm = mpl.colors.BoundaryNorm(levels, ncolors=len(levels)-1, clip=False)
    return colormap, norm


# colormap scaling code authors: Paul H, Horea Christian, Leonor Carcia Gutierrez.
#  Modified to work properly when abs(min) > max

def auto_remap(data):
    start = 0
    midpoint = 0.5
    stop = 1.0
    if np.nanmin(data) >= 0:
        raise ValueError('You do not need to rescale your cmap to center zero.')
    if np.nanmax(data) > abs(np.nanmin(data)):
        start = (np.nanmax(data) - abs(np.nanmin(data))) / (2. * np.nanmax(data))
        midpoint = abs(np.nanmin(data)) / (np.nanmax(data) + abs(np.nanmin(data)))
        stop = 1.0
    if np.nanmax(data) == abs(np.nanmin(data)):
        start = 0
        midpoint = 0.5
        stop = 1.0
    if np.nanmax(data) < abs(np.nanmin(data)):
        start = 0
        midpoint = abs(np.nanmin(data)) / (np.nanmax(data) + abs(np.nanmin(data)))
        stop = (abs(np.nanmin(data)) + np.nanmax(data)) / (2. * abs(np.nanmin(data)))
    return start, midpoint, stop


def remappedColorMap(cmap, data, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
    cmap : The matplotlib colormap to be altered
    data: You can provide your data as a numpy array, and the following
        operations will be computed automatically for you.
    start : Offset from lowest point in the colormap's range.
        Defaults to 0.0 (no lower ofset). Should be between
        0.0 and 0.5; if your dataset vmax <= abs(vmin) you should leave
        this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
    midpoint : The new center of the colormap. Defaults to
        0.5 (no shift). Should be between 0.0 and 1.0; usually the
        optimal value is abs(vmin)/(vmax+abs(vmin))
    stop : Offset from highets point in the colormap's range.
        Defaults to 1.0 (no upper ofset). Should be between
        0.5 and 1.0; if your dataset vmax >= abs(vmin) you should leave
        this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))
    '''

    start, midpoint, stop = auto_remap(data)

    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False),
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


class StripyPatch(HandlerBase):
    def __init__(self, color_list, **kw):
        HandlerBase.__init__(self, **kw)
        self.color_list = color_list
        self.num_stripes = len(color_list)
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
                          width / self.num_stripes, 
                          height, 
                          fc=self.color_list[i], 
                          transform=trans)
            stripes.append(s)
        return stripes