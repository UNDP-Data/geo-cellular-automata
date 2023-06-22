import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from camodel import model as m
from typing import  Dict
import rasterio.coords
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from matplotlib.widgets import CheckButtons, Button, RadioButtons
import math
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

def animate(source_pattern, outfile, delay=100, compression=20):

    import glob
    import subprocess

    fmt = outfile.split('.')[-1]

    if fmt == 'gif':
        cmd = ['convert', '-loop', '50', '-dither', 'None', '-colors', '256', '-delay', str(delay/10), source_pattern, outfile]
    elif fmt == 'mp4':
        fps = 1000 / delay
        cmd = ['ffmpeg', '-r', str(fps), '-pattern_type', 'glob', '-i', source_pattern, '-c:v', 'libx264',
               '-crf', str(compression), '-pix_fmt', 'yuv420p', '-y', outfile]
    else:
        raise ValueError('Unsupported animation format. Select "gif" or "mp4.')

    result = subprocess.call(cmd, stderr=subprocess.STDOUT)

    #remove original files
    for tmpfilename in glob.glob(source_pattern):
        os.remove(tmpfilename)

    print('Conversion result: {}'.format(result))
    return result

def plot_neigh_bars(binary_hrea_array=None, target_year='first'):
    """
    Cretae a bar plor showing the number of neighbours for all pixels that have been turned
    on in an interval delineated by any two consecutive years
    :param binary_hrea_array:
    :param target_year:
    :return:
    """

    assert target_year in ['first', 'last'], f'target_year={target_year} is invalid. valid values are {["first", "last"]}'

    y0s = binary_hrea_array.dtype.names[:-1]
    y1s = binary_hrea_array.dtype.names[1:]

    nneighs = 9

    years_labels = []
    df = pd.DataFrame(columns=['nneigh', 'perc', 'years'])

    #aggregate the data
    for y0, y1 in zip(y0s, y1s):
        neigh, percs, counts = m.compute_neighbour_stats(
            binary_array=binary_hrea_array, y0=y0, y1=y1, target_year=y0 if target_year == 'first' else y1
        )
        row_name = f'{y0}-{y1}'
        years_labels.append(row_name)

        for n in range(neigh.size):
            df.loc[len(df.index)] = [neigh[n], percs[n], row_name]



    fig, ax = plt.subplots()

    df = df.set_index('years')
    a = df.groupby(["years", "nneigh"]).sum().unstack()

    a.plot(kind='bar', stacked=True, ax=ax, colormap='Paired')
    for c in ax.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [f'{v.get_height():.0f}%' if v.get_height() > 0.5 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center')

    handles, labels = ax.get_legend_handles_labels()
    plt.subplots_adjust(right=.7)
    labels_new = [f"{label.strip('()').split(',')[1]} neigh" for label in labels]
    plt.legend(handles, labels_new, bbox_to_anchor=(1.2, 0.5), loc='center')
    plt.ylabel(f"Percentage")
    plt.yticks()
    nlabel = 'next' if target_year == 'last' else 'previous'
    plt.title(f'The percentage of block turned on grouped by number of neighbours in {nlabel} time step')

    plt.show()


def geo_plot(arrays=Dict[str, np.ndarray], interpolation='nearest',
            mask_val=None, cmap='gist_rainbow',
             arrays_bounds: Dict[str, rasterio.coords.BoundingBox] = None,
             style_name = 'map',

             ):

    #state vars
    current_index = 0
    layer_names = list(arrays.keys())

    state = {'current_layer_name': layer_names[0]}
    status = [False for e in layer_names]
    status[0] = True
    current_bounds = arrays_bounds[state['current_layer_name']]
    names = arrays[state['current_layer_name']].dtype.names
    style_names = ['map', 'satellite']

    styles = dict(zip(style_names, (cimgt.OSM(), cimgt.QuadtreeTiles())))
    style = styles[style_name]

    gl = None

    fig = plt.figure(figsize=(10, 10))  # open matplotlib figure

    ax = plt.axes(projection=ccrs.GOOGLE_MERCATOR)  # project using coordinate reference system (CRS) of street map



    current_layer_extent = [current_bounds.left, current_bounds.right, current_bounds.bottom, current_bounds.top]
    ax.set_extent(current_layer_extent)  # set extents
    ax.add_image(style, 12, zorder=2)  # add OSM with zoom specification

    # [top_left[0], bot_right[0], bot_right[1], top_left[1]]
    arr = arrays[state['current_layer_name']][names[current_index]]

    im = ax.imshow(arr,
                   transform=ccrs.PlateCarree(),
                   extent=current_layer_extent,
                   cmap=cmap,
                   interpolation=interpolation,
                   zorder=5,
                   origin='upper',


                   )
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')

    # Declare and register callbacks
    def on_zoom(event_ax):
        xmin, xmax = event_ax.get_xlim()
        ymin, ymax = event_ax.get_ylim()
        zl = get_zoom_level(extent=[xmin, ymin, xmax, ymax], ax=event_ax)
        if zl < 20:
            event_ax.img_factories[0][1] = (zl,)
            event_ax._done_img_factory = False

    def get_fig_size_cm(ax):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= 2.54
        height *= 2.54
        return int(round(width)), int(round(height))

    def scale_to_zoom(scale, tile_size=256, platform='google'):
        if platform == 'google':
            C = 256
        else:
            raise ValueError('Unsupported platform: {}'.format(platform))
        zoom_level = math.log2(C * scale / (tile_size * 40075017))
        return 21 if int(zoom_level) > 21 else int(zoom_level)

    def get_zoom_level(extent, ax):
        """
        Computes the zoom level equivalent of a map extent in the Google Web Mercator projection.

        Arguments:
        extent -- A tuple containing the extent of the map in Web Mercator units (xmin, ymin, xmax, ymax).

        Returns:
        The zoom level equivalent of the map extent.
        """
        # Extract the width and height from the extent tuple
        xmin, ymin, xmax, ymax = extent
        px_width, px_height = get_fig_size_cm(ax)

        map_width = xmax - xmin

        scale = map_width / px_width
        # gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')
        nonlocal gl

        if gl is not None:
            for artist_coll in [gl.xline_artists, gl.yline_artists, gl.xlabel_artists, gl.ylabel_artists]:
                [a.remove() for a in artist_coll]
            ax._gridliners = []

        gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')
        return abs(scale_to_zoom(scale))

    class Index:

        def __init__(self, index=None, mask_val=None):
            self.current_index = index
            self.mask_val = mask_val

        def update(self):
            try:
                arr = arrays[state['current_layer_name']][names[self.current_index]][::-1, :]
                im.set_array(arr)
                im.set_data(arr)
                im.set_clim(vmin=arr.min(), vmax=arr.max())
                im.autoscale()
                im.set_array(im.get_array())
                fig.suptitle(f'{names[self.current_index]}')
                fig.canvas.draw_idle()
            except KeyError:
                fig.canvas.flush_events()
                # arr = np.array([])
                # im.set_array(arr)
                # im.set_data(arr)
                # im.set_clim(vmin=arr.min(), vmax=arr.max())
                # im.autoscale()
                # im.set_array(im.get_array())
                # fig.suptitle(f'{names[self.current_index]}')
                fig.canvas.draw_idle()

        # Define functions to update the plot when the buttons are clicked
        def update_previous(self, event):

            if self.current_index > 0:
                self.current_index -= 1
                self.update()

        def update_next(self, event):
            #fig.savefig(f'/tmp/vis_cartopy_frame_{names[self.current_index]}.png', dpi=100)
            if self.current_index < len(names) - 1:

                self.current_index += 1
                self.update()

            # if self.current_index == len(names) -1:
            #     res = animate('/tmp/vis_cartopy_frame_*.png', outfile='/home/janf/Documents/ibm/hrea_block_anim.gif', delay=800, compression=10)

    def checkbox_update(label):
        index = layer_names.index(label)
        new_status = not status[index]
        status[index] = new_status  # Toggle the checkbox status
        if new_status is False:
            off = any(status)
            im.set_visible(off)
            if off is False:
                state['current_layer_name'] = None
            else:
                state['current_layer_name'] = layer_names[status.index(off)]
        else:
            state['current_layer_name'] = label
            if im.get_visible() is False:
                im.set_visible(new_status)

        callback.update()

    def set_bg_layer(label):

        style_name = label
        style = styles[style_name]

        ax.img_factories[0][0] = style
        ax._done_img_factory = False
        fig.canvas.draw()
        fig.canvas.flush_events()


    # Create two subplots for the buttons
    #fig.subplots_adjust(bottom=0.2, right=.7)
    #fig.subplots_adjust(bottom=0.2)

    ax_prev = plt.axes([0.4, 0.01, 0.1, 0.075])
    ax_next = plt.axes([0.5, 0.01, 0.1, 0.075])
    ax_check = plt.axes([0.6, 0.01, 0.2, 0.075], frameon=True)
    ax_bg = plt.axes([0.2, 0.01, 0.2, 0.075], frameon=True, )

    checkboxes = CheckButtons(ax_check, layer_names, status)
    checkboxes.on_clicked(checkbox_update)

    radio_bg = RadioButtons(ax_bg, style_names, active=style_names.index(style_name))
    radio_bg.on_clicked(set_bg_layer)
    # for circle in radio_bg.circles: # adjust radius here. The default is 0.05
    #     circle.set_radius(0.05)
    # Add the buttons to the subplots
    button_previous = Button(ax_prev, 'Previous')
    button_next = Button(ax_next, 'Next')
    # Connect the buttons to the update functions
    callback = Index(index=current_index, mask_val=mask_val)
    button_previous.on_clicked(callback.update_previous)
    button_next.on_clicked(callback.update_next)

    fig.suptitle(f'{names[current_index]}')

    ax.callbacks.connect('xlim_changed', on_zoom)
    # ax.coastlines(resolution='110m')
    # ax.add_feature(cartopy.feature.BORDERS, linestyle=':', linecolor='red')
    #ax.add_feature(cartopy.feature.LAKES.with_scale('110m'), zorder=10)
    if hasattr(cmap, 'labels'):
        labels, handles = zip(*[(k, mpatches.Rectangle((0, 0), 1, 1, facecolor=v)) for k, v in cmap.labels.items()])
        ax.legend(handles, labels, loc=4, framealpha=1)
    else:
        cb = plt.colorbar(im,ax=ax, orientation='horizontal', location='top', fraction=0.046, pad=0.04)
        #ax.set_aspect('auto', adjustable=None)
    plt.show()
