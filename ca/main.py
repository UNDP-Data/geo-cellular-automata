import json
import logging
import os
from urllib.request import urlopen, Request
from typing import List, Dict
import rasterio.coords
from PIL import Image
import pandas
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib
from ca import io as caio, model
from matplotlib import pyplot as plt, cm
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import pickle
import cartopy.io.img_tiles as cimgt
import ca
from matplotlib.widgets import CheckButtons, Button
import cartopy
import math
import matplotlib.patches as mpatches
from cartopy import mpl as cmpl

def display(data=dict(), interpolation='nearest', title='', cmap='Paired'):
    """
        Displays one or more arrays with legend.
        @args
            @da -  dictionary or ordered dictionary if one wants to preserve the order of arrays, ex {'name:'np.array(2D)}

    """
    fig = plt.figure()
    ncols = 2  # we want two columns
    nrows = int((len(data) / 2) + 1)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for i, v in enumerate(data.items()):
        iname, a = v

        if i == 0:

            ax = fig.add_subplot(gs[i])
            fax = ax
            ax.set_title(iname)
        else:
            ax = fig.add_subplot(gs[i], sharex=fax, sharey=fax)
            ax.set_title(iname)

        im = ax.imshow(a, interpolation=interpolation, cmap=cmap)
        plt.colorbar(im, use_gridspec=True, orientation='vertical')
        ax.set_aspect('equal')
    # fig.show()
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()


def display_rarray(data=None, ncols=3, interpolation='nearest', title='', mask_val=None, cmap='gist_rainbow'):
    """
        Displays one or more arrays with legend.
        a

    """

    fig = plt.figure()
    # ncols = 2 # we want two columns
    nrows = int((len(data.dtype.names) / 2) + 1)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, hspace=.5)

    for i, iname in enumerate(data.dtype.names):

        a = data[iname]
        if mask_val:
            a = np.ma.masked_array(data=a, mask=a == mask_val)

        if i == 0:

            ax = fig.add_subplot(gs[i], )
            fax = ax
            ax.set_title(iname)
        else:
            ax = fig.add_subplot(gs[i], sharex=fax, sharey=fax, )
            # ax.set_xticklabels('')
            # ax.set_yticklabels('')
            ax.set_title(iname)

        im = ax.imshow(a, interpolation=interpolation, cmap=cmap)
        fig.subplots_adjust(top=0.8)
        cbar_ax = fig.add_axes([0.2, 0.9, 0.6, 0.04])
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        ax.set_aspect('equal')
    # fig.show()
    # plt.tight_layout()
    fig.suptitle(title)
    plt.show()

# def geo_plot(arrays=None, style='map', interpolation='nearest',
#              title='', mask_val=None, cmap='gist_rainbow',
#              arrays_bounds: List[rasterio.coords.BoundingBox] = None,
#
#
#              ):
#     current_index = 0
#     layer_names = list(arrays.keys())
#     current_layer_name = layer_names[0]
#     status = [False]
#     status[0] = True
#
#
#
#
#     exit()
#     def get_fig_size_cm(ax):
#         bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         width, height = bbox.width, bbox.height
#         width *= 2.54
#         height *= 2.54
#         return int(round(width)), int(round(height))
#
#     def scale_to_zoom(scale, tile_size=256, platform='google'):
#         if platform == 'google':
#             C = 256
#         else:
#             raise ValueError('Unsupported platform: {}'.format(platform))
#         zoom_level = math.log2(C * scale / (tile_size * 40075017))
#         return 21 if int(zoom_level) > 21 else int(zoom_level)
#
#     def get_zoom_level(extent, ax):
#         """
#         Computes the zoom level equivalent of a map extent in the Google Web Mercator projection.
#
#         Arguments:
#         extent -- A tuple containing the extent of the map in Web Mercator units (xmin, ymin, xmax, ymax).
#
#         Returns:
#         The zoom level equivalent of the map extent.
#         """
#         # Extract the width and height from the extent tuple
#         xmin, ymin, xmax, ymax = extent
#         px_width, px_height = get_fig_size_cm(ax)
#
#         map_width = xmax - xmin
#
#         scale = map_width / px_width
#
#         return abs(scale_to_zoom(scale))
#
#     class Index:
#
#         def __init__(self, index=None, mask_val=None):
#             self.current_index = index
#             self.mask_val = mask_val
#
#         # Define functions to update the plot when the buttons are clicked
#         def update_previous(self, event):
#
#             if self.current_index > 0:
#                 self.current_index -= 1
#
#                 arr = array1[names[self.current_index]][::-1,:]
#                 im.set_array(arr)
#                 im.set_data(arr)
#                 im.set_clim(vmin=arr.min(), vmax=arr.max())
#                 im.autoscale()
#                 im.set_array(im.get_array())
#                 fig.suptitle(f'{names[self.current_index]}')
#                 fig.canvas.draw_idle()
#
#         def update_next(self, event):
#
#             if self.current_index < data.shape[2] - 1:
#                 self.current_index += 1
#
#                 arr = array1[names[self.current_index]][::-1,:]
#                 im.set_array(arr)
#                 im.set_data(arr)
#                 im.set_clim(vmin=arr.min(), vmax=arr.max())
#                 im.autoscale()
#                 im.set_array(im.get_array())
#
#                 fig.suptitle(f'{names[self.current_index]}')
#                 fig.canvas.draw_idle()
#
#     dt = array1.dtype[0]
#
#     names = array1.dtype.names
#
#     # convert struct array to regular
#     data = array1.view((dt, len(array1.dtype)))
#
#
#     if style == 'map':
#         img = cimgt.OSM()  # spoofed, downloaded street map
#     elif style == 'satellite':
#         # SATELLITE STYLE
#         img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
#     else:
#         raise Exception('no valid style')
#     fig = plt.figure(figsize=(10, 10))  # open matplotlib figure
#
#     ax = plt.axes(projection=ccrs.GOOGLE_MERCATOR)  # project using coordinate reference system (CRS) of street map
#
#     map_extent = [map_bounds.left, map_bounds.right, map_bounds.bottom, map_bounds.top]
#     extent = [array_bounds.left, array_bounds.right, array_bounds.bottom, array_bounds.top]
#
#     ax.set_extent(map_extent)  # set extents
#     ax.add_image(img, 12, zorder=2)  # add OSM with zoom specification
#
#     # [top_left[0], bot_right[0], bot_right[1], top_left[1]]
#
#     im = ax.imshow(array1['2012'],
#                    transform=ccrs.PlateCarree(),
#                    extent=extent,
#                    cmap=cmap,
#                    interpolation=interpolation,
#                    zorder=5,
#                    origin='upper'
#
#                    )
#
#     # Declare and register callbacks
#     def on_zoom(event_ax):
#         xmin, xmax = event_ax.get_xlim()
#         ymin, ymax = event_ax.get_ylim()
#         zl = get_zoom_level(extent=[xmin, ymin, xmax, ymax], ax=event_ax)
#         if zl < 20:
#             event_ax.img_factories[0][1] = (zl,)
#             event_ax._done_img_factory = False
#
#     # Create two subplots for the buttons
#     fig.subplots_adjust(bottom=0.2, right=.7)
#     ax_prev = plt.axes([0.45, 0.05, 0.1, 0.075])
#     ax_next = plt.axes([0.55, 0.05, 0.1, 0.075])
#
#     # Add the buttons to the subplots
#     button_previous = Button(ax_prev, 'Previous')
#     button_next = Button(ax_next, 'Next')
#     # Connect the buttons to the update functions
#     callback = Index(index=current_index, mask_val=mask_val)
#     button_previous.on_clicked(callback.update_previous)
#     button_next.on_clicked(callback.update_next)
#
#     fig.suptitle(f'{names[current_index]}')
#
#     ax.callbacks.connect('xlim_changed', on_zoom)
#     # ax.coastlines(resolution='110m')
#     ax.add_feature(cartopy.feature.BORDERS, linestyle=':', linecolor='red')
#     # ax.add_feature(cartopy.feature.LAKES.with_scale('110m'))
#
#     # xticks = np.arange(xmin, xmax, 10000)
#     # yticks = np.arange(ymin, ymax, 10000)
#     # ax.set_xticks(xticks, crs=ccrs.GOOGLE_MERCATOR)
#     # ax.set_yticks(yticks, crs=ccrs.GOOGLE_MERCATOR)
#     gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')
#     fig.subplots_adjust(bottom=0.2)
#
#     # plt.colorbar(im, ax=ax, orientation='horizontal')
#     labels, handles = zip(*[(k, mpatches.Rectangle((0, 0), 1, 1, facecolor=v)) for k, v in cmp.labels.items()])
#     ax.legend(handles, labels, loc=4, framealpha=1)
#
#     plt.show()

def geo_plot(arrays=None, style='map', interpolation='nearest',
             title='', mask_val=None, cmap='gist_rainbow',
             arrays_bounds: Dict[str,rasterio.coords.BoundingBox] = None,


             ):
    current_index = 0
    layer_names = tuple(arrays.keys())
    print(layer_names)
    state = {'current_layer_name':layer_names[0]}
    #current_layer_name = layer_names[0]
    #status = dict([(e, False) for e in layer_names])
    status = [False for e in layer_names]
    status[0] = True
    current_bounds = arrays_bounds[state['current_layer_name']]
    #dts = dict([(k, v.dtype[0])  for k, v in arrays.items()])
    names = arrays[state['current_layer_name']].dtype.names
    #data = dict([(k, v.view((dts[k], len(v.dtype)))) for k, v in arrays.items()])


    def hinit():
        current_bounds = arrays_bounds[state['current_layer_name']]
        current_layer_extent = [current_bounds.left, current_bounds.right, current_bounds.bottom, current_bounds.top]
        #ax.set_extent(current_layer_extent)  # set extents
        callback.update()



    gl = None



    if style == 'map':
        img = cimgt.OSM()  # spoofed, downloaded street map
    elif style == 'satellite':
        # SATELLITE STYLE
        img = cimgt.QuadtreeTiles()  # spoofed, downloaded street map
    else:
        raise Exception('no valid style')
    fig = plt.figure(figsize=(10, 10))  # open matplotlib figure

    ax = plt.axes(projection=ccrs.GOOGLE_MERCATOR)  # project using coordinate reference system (CRS) of street map

    #map_extent = [map_bounds.left, map_bounds.right, map_bounds.bottom, map_bounds.top]


    current_layer_extent = [current_bounds.left, current_bounds.right, current_bounds.bottom, current_bounds.top]
    ax.set_extent(current_layer_extent)  # set extents
    ax.add_image(img, 12, zorder=2)  # add OSM with zoom specification

    # [top_left[0], bot_right[0], bot_right[1], top_left[1]]
    arr = arrays[state['current_layer_name']][names[current_index]]

    im = ax.imshow(arr,
                   transform=ccrs.PlateCarree(),
                   extent=current_layer_extent,
                   cmap=cmap,
                   interpolation=interpolation,
                   zorder=5,
                   origin='upper'

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
        #gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')
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
                arr = arrays[state['current_layer_name']][names[self.current_index]][::-1,:]
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

            if self.current_index < len(names) - 1:
                self.current_index += 1
                self.update()


    def checkbox_update(label):

        index = layer_names.index(label)
        new_status = not status[index]
        status[index] = new_status  # Toggle the checkbox status
        print(status)
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

        # print(f"Changed layer name to {state['current_layer_name']} ")
        callback.update()

    # Create two subplots for the buttons
    fig.subplots_adjust(bottom=0.2, right=.7)
    ax_prev = plt.axes([0.45, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.55, 0.05, 0.1, 0.075])
    ax_check = plt.axes([0.75, 0.8, 0.1, 0.075], frameon=True)
    checkboxes = CheckButtons(ax_check, layer_names, status)
    checkboxes.on_clicked(checkbox_update)
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
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':', linecolor='red')
    # ax.add_feature(cartopy.feature.LAKES.with_scale('110m'))

    # xticks = np.arange(xmin, xmax, 10000)
    # yticks = np.arange(ymin, ymax, 10000)
    # ax.set_xticks(xticks, crs=ccrs.GOOGLE_MERCATOR)
    # ax.set_yticks(yticks, crs=ccrs.GOOGLE_MERCATOR)
    # gl = ax.gridlines(draw_labels=True, linewidth=1, color='red', linestyle='-')
    fig.subplots_adjust(bottom=0.2)

    # plt.colorbar(im, ax=ax, orientation='horizontal')
    labels, handles = zip(*[(k, mpatches.Rectangle((0, 0), 1, 1, facecolor=v)) for k, v in cmp.labels.items()])
    ax.legend(handles, labels, loc=4, framealpha=1)

    plt.show()


def ploti(rec_array=None, interpolation='nearest', title='', mask_val=None, cmap='gist_rainbow', meta=None):
    dt = rec_array.dtype[0]

    names = rec_array.dtype.names

    # convert struct array to regular
    data = rec_array.view((dt, len(rec_array.dtype)))

    current_index = 0

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the first time step
    arr = rec_array[names[current_index]]
    col_dict = {-1: "white",
                0: "black",
                1: "orange"
                }

    # We create a colormar from our list of colors
    cmp = ListedColormap([col_dict[x] for x in col_dict.keys()])
    # cmp = cm.get_cmap(cmap)

    im = ax.imshow(arr, cmap=cmp, interpolation=interpolation, origin='upper')

    # im = ax.imshow(arr, cmap=cmap, interpolation=interpolation)

    # # Create previous and next buttons
    # button_previous = Button(ax, 'Previous')
    # button_next = Button(ax, 'Next')

    class Index:

        def __init__(self, index=None, mask_val=None):
            self.current_index = index
            self.mask_val = mask_val

        # Define functions to update the plot when the buttons are clicked
        def update_previous(self, event):

            if self.current_index > 0:
                self.current_index -= 1

                arr = rec_array[names[self.current_index]]
                im.set_array(arr)
                im.set_data(arr)
                im.set_clim(vmin=arr.min(), vmax=arr.max())
                im.autoscale()
                im.set_array(im.get_array())
                fig.suptitle(f'{names[self.current_index]}')
                fig.canvas.draw_idle()

        def update_next(self, event):

            if self.current_index < data.shape[2] - 1:
                self.current_index += 1

                arr = rec_array[names[self.current_index]]
                im.set_array(arr)
                im.set_data(arr)
                im.set_clim(vmin=arr.min(), vmax=arr.max())
                im.autoscale()
                im.set_array(im.get_array())
                fig.suptitle(f'{names[self.current_index]}')
                fig.canvas.draw_idle()

    # Create two subplots for the buttons
    fig.subplots_adjust(bottom=0.2)
    ax_prev = plt.axes([0.45, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.55, 0.05, 0.1, 0.075])

    # Add the buttons to the subplots
    button_previous = Button(ax_prev, 'Previous')
    button_next = Button(ax_next, 'Next')
    # Connect the buttons to the update functions
    callback = Index(index=current_index, mask_val=mask_val)
    button_previous.on_clicked(callback.update_previous)
    button_next.on_clicked(callback.update_next)

    fig.suptitle(f'{names[current_index]}')
    # Show the plot
    # plt.colorbar(im, cax=ax, orientation='horizontal')
    plt.show()


def distances(xy1, xy2):
    """
        Computes the pairwise distance between the elements (x,y) coordinates in the arrays
        @args
            @xy1, numpy.array, 2D, ex np.array([[0,0]])
            @xy2, numpy array, 2D, ex. np.argwhere(np.ones(5,5))
        @returns
            a 1D array representing the pairwise distances fir each pair, that is for each element in xy1 an array of distances to each element
            of xy2. NB, an element consist of a pair of coordinates
    """
    d0 = np.subtract.outer(xy1[:, 0], xy2[:, 0])
    d1 = np.subtract.outer(xy1[:, 1], xy2[:, 1])
    dist = np.hypot(d0, d1)

    return dist


def plot_semivariogran(array=None):
    years = array.dtype.names
    for year in years:
        arr = array[year]
        mindtp = np.min_scalar_type(max(arr.shape))

        c = np.argwhere(arr == 1).astype(mindtp)

        dist = distances(c, c)

        print(dist)


def plot_neigh_bars(hrea_array=None, binary_hrea_array=None, ):
    y0s = binary_hrea_array.dtype.names[:-1]
    y1s = binary_hrea_array.dtype.names[1:]
    years = [f'{k}-{v}' for k, v in dict(zip(y0s, y1s)).items()]
    nneighs = 9
    colors = plt.cm.Paired(np.linspace(0, 1, nneighs - 1))
    index = np.arange(len(years)) + 0.3
    cell_text = []
    bar_width = 0.4
    y_offset = np.zeros(nneighs)
    print(y_offset.size)
    years_labels = []
    df = pandas.DataFrame(columns=['nneigh', 'perc', 'years'])
    ci = 0
    for y0, y1 in zip(y0s, y1s):
        neigh, percs, counts = model.compute_on_stats(binary_array=binary_hrea_array, y0=y0, y1=y1, n=3)

        print(y0, y1, neigh, percs)

        row_name = f'{y0}-{y1}'
        years_labels.append(row_name)

        for n in range(neigh.size):
            # df.loc[len(df.index)] = [neigh5[i],percs5[i], row_name]
            df.loc[len(df.index)] = [neigh[n], percs[n], row_name]

        # for ni in range(nneighs):
        # plt.bar(neigh5, counts5, bar_width, bottom=y_offset, color=colors[ci])
        # plt.bar( neighs, percs5, .5, label=row_name, bottom=y_offset, color=colors[ci])

        # plt.barh(neighs, percs5, .5, label=row_name,  color=colors[ci])
        ci += 1
        # y_offset = y_offset[neigh] + percs[neigh]
        # cell_text.append([f'{e:.2f}%' for e in percs])

    # df.reset_index()

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
    labels_new = [f"{label.strip('()').split(',')[1]} neigh" for label in labels]
    plt.legend(handles, labels_new)

    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    # the_table = plt.table(cellText=cell_text,
    #                       rowLabels=years_labels,
    #                       rowColours=colors,
    #                       colLabels=neighs,
    #                       loc='bottom')

    # Adjust layout to make room for the table:
    # plt.subplots_adjust(left=0.2, bottom=0.3)

    plt.ylabel(f"Percentage")
    plt.yticks()

    # plt.xticks([])
    # plt.xlabel(years)
    # plt.xlabel('No neighs')
    plt.title('The percentage of block turned on grouped by number of neighbours in previous time step')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    root_folder = '/data/hrea/kenya_lightscore/kisumu'
    hrea_array_path = os.path.join(root_folder, 'hrea.npy')
    hrea_meta_path = os.path.join(root_folder, 'hrea.meta')

    # data = io.read_xarray(src_folder=root_folder)
    # print( data)

    if not os.path.exists(hrea_array_path):
        hrea, profile = caio.read_rio(src_folder=root_folder)
        np.save(hrea_array_path, hrea, allow_pickle=True)
        with open(hrea_meta_path, 'wb') as dst:
            p = pickle.Pickler(file=dst)
            p.dump(profile)


    else:
        #os.remove(hrea_array_path)
        print('reading')
        hrea = np.load(hrea_array_path)

        profile = None
        with open(hrea_meta_path, 'rb') as src:
            p = pickle.Unpickler(file=src)
            profile = p.load()

    binary = model.apply_threhold(hrea)

    # ploti(rec_array=binary, mask_val=-1, cmap='gray_r')
    onekm_agg = model.aggregate(binary_rec_array=binary, block_size=29)
    transform, bounds = model.get_tranform_and_bounds(profile=profile, array_shape=onekm_agg.shape)

    # plot_neigh_bars(hrea_array=hrea, binary_hrea_array=onekm_agg,)
    # plot_semivariogran(array=binary)
    # model.compute_temp_autocorr(array=hrea)
    data = model.compute_bsum(rec_array=hrea, n=99)

    col_dict = {
                -1: "#FFFFFF00",
                #-1: "red",
                0: "black",
                1: "orange"
                }

    #ploti(rec_array=onekm_agg)
    # We create a colormar from our list of colors
    cmp = ListedColormap([col_dict[x] for x in col_dict.keys()])
    cmp.labels = {'no electricity': 'black', 'electrified': 'orange'}
    arrays={'binary':binary, 'block':onekm_agg}
    arrays_bounds = {'binary':profile['bounds'], 'block':bounds}
    geo_plot(arrays=arrays, arrays_bounds=arrays_bounds, cmap=cmp, style='satellite')


    def mark():
        pass


    exit()
