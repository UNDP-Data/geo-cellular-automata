import logging
import os

import pandas
from scipy.spatial.distance import cdist
import numpy as np

from ca import io, model, S_A
from matplotlib import pyplot as plt
from matplotlib import gridspec


def display(data=dict(), interpolation='nearest', title='', cmap='gist_rainbow'):
    """
        Displays one or more arrays with legend.
        @args
            @da -  dictionary or ordered dictionary if one wants to preserve the order of arrays, ex {'name:'np.array(2D)}

    """
    fig = plt.figure()
    ncols = 2 # we want two columns
    nrows = int((len(data) / 2) + 1)

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols )

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
    #fig.show()
    plt.tight_layout()
    fig.suptitle(title)
    plt.show()

def display_rarray(data=None, ncols=3,  interpolation='nearest', title='', mask_val=None, cmap='gist_rainbow'):
    """
        Displays one or more arrays with legend.
        a

    """
    fig = plt.figure()
    #ncols = 2 # we want two columns
    nrows = int((len(data.dtype.names) / 2) + 1)


    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols,hspace=.5 )

    for i, iname in enumerate(data.dtype.names):

        a = data[iname]
        if mask_val:
            a = np.ma.masked_array(data=a, mask=a==mask_val)


        if i == 0:

            ax = fig.add_subplot(gs[i],)
            fax = ax
            ax.set_title(iname)
        else:
            ax = fig.add_subplot(gs[i], sharex=fax, sharey=fax, )
            #ax.set_xticklabels('')
            #ax.set_yticklabels('')
            ax.set_title(iname)

        im = ax.imshow(a, interpolation=interpolation, cmap=cmap)
        fig.subplots_adjust(top=0.8)
        cbar_ax = fig.add_axes([0.2, 0.9, 0.6, 0.04])
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        ax.set_aspect('equal')
    #fig.show()
    #plt.tight_layout()
    fig.suptitle(title)
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


        dist = distances(c,c)

        print(dist)


def plot_neigh_bars(array=None, n=9):
    y0s = array.dtype.names[:-1]
    y1s = array.dtype.names[1:]
    years = [f'{k}-{v}' for k, v in dict(zip(y0s, y1s)).items()]
    neighs = range(n)
    nneighs = n
    colors = plt.cm.Paired(np.linspace(0, 1, n-1))
    index = np.arange(len(years)) + 0.3
    cell_text = []
    bar_width = 0.4
    y_offset = np.zeros(nneighs)
    years_labels = []
    df = pandas.DataFrame(columns=['nneigh', 'perc', 'years'])
    ci = 0
    for y0, y1 in zip(y0s, y1s):
        neigh, percs, counts = model.compute_on_stats(array=array, y0=y0, y1=y1, n=9)

        percs5 = percs[:nneighs]
        counts5 = counts[:nneighs]
        row_name = f'{y0}-{y1}'
        years_labels.append(row_name)
        neigh5 = neigh[:nneighs]
        for i in range(nneighs):
            df.loc[len(df.index)] = [neigh5[i],percs5[i], row_name]

        #for ni in range(nneighs):
        #plt.bar(neigh5, counts5, bar_width, bottom=y_offset, color=colors[ci])
        #plt.bar( neighs, percs5, .5, label=row_name, bottom=y_offset, color=colors[ci])

        #plt.barh(neighs, percs5, .5, label=row_name,  color=colors[ci])
        ci+=1
        y_offset = y_offset + percs5
        cell_text.append([f'{e:.2f}%' for e in percs5])

    #df.reset_index()

    fig, ax = plt.subplots()

    df = df.set_index('years')
    a = df.groupby(["years","nneigh"]).sum().unstack()
    print(a)
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
    #plt.subplots_adjust(left=0.2, bottom=0.3)

    plt.ylabel(f"Percentage")
    plt.yticks()

   # plt.xticks([])
    #plt.xlabel(years)
    #plt.xlabel('No neighs')
    plt.title('The percentage of cells turned on grouped by number of neighbours')


    plt.show()

if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    root_folder  = '/data/hrea/kenya_lightscore/kisumu'
    hrea_array_path = os.path.join(root_folder, 'hrea.np')


    # data = io.read_xarray(src_folder=root_folder)
    # print( data)

    if not os.path.exists(hrea_array_path):
        hrea = io.read_all_data(src_folder=root_folder)
        np.save(hrea_array_path, hrea, allow_pickle=True)
    else:
        hrea = np.load(hrea_array_path)

    binary = model.apply_threhold(hrea)

    #plot_neigh_bars(array=binary, n=9)
    #plot_semivariogran(array=binary)
    model.compute_temp_autocorr(array=hrea)

    exit()



    # for i, year in enumerate(binary.dtype.names):
    #     a = binary[year]
    #     ma = a[a!=-1]
    #     el = ma[ma==1]
    #     no_el = ma[ma==0]
    #     perc_el = el.size/ma.size*100
    #     perc_unel = no_el.size/ma.size*100
    #     print(year, el.size, f'{perc_el:.2f}%', no_el.size, f'{perc_unel:.2f}%', ma.size)






    #deltas = model.compute_delta(hrea)
    neigh = model.compute_neighbours(rec_array=hrea, n=3)
    #print(neigh.shape, hrea.shape)
    display_rarray(data=neigh['2015'], title='neighs')
    #display_rarray(data=binary, title='test', mask_val=-1, cmap='viridis')
    abs_dif = hrea['2020']-hrea['2012']
    pos_abs_dif = np.where(abs_dif<0, np.nan, abs_dif)
    neg_abs_dif = np.where(abs_dif<0,  abs_dif, np.nan)

    data = {'2012':hrea['2012'], '2020':hrea['2020']}
    data.update(dict(
        diff=abs_dif,
        pos_diff=pos_abs_dif,
        neg_diff=neg_abs_dif
    ))

    #display(data=data,cmap='coolwarm')
