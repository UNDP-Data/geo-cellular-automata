import matplotlib.pyplot as plt
import numpy as np
import functools
import itertools
import operator
import pandas as pd
from scipy.signal import correlate2d, fftconvolve

def gen_offsets(n=None):
    assert n%2==1, f'n={n} is not odd'
    middle = n//2
    indices = (e-middle for e in range(n))
    return itertools.product(indices, repeat=2)

def gen_indices(n=None):

    assert n%2==1, f'n={n} is not odd'
    indices = range(n)
    return itertools.product(indices, repeat=2)





def neighbour_name_from_offset(axis0_offset=None, axis1_offset=None):

    vertical_names = 'bottom', 'center', 'top'
    horizontal_names = 'right', 'center', 'left'
    indices = range(3)
    rel_indices = [e-1 for e in indices]
    vertical_dict = dict(zip(rel_indices, vertical_names))
    horizontal_dict = dict(zip(rel_indices, horizontal_names))
    vert_name = vertical_dict[axis0_offset]
    hor_name = horizontal_dict[axis1_offset]
    if hor_name == vert_name:
        return hor_name
    return f'{vert_name}_{hor_name}'

def compute_neighbours_dtype(array=None, n=3):

    offsets = list(gen_offsets(n=n))
    neigh_names = [neighbour_name_from_offset(axis0_offset=e[0], axis1_offset=e[1]) for e in offsets]

    return [(n, array.dtype) for n in neigh_names]

def compute_neighbours(rec_array=None, n=3):
    dt = []
    for name in rec_array.dtype.names:
        a = rec_array[name]
        dt.append((name, compute_neighbours_dtype(array=a, n=n)))
    data = None
    for name in rec_array.dtype.names:
        a = rec_array[name]
        if data is None:
            na = compute_neighbours_for_array(array=a, n=n)
            data = np.empty(shape=na.shape, dtype=dt)
            data[name] = na
        else:
            data[name] = compute_neighbours_for_array(array=a, n=n)
    return data

def apply_threhold(rec_array=None):
    dtype = rec_array.dtype
    ndtype = [(e, 'i1') for e in dtype.names]
    data = np.empty(shape=rec_array.shape, dtype=ndtype)
    for name in dtype.names:
        a = rec_array[name]
        valid = ~np.isnan(a)
        data[name][~valid] = -1
        data[name][valid] = np.where(rec_array[name][valid]>.8, 1, 0)
    return data







def compute_neighbours_for_array(array=None, n=3):

    offset = n//2
    offsets = list(gen_offsets(n=n))
    # neigh_names = [neighbour_name_from_offset(axis0_offset=e[0], axis1_offset=e[1]) for e in offsets]
    indices = list(gen_indices(n=n))
    data = None
    nl, nc = array.shape
    ysize= nl-2*offset
    xsize = nc-2*offset
    raw_data = np.lib.stride_tricks.sliding_window_view(array,window_shape=(ysize, xsize))

    dt = compute_neighbours_dtype(array=array, n=n)
    for i, j in indices:
        neigh_name = neighbour_name_from_offset(axis0_offset=i-offset, axis1_offset=j-offset)
        if data is None:
            data = np.empty(shape=(ysize, xsize), dtype=dt)
        data[neigh_name] = raw_data[i, j]
    return data







def compute_delta(array=None):

    years = array.dtype.names
    delta_dtype = []
    data = None

    for i in range(len(years)-1):
        n = f'{years[i+1]}-{years[i]}'
        t = 'f4'
        delta_dtype.append((n,t))
    a = np.zeros(shape=array.shape, dtype=delta_dtype)

    for n, dt in a.dtype.fields.items():
        sy, ey = n.split('-')
        #if data = None:

        a[n] = array[sy]-array[ey]
    return a



def compute_on_stats(binary_array=None, y0=None, y1=None, n=None):
    t0 = binary_array[y0]
    t1 = binary_array[y1]
    offset = n//2
    nl, nc = binary_array.shape
    ysize= nl-2*offset
    xsize = nc-2*offset
    t0_win = np.lib.stride_tricks.sliding_window_view(t0,window_shape=(ysize, xsize))
    t1_win = np.lib.stride_tricks.sliding_window_view(t1,window_shape=(ysize, xsize))
    t01_win = np.lib.stride_tricks.sliding_window_view(np.where(t0==1, 1, 0),window_shape=(ysize, xsize))
    t0_center = t0_win[offset,offset,...].ravel()
    t1_center = t1_win[offset,offset,...].ravel()
    t1_el_ind = np.argwhere(t1_center == 1).ravel()
    t0_noel_ind = np.argwhere(t0_center == 0).ravel()
    on_indices = np.intersect1d(t0_noel_ind, t1_el_ind)


    a = t01_win.sum(axis=0).sum(axis=0)
    b = a.flat[on_indices]

    uv, counts = np.unique(b, return_counts=True)

    percs = counts/sum(counts) *100

    # percs_str = [f'{e:.2f}%' for e in percs]
    # nn_dict = dict(zip(uv, percs_str))
    #print(f'cells (0->1) from {y0}->{y1} {nn_dictm} :: {sum(countsm) } (total)')
    return uv, percs, counts


def compute_yearly_on_stats(array=None, y0=None, y1=None, n=3):
    t0 = array[y0]
    t1 = array[y1]
    t0_neigh = compute_neighbours_for_array(array=t0, n=n)
    t1_neigh = compute_neighbours_for_array(array=t1, n=n)
    t1_el_ind = np.argwhere(t1_neigh['center'].ravel() == 1).ravel()
    t0_noel_ind = np.argwhere(t0_neigh['center'].ravel() == 0).ravel()

    on_flat = np.intersect1d(t0_noel_ind, t1_el_ind)
    on_indices = np.dstack(np.unravel_index(on_flat,shape=t1_neigh.shape)).squeeze()
    l = []
    for pos in t0_neigh.dtype.names:
        arr = t0_neigh[pos]
        l.append((arr.flat[on_flat] == 1).view('u1'))
    n = np.dstack(l).squeeze()
    nn = n.sum(axis=1)
    uv, counts = np.unique(nn, return_counts=True)
    percs = counts/sum(counts) *100
    percs_str = [f'{e:.2f}%' for e in percs]
    nn_dict = dict(zip(uv, percs_str))
    print(f'cells (0->1) from {y0}->{y1} {nn_dict} :: {sum(counts) } (total)')


def compute_temp_autocorr(array=None):
    years = array.dtype.names
    indices = None
    for year in years:
        yarr = array[year]
        year_el_ind = np.argwhere(~np.isnan(yarr.ravel())).ravel()
        if  indices is None:
            indices = year_el_ind
        else:
            indices = np.intersect1d(indices, year_el_ind)

    ts = array.flat[indices]
    print(ts[:5])

def compute_spatial_autocorr(array=None):
    years = array.dtype.names
    corr = {}
    y0s = years[:-1]
    y1s = years[1:]
    for y0, y1 in zip(y0s, y1s):
        y0arr = array[y0]
        y1arr = array[y1]
        #Compute the spatial correlation using FFT
        fft_arr1 = np.fft.fft2(y0arr)
        fft_arr2_conj = np.conj(np.fft.fft2(y1arr))
        product = np.multiply(fft_arr1, fft_arr2_conj)
        corr = np.fft.ifft2(product)
        # Normalize the correlation to get the correlation coefficient
        corr_coef = corr / (y0arr.std() * y1arr.std() * y1arr.size)
        print(y0, y1, corr_coef.mean())







if __name__ == '__main__':
    #print(list(gen_indices(n=5, relative_to_center=False)))
    pass
