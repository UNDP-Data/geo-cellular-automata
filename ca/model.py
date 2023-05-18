
from skimage import util
import numpy as np
import itertools
from scipy.signal import fftconvolve
from affine import  Affine



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







def compute_diff(array=None):

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

def aggregate(binary_rec_array=None, block_size=None, th=.8):
    years = binary_rec_array.dtype.names

    ndtype = [(e, 'u2') for e in years]

    nl, nc = binary_rec_array.shape
    lmod = nl%block_size
    cmod = nc%block_size
    data = np.empty(shape=(nl//block_size, nc//block_size), dtype=ndtype)

    for year in years:
        ydata = binary_rec_array[year][0:nl-lmod, 0:nc-cmod]
        #bw = util.view_as_blocks(arr_in=np.where(ydata==-1,np.nan,ydata),block_shape=(block_size,block_size))
        bw = util.view_as_blocks(arr_in=np.where(ydata==-1,0,ydata),block_shape=(block_size,block_size))
        nozero = np.where(ydata==0,1,ydata)
        nozero = np.where(nozero==-1, 0, nozero)

        bw1 = util.view_as_blocks(arr_in=nozero,block_shape=(block_size,block_size))
        bsum_all = bw1.sum(axis=-1).sum(axis=-1)
        bsum_all = bsum_all / 4
        bsum = bw.sum(axis=-1).sum(axis=-1)
        bs = bsum/bsum_all

        data[year] = np.where(bs>th,1,0)
        #data[year] = bsum_all
        #data[year] = np.where(np.nanmean(np.nanmean(bw, axis=-1), axis=-1) > .8, 1, 0)
    return data

def aggregate1(binary_rec_array=None, block_size=None):
    years = binary_rec_array.dtype.names

    ndtype = [(e, 'u1') for e in years]

    nl, nc = binary_rec_array.shape
    lmod = nl%block_size
    cmod = nc%block_size
    data = np.empty(shape=(nl//block_size, nc//block_size), dtype=ndtype)

    for year in years:
        ydata = binary_rec_array[year][0:nl-lmod, 0:nc-cmod]
        bw = util.view_as_blocks(arr_in=np.where(ydata==-1,np.nan,ydata),block_shape=(block_size,block_size))
        #bw = util.view_as_blocks(arr_in=np.where(ydata==-1,0,ydata),block_shape=(block_size,block_size))
        #data[year] = np.where(bw.sum(axis=-1).sum(axis=-1)>=1,1,0)
        data[year] = np.where(np.nanmean(np.nanmean(bw, axis=-1), axis=-1) > .8, 1, 0)
    return data

def get_tranform_and_bounds(profile=None, array_shape=None ):
    src_transform = profile['transform']
    nl, nc = profile['height'], profile['width']
    yblock_size = nl // array_shape[0]
    xblock_size = nc // array_shape[1]

    tlx, xres, xrot, tly, yrot, yres  = src_transform.to_gdal()
    new_xres = xblock_size*xres
    new_yres = yblock_size*yres
    new_gt = tlx, new_xres, xrot, tly, yrot, new_yres
    new_transform = Affine.from_gdal(*new_gt)
    left, top = new_transform * [0,0]
    right, bottom = new_transform * array_shape[::-1]
    new_bounds = profile['bounds'].__class__(left, bottom, right, top)
    return new_transform, new_bounds


def compute_on_bsum(binary_rec_array=None,n=None):
    years = binary_rec_array.dtype.names
    year_pairs = dict(zip(years[1:], years[:-1]))
    dtype = binary_rec_array.dtype
    ndtype = [(e, 'i2') for e in years[1:]]
    data = np.empty(shape=binary_rec_array.shape, dtype=ndtype)
    offset = n//2
    nl, nc = binary_rec_array.shape

    lmod = nl%n
    cmod = nc%n
    for y0, y1 in year_pairs.items():
        t0 = binary_rec_array[y0][0:nl-lmod, 0:nc-cmod]

        bw = util.view_as_blocks(arr_in=t0,block_shape=(n,n))
        print(bw.shape)
        exit()
        t1 = binary_rec_array[y1]
        t0_win = np.lib.stride_tricks.sliding_window_view(t0,window_shape=(n, n))
        t1_win = np.lib.stride_tricks.sliding_window_view(t1,window_shape=(n, n))

        #t01_win = np.lib.stride_tricks.sliding_window_view(np.where(t0==1, 1, 0),window_shape=(ysize, xsize))
        t0_center = t0_win[...,offset,offset].ravel()
        t1_center = t1_win[...,offset,offset].ravel()
        t1_el_ind = np.argwhere(t1_center == 1).ravel()
        t0_noel_ind = np.argwhere(t0_center == 0).ravel()
        on_indices = np.intersect1d(t0_noel_ind, t1_el_ind)
        fftconvolve(np.where(t1==1, 1, 0), np.ones((n,n), dtype='u1'), mode='same')


def compute_bsum(rec_array=None,n=None):
    dtype = rec_array.dtype
    ndtype = [(e, 'i2') for e in dtype.names]
    data = np.empty(shape=rec_array.shape, dtype=ndtype)
    for name in dtype.names:
        a = rec_array[name]
        data[name] = fftconvolve(np.where(np.isnan(a), 0, a), np.ones((n,n), dtype='u1'), mode='same')
    return data

def compute_off_stats(binary_array=None, y0=None, y1=None, n=None):
    t0 = binary_array[y0]
    t1 = binary_array[y1]
    offset = n//2
    nl, nc = binary_array.shape
    ysize= nl-2*offset
    xsize = nc-2*offset
    t0_win = np.lib.stride_tricks.sliding_window_view(t0,window_shape=(ysize, xsize))
    t1_win = np.lib.stride_tricks.sliding_window_view(t1,window_shape=(ysize, xsize))
    t01_win = np.lib.stride_tricks.sliding_window_view(np.where(t1==1, 1, 0),window_shape=(ysize, xsize))
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
    print(array.shape, array.size, indices.size)
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
