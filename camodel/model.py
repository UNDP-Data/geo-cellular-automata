import numpy as np
import logging
from skimage import util as skutil
from scipy.signal import fftconvolve
from affine import Affine

logger = logging.getLogger(__name__)


def apply_threhold(hrea_struct_array=None, threshold=.8, nan_value=-1) -> np.ndarray:
    """
    Convert float HREA electrification likelihood in range [0-1] to
    signed 1 byte form 0-> not electrified, 1-> electrified.
    The nans are set to nan_value
    :param hrea_struct_array: np structured array
    :return: binary hrea np array
    """
    dtype = hrea_struct_array.dtype
    ndtype = [(e, 'i1') for e in dtype.names]
    data = np.empty(shape=hrea_struct_array.shape, dtype=ndtype)
    for year in dtype.names:
        a = hrea_struct_array[year]
        valid = ~np.isnan(a)
        data[year][~valid] = nan_value
        data[year][valid] = np.where(hrea_struct_array[year][valid] > threshold, 1, 0)
    return data



def get_tranform_and_bounds(profile=None, array_shape=None ):
    """
    Compute the transform and bounds for an array with shape array_shape
    based on the rasterio profile arg
    :param profile: dict representing the rasterio profile
    :param array_shape: the new shape for which the profile will be computed
    :return: tuple (Affine, bounds)
    """
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




def aggregate(binary_hrea_array=None, block_size=None, threshold=.8, profile=None):
    """
    Aggregate binary_hrea_array into square blocks where the size of the new block or element in the output array
    across one dimension
    :param binary_hrea_array: np.ndarray, input binary hrea array
    :param block_size: int, specifies how many number of elements from binary_hrea_array will be aggregated across
            one dimension
    :param threshold: float, a number that controls whenet a new aggregated element will be
     set to 1 or 0. The percentage of settlements or elements in the original input array that fall into one block or new element
    :param profile, a dict representing the ratserio profile for the binary_hrea_array
    :return:
    """
    years = binary_hrea_array.dtype.names

    ndtype = [(e, 'u2') for e in years]

    nl, nc = binary_hrea_array.shape
    lmod = nl%block_size
    cmod = nc%block_size
    data = np.empty(shape=(nl//block_size, nc//block_size), dtype=ndtype)

    for year in years:
        ydata = binary_hrea_array[year][0:nl - lmod, 0:nc - cmod]
        #bw = util.view_as_blocks(arr_in=np.where(ydata==-1,np.nan,ydata),block_shape=(block_size,block_size))
        bw = skutil.view_as_blocks(arr_in=np.where(ydata==-1,0,ydata),block_shape=(block_size,block_size))
        nozero = np.where(ydata==0,1,ydata)
        nozero = np.where(nozero==-1, 0, nozero)

        bw1 = skutil.view_as_blocks(arr_in=nozero,block_shape=(block_size,block_size))
        bsum_all = bw1.sum(axis=-1).sum(axis=-1)
        bsum_all = bsum_all / 4
        bsum = bw.sum(axis=-1).sum(axis=-1)
        bs = bsum/bsum_all

        data[year] = np.where(bs>threshold,1,0)
        #data[year] = bsum_all
        #data[year] = np.where(np.nanmean(np.nanmean(bw, axis=-1), axis=-1) > .8, 1, 0)

    transform, bounds = get_tranform_and_bounds(profile=profile, array_shape=data.shape)
    return data, transform, bounds

def compute_neighbour_stats(binary_array=None, y0=None, y1=None,  target_year=None):
    """
    Given a  multiyear HREA dataset compute the neighbours stats (Moore neighbourhood)  for the target_year for
    all the elements/settlements that have been turned on (0->1) between year y0 and y1.
    The target_year has to be on of the y0 or y1
    COmpute the number of neighbors at target_year for all elements in the binary array representing settlements
    thathave been tuned
    :param binary_array:the HREA binary/thresholded array
    :param y0: start year
    :param y1: end year
    :return: tuple
                first ele = the unique nuber of neighbors (0-8)
                second ele = the percentage of each no of neigh. aggregated for all on indices
                third ele = the counts of each no of neigh. aggregated for all on indices
    """
    assert target_year in [y0, y1], f'Invalid target_year={target_year}. valid values are {[y0, y1]}'
    # array for year 0
    t0 = binary_array[y0]
    #array for year 1
    t1 = binary_array[y1]
    #target year array
    ttarget = t0 if target_year == y0 else t1

    # the sliding window function is going to be used to create a 4dim array. Persormin operations
    #on this arrays on specific axes is equal with performing the same operations over block of size
    # block_size == 3 (simple neighbourhood
    block_size = 3 #Moore neigh
    offset = block_size//2
    nl, nc = binary_array.shape
    #compute the window size
    ysize= nl-2*offset
    xsize = nc-2*offset
    #use stride tricks for max efficiency
    t0_win = np.lib.stride_tricks.sliding_window_view(t0,window_shape=(ysize, xsize))
    t1_win = np.lib.stride_tricks.sliding_window_view(t1,window_shape=(ysize, xsize))

    # create an array for tharegt_year where the settlemets are on only (get rid of -1 or empty elements)
    t01_win = np.lib.stride_tricks.sliding_window_view(np.where(ttarget==1, 1, 0),window_shape=(ysize, xsize))
    #extract center and take a flat view
    t0_center = t0_win[offset,offset,...].ravel()
    t1_center = t1_win[offset,offset,...].ravel()
    t1_el_ind = np.argwhere(t1_center == 1).ravel()
    t0_noel_ind = np.argwhere(t0_center == 0).ravel()
    # compute the indices of settlements that have been electrified between y0 and y1
    on_indices = np.intersect1d(t0_noel_ind, t1_el_ind)
    #sum over the first 2 axis === sum over blocks === get the number of neighbours inside the block
    on_sum = t01_win.sum(axis=0).sum(axis=0)
    #flatten  the on_sum and read only the on indices to get exactly the pixels that have been turned on
    on_sum_flat = on_sum.flat[on_indices]
    #get the uniqye values including the counts, This will calculate the number of neighbours

    uv, counts = np.unique(on_sum_flat, return_counts=True)
    #compute the percentage
    percs = counts/sum(counts) *100
    return uv-1, percs, counts


def focal_mean(binary_hrea_array=None, window_size=None):
    """
    Fast Fourier based focal mean
    :param binary_hrea_array:
    :param window_size:
    :return:
    """
    dtype = binary_hrea_array.dtype
    ndtype = [(e, 'f4') for e in dtype.names]
    data = np.empty(shape=binary_hrea_array.shape, dtype=ndtype)
    for name in dtype.names:
        a = binary_hrea_array[name]
        n = fftconvolve(np.where(np.isnan(a), 0, 1), np.ones((window_size, window_size), dtype='u1'), mode='same')
        data[name] = fftconvolve(np.where(np.isnan(a), 0, a), np.ones((window_size, window_size), dtype='u1'), mode='same') / n

    return data


def focal_sum(binary_hrea_array=None, window_size=None):
    """
    Fast Fourier based  focal sum
    :param binary_hrea_array:
    :param window_size:
    :return:
    """
    dtype = binary_hrea_array.dtype
    ndtype = [(e, 'f2') for e in dtype.names]
    data = np.empty(shape=binary_hrea_array.shape, dtype=ndtype)
    for name in dtype.names:
        a = binary_hrea_array[name]
        data[name] = fftconvolve(np.where(np.isnan(a), 0, a), np.ones((window_size, window_size), dtype='u1'), mode='same')

    return data



def block_mean(array_in=None, block_size=None, profile=None ):


    print(array_in.shape, np.array(array_in.shape)%block_size)
    nl, nc = array_in.shape
    lmod = nl%block_size
    cmod = nc%block_size
    nl-=lmod
    nc-=cmod
    a = array_in[:nl, :nc]
    print(nl, nc, a.shape, np.array(a.shape)%block_size)
    n = skutil.view_as_blocks(arr_in=np.where(np.isnan(a), 0, 1),block_shape=(block_size,block_size))
    bw = skutil.view_as_blocks(arr_in=np.where(np.isnan(a), 0, a),block_shape=(block_size,block_size))
    bsum = bw.sum(axis=-1).sum(axis=-1)
    nsum = n.sum(axis=-1).sum(axis=-1)
    transform, bounds = get_tranform_and_bounds(profile=profile, array_shape=bsum.shape)
    bmean = bsum/nsum
    return  bmean, transform, bounds




