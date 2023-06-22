import logging
import os
import pickle

import scipy.stats

from camodel.io import read_rio, toGDAL, read
import numpy as np
from camodel import plot_tools, model
from scipy.signal import fftconvolve, correlate2d
from numpy.fft import fft2, ifft2

logger = logging.getLogger()

def slope(dem=None, cellsize=None, degrees=False):
    """
    Compute slope on a elevation model
    :param dem: 2D np.ndarray
    :param cellsize: the cellsize in meters
    :param degrees: return slope in degree rather than percent
    :return: ndarray representing slope
    """
    px, py = np.gradient(dem, cellsize)
    px**=2
    py**=2
    px+=py
    #slope = np.sqrt(px ** 2 + py ** 2)
    slope = np.sqrt(px)
    if degrees:
        return np.degrees(np.arctan(slope))
    return slope


def compute_block_oper(struct_array=None,block_size=None):
    """
    Fast Fourier based block sum
    :param binary_hrea_array:
    :param block_size:
    :return:
    """
    dtype = struct_array.dtype
    ndtype = [(e, 'f2') for e in dtype.names]
    data = np.empty(shape=struct_array.shape, dtype=ndtype)
    for name in dtype.names:
        a = struct_array[name]
        arr = fftconvolve(np.where(np.isnan(a), 0, a), np.ones((block_size,block_size), dtype='u1'), mode='same')
        data[name] = arr

    return data

def local_correlation_loop(a, window_size=3):
    # Define the size of the sliding window (neighborhood size)


    # Calculate the padding size for the sliding window
    padding = window_size // 2

    # Pad the dataset to handle window at the edges
    padded_dataset = np.pad(a, padding, mode='reflect')

    # Initialize an empty array to store the correlation values
    correlation_map = np.zeros_like(a, dtype=float)

    # Slide the window over the padded dataset and compute correlation
    for i in range(padding, padding + a.shape[0]):
        for j in range(padding, padding + a.shape[1]):
            # Extract the local patch
            patch = padded_dataset[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Compute the correlation between the patch and the central pixel
            correlation = np.corrcoef(patch.flatten(), patch[padding, padding])[0, 1]

            # Store the correlation value in the correlation map
            correlation_map[i - padding, j - padding] = correlation

    print(correlation_map)



def local_corr(a, b, window_size=3):
    # Define the size of the sliding window (neighborhood size)


    # Calculate the padding size for the sliding window
    padding = window_size // 2

    # Pad the datasets to handle window at the edges
    padded_dataset1 = np.pad(a, padding, mode='reflect')
    padded_dataset2 = np.pad(b, padding, mode='reflect')

    # Create a 4D array of patches for each dataset
    patches1 = np.lib.stride_tricks.sliding_window_view(padded_dataset1, window_shape=(window_size, window_size))
    patches2 = np.lib.stride_tricks.sliding_window_view(padded_dataset2, window_shape=(window_size, window_size))

    # Reshape the patches for correlation computation
    reshaped_patches1 = patches1.reshape((-1, window_size ** 2))
    reshaped_patches2 = patches2.reshape((-1, window_size ** 2))

    # Compute the correlation coefficients for each pair of patches
    correlation_map = np.corrcoef(reshaped_patches1.T, reshaped_patches2.T, rowvar=False)[:-1, -1]

    # Reshape the correlation map to match the original dataset shape
    correlation_map = correlation_map.reshape(a.shape)

    return

def local_corr_fft(a, b, window_size=3):
    # Define the size of the sliding window (neighborhood size)


    # Calculate the padding size for the sliding window
    padding = window_size // 2

    # Pad the datasets to handle window at the edges
    padded_dataset1 = np.pad(a, padding, mode='reflect')
    padded_dataset2 = np.pad(b, padding, mode='reflect')

    # Create a 4D array of patches for each dataset
    patches1 = np.lib.stride_tricks.sliding_window_view(padded_dataset1, window_shape=(window_size, window_size))
    patches2 = np.lib.stride_tricks.sliding_window_view(padded_dataset2, window_shape=(window_size, window_size))

    # Compute the correlation using FFT-based convolution
    correlation_map = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(patches1) * np.fft.fft2(patches2.conj()))))

    # Reshape the correlation map to match the original dataset shape
    correlation_map = correlation_map[:, :, padding, padding]

    return correlation_map


def corr2d(a, b):
    pad = np.max(a.shape) // 2
    fft1 = np.fft.fft2(np.pad(a, pad))
    fft2 = np.fft.fft2(np.pad(b, pad))
    prod = fft1 * fft2.conj()
    result_full = np.fft.fftshift(np.fft.ifft2(prod))
    #return result_full.real[pad:-pad, pad:-pad]
    return result_full.real

def corr(a, b, window_size=3):
    a = scipy.stats.zscore(a, nan_policy='omit')
    b = scipy.stats.zscore(b)
    # Pad the arrays to handle the window at the edges
    paddeda = np.pad(np.where(np.isnan(a), -1, a), window_size // 2, mode='reflect')
    #paddeda = np.pad(a, window_size // 2, mode='reflect')

    paddedb = np.pad(b, window_size // 2, mode='reflect')
    # Compute the local patches for both datasets using array slicing
    patches1 = np.lib.stride_tricks.sliding_window_view(paddeda, window_shape=(window_size, window_size))

    patches2 = np.lib.stride_tricks.sliding_window_view(paddedb, window_shape=(window_size, window_size))

    # Reshape the patches to prepare for cross-correlation computation
    reshaped_patches1 = patches1.reshape((-1, window_size, window_size))
    reshaped_patches2 = patches2.reshape((-1, window_size, window_size))


    # Compute cross-correlation using NumPy's correlate2d function
    cross_correlation = np.array([correlate2d(p1, p2, mode='valid')[0, 0]
    #cross_correlation = np.array([corr2d(p1, p2)
    #cross_correlation = np.array([fftconvolve(p1, p2[::-1, ::-1], mode='valid')[0, 0]
                                  for p1, p2 in zip(reshaped_patches1, reshaped_patches2)])
    # Normalize the cross-correlation values to the range of -1 to 1
    cross_correlation = (cross_correlation - cross_correlation.min()) / (cross_correlation.max() - cross_correlation.min())
    cross_correlation = 2 * cross_correlation - 1
    cross_correlation = cross_correlation.reshape(a.shape)
    #cross_correlation = np.where(np.isnan(a), np.nan, cross_correlation)
    return cross_correlation


if __name__ == '__main__':
    logging.basicConfig()
    src_folder = '/data/hrea/kenya_lightscore/kisumu'
    logger.name = os.path.split(__file__)[-1]

    logger.setLevel(logging.INFO)
    hrea_array_path = os.path.join(src_folder, 'hrea.npy')
    hrea_meta_path = os.path.join(src_folder, 'hrea.meta')
    dsm_path = os.path.join(src_folder, 'kisumu_dsm.tif')
    lulc_2017_path = os.path.join(src_folder, 'kisumu_lulc_2017.tif')
    dist_to_roads_path = os.path.join(src_folder, 'dist_to_roads.tif')

    if not os.path.exists(hrea_array_path):
        logger.debug(f"Reading sample HREA from COG's")
        hrea, profile = read_rio(src_folder=src_folder, filter_string='lightscore')
        np.save(hrea_array_path, hrea, allow_pickle=True)
        with open(hrea_meta_path, 'wb') as dst:
            p = pickle.Pickler(file=dst)
            p.dump(profile)


    else:
        # os.remove(hrea_array_path)
        logger.debug(f"Reading sample HREA from npy")
        hrea = np.load(hrea_array_path)
        profile = None
        with open(hrea_meta_path, 'rb') as src:
            p = pickle.Unpickler(file=src)
            profile = p.load()



    lulc_2017 = read(src_path=lulc_2017_path)
    dsm = read(src_path=dsm_path)
    dist_to_roads = read(src_path=dist_to_roads_path)


    #slope = slope(dsm, cellsize=30)

    c = corr(hrea['2017'], dist_to_roads, window_size=9)

    #bsum = model.compute_block_sum(binary_hrea_array=hrea, block_size=29)
    #bsum = compute_block_oper(struct_array=hrea, block_size=5)
    #bsum = corr2d(lulc_2017, lulc_2017 )
    a, t, b = model.block_mean(array_in=c, block_size=29, profile=profile)
    print(t.to_gdal(), profile['transform'].to_gdal())
    toGDAL(struct_array=c,
           path=os.path.join(src_folder, 'corr_distr_hrea2017.tif'),dtype=np.float32,nodata=2**32-1,
           transform=profile['transform'],overviews=[2,4, 6, 8, 10])
    exit()
    # custom geospatial visualization using cartopy
    arrays = {'hrea': hrea, 'bsum':bsum }

    arrays_bounds = {'hrea': profile['bounds'], 'bsum': profile['bounds'] }
    plot_tools.geo_plot(arrays=arrays, arrays_bounds=arrays_bounds, cmap='RdYlGn', )

