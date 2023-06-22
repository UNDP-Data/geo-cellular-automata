
import numpy as np
import os
import xarray as xr
import rasterio
from rasterio.enums import Resampling

def read_xarray(src_folder:str=None) -> xr.Dataset:
    """
    read sample HREA cogs using xarray
    :param src_folder: str
    :return: xarray Dataset with the data
    """

    names = [e for e in os.listdir(src_folder) if e.endswith('.tif')]
    paths = [os.path.join(src_folder, e) for e in names]

    return xr.open_mfdataset(paths=paths,
                             engine='rasterio',
                             combine='nested',
                             concat_dim='time',
                             coords='minimal',
                             compat='override',
                             chunks={'band':'auto'}

    )

def read(src_path=None, band=1):
    with rasterio.open(src_path) as src:
        return src.read(band)



def read_rio(src_folder=None, filter_string=None) -> np.ndarray:
    """
    Read HREA cogs using rasterio from the src folder into a structured array
    :param src_folder: str, path to the folder folding the layers
    :return: tuple where the first element is a
    np.ndarray with following dtype [('2012', '<f4'), ('2013', '<f4'), ('2014', '<f4'), ('2015', '<f4'), ('2016', '<f4'), ('2017', '<f4'), ('2018', '<f4'), ('2019', '<f4'), ('2020', '<f4')]
    and the second is the rasterio profile dict holding info related to the data including  the spatial extent
    The  original nodata value is replaced with nan's

    """
    names = [e for e in os.listdir(src_folder) if filter_string in e]
    names.sort()
    years = [os.path.splitext(name)[0].split('_')[2] for name in names]
    data = None
    profile = None
    for i, fname in enumerate(names):
        year = years[i]
        fpath = os.path.join(src_folder, fname)

        with rasterio.open(fpath, 'r') as src:
            for i in range(1, src.count + 1):
                # read image into ndarray
                yearly_data = src.read().squeeze()
                if profile is None:
                    profile = src.profile

                shape = src.shape
                dtype = src.dtypes[i-1]
                yearly_data[np.isclose(yearly_data, src.nodata)] = np.nan
                if data is None:
                    ddtype = np.dtype([(year, dtype) for year in years])
                    data = np.empty(shape=shape, dtype=ddtype)
                    profile['bounds']=src.bounds

                data[year] = yearly_data

    return data, profile


def toGDAL(struct_array=None,
           path=None,
           dtype=None,
           nodata=None,
           transform=None,
           epsg=4326,
           compress='zstd',
           tiled=True,
           blockxsize=256,
           blockysize=256,
           overviews=None

           ):

    height, width = struct_array.shape


    if struct_array.dtype.names:
        count = len(struct_array.dtype.names)
    else:
        count = 1
    with rasterio.Env():

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        nprofile = dict(
            count=count,
            driver='GTiff',
            dtype=dtype,
            nodata=nodata,
            width=width,
            height=height,
            transform=transform,
            crs=rasterio.crs.CRS.from_epsg(epsg),
            compress=compress,
            tiled=tiled,
            blockxsize=blockxsize,
            blockysize=blockysize,

        )
        band = 1
        with rasterio.open(path, 'w', **nprofile) as dst:
            if struct_array.dtype.names:
                for name in struct_array.dtype.names:
                    arr = struct_array[name]
                    dst.write(arr, band)
                    # if overviews:
                    #     dst.build_overviews(overviews, Resampling.nearest)
                    #     dst.update_tags(ns='rio_overview', resampling='nearest')
                    band+=1
            else:
                dst.write(struct_array, band)