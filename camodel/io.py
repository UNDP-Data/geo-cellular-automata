import xarray
from osgeo import gdal, gdal_array
import numpy as np
import os
import xarray as xr
import rasterio


def read_xarray(src_folder:str=None) -> xarray.Dataset:
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

def read_all_data(src_folder=None) -> np.ndarray:
    """
    Read HREA cogs using GDAL from the src folder into a structured array
    :param src_folder:
    :return: a strcutured array with the data for every year located in
    a column determined by the year
    [('2012', '<f4'), ('2013', '<f4'), ('2014', '<f4'), ('2015', '<f4'), ('2016', '<f4'), ('2017', '<f4'), ('2018', '<f4'), ('2019', '<f4'), ('2020', '<f4')]
    """
    names = [e for e in os.listdir(src_folder) if e.endswith('.tif')]
    names.sort()
    years = [os.path.splitext(name)[0].split('_')[2] for name in names]
    data = None
    for i, fname in enumerate(names):
        year = years[i]
        fpath = os.path.join(src_folder, fname)
        ds = gdal.OpenEx(fpath, gdal.OF_RASTER | gdal.OF_READONLY)
        band = ds.GetRasterBand(1)
        #band.SetNoDataValue(-1.0)
        yearly_data = band.ReadAsArray()
        band_dtype = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)

        band_shape = band.YSize, band.XSize
        nodata = band.GetNoDataValue()
        yearly_data[np.isclose(yearly_data, nodata)] = np.nan

        if data is None:
            ddtype = [(year, band_dtype) for year in years]
            data = np.empty(shape=band_shape, dtype=ddtype)
        data[year] = yearly_data
    return data

def read_rio(src_folder=None) -> np.ndarray:
    """
    Read HREA cogs using rasterio from the src folder into a structured array
    :param src_folder: str, path to the folder folding the layers
    :return: tuple where the first element is a
    np.ndarray with following dtype [('2012', '<f4'), ('2013', '<f4'), ('2014', '<f4'), ('2015', '<f4'), ('2016', '<f4'), ('2017', '<f4'), ('2018', '<f4'), ('2019', '<f4'), ('2020', '<f4')]
    and the second is the rasterio profile dict holding info related to the data including  the spatial extent
    The  original nodata value is replaced with nan's

    """
    names = [e for e in os.listdir(src_folder) if e.endswith('.tif')]
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
