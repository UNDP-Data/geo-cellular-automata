from osgeo import gdal, gdal_array
import numpy as np
import os
import pandas as pd
import xarray as xr
import rasterio
def read_xarray(src_folder=None):
    names = [e for e in os.listdir(src_folder) if e.endswith('.tif')]
    paths = [os.path.join(src_folder, e) for e in names]
    years = [os.path.splitext(name)[0].split('_')[2] for name in names]
    td = pd.to_datetime(years, format='%Y')
    time_var = xr.Variable('time', td)
    print(td)
    datasets = []

    data = xr.open_mfdataset(paths=paths,
                             engine='rasterio',
                             combine='nested',
                             concat_dim='time',
                             coords='minimal',
                             compat='override',
                             chunks={'band':'auto'}

    )
    #for path in paths:


    return data

def read_all_data(src_folder=None):
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

            #data = np.empty(shape=[len(names)]+list(yearly_data.shape), dtype=data.dtype)
        data[year] = yearly_data

        ds = None

    return data

def read_rio(src_folder=None):
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
