import rasterio

lulc = '/data/hrea/kenya_lightscore/kisumu/kisumu_lulc_2018.tif'
lisc = '/data/hrea/kenya_lightscore/kisumu/kisumu_lightscore_2017.tif'

with rasterio.open(lisc) as src:
    trans = src.read_transform()
    with rasterio.open(lulc, 'r+', ) as dst:
        dst.write_transform(trans)


