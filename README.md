# geo-cellular-automata
Forecasting HREA layers using Cellular automata

## Install

```bash
    git clone https://github.com/UNDP-Data/geo-cellular-automata.git
  
```

Use you favorite tool to create a virtual environment and then install re

```commandline
    pip install scikit-image scipy affine matplotlib pandas rasterio cartopy xarray azure-storage-blob aiohttp
    
```
## Set up the env var
```commandline
HREA_SAMPLE_DATA_SAS=
```

## Download HREA sample data
```commandline
cd geo-cellular-automata
python -m camodel.download_data -h
python -m camodel.download_data -l Kisumu -f /tmp

```
## Explore data

```commandline
python -m camodel.main
```
In case there are issues with cartopy you can force a clean build using

```commandline
pip install --upgrade --no-binary shapely
```