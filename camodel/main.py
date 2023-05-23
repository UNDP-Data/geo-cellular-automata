import logging
import argparse
import os
from camodel.io import read_rio
import numpy as np
import pickle
from camodel import model as m
from camodel import plot_tools
from matplotlib.colors import ListedColormap
logger = logging.getLogger()

def main(sample_hrea_folder=None):
    assert sample_hrea_folder not in ['', None], f'Invalid sample_hrea_folder={sample_hrea_folder}'
    assert os.path.exists(sample_hrea_folder), f'{sample_hrea_folder} does not exist'

    #1. read data and store locally  in npy format to optimize reading

    hrea_array_path = os.path.join(sample_hrea_folder, 'hrea.npy')
    hrea_meta_path = os.path.join(sample_hrea_folder, 'hrea.meta')

    if not os.path.exists(hrea_array_path):
        logger.debug(f"Reading sample HREA from COG's")
        hrea, profile = read_rio(src_folder=sample_hrea_folder)
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

    # threshold data at 80%
    binary_hrea_data = m.apply_threhold(hrea_struct_array=hrea)
    onekm_agg_50perc, transform, bounds = m.aggregate(
        binary_hrea_array=binary_hrea_data,
        block_size=29,
        threshold=.5,
        profile=profile
    )

    #plot_tools.plot_neigh_bars(binary_hrea_array=onekm_agg_50perc, target_year='last')

    colors_dict = {
        -1: "#FFFFFF00",
        0: "black",
        1: "orange"
    }

    # ploti(rec_array=onekm_agg)
    # We create a colormar from our list of colors
    cmp = ListedColormap([colors_dict[x] for x in colors_dict.keys()])
    cmp.labels = {'no electricity': 'black', 'electrified': 'orange'}

    # custom geospatial visualization using cartopy
    arrays = {'binary': binary_hrea_data,  '1km_sum_50perc': onekm_agg_50perc}

    arrays_bounds = {'binary': profile['bounds'],  '1km_sum_50perc': bounds}
    plot_tools.geo_plot(arrays=arrays, arrays_bounds=arrays_bounds, cmap=cmp)

if __name__ == '__main__':

    import sys
    logging.basicConfig()

    logger.name = os.path.split(__file__)[-1]

    logger.setLevel(logging.INFO)
    # silence azure http logger
    azlogger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    azlogger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='Download sample HREA data from UNDP Azure blob')
    src_folder = '/data/hrea/kenya_lightscore/kisumu'
    if not os.path.exists(src_folder):

        parser.add_argument('-f', '--source-folder',
                            help='Full absolute path to the folder where the data will be downloaded',
                            type=str, required=True )
        args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
        src_folder = args.source_folder
    main(sample_hrea_folder=src_folder)
