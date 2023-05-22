import os
import asyncio
from azure.storage.blob.aio import ContainerClient
import math
import logging
logger = logging.getLogger(__name__)
SAMPLE_LOCATIONS = 'Mangochi', 'Antsiranana', 'Kisumu', 'Apurimac'

async def download(sas_token=None, sample_location=None, dst_folder=None):
    assert dst_folder not in ['', None], f'Invalid dst_folder={dst_folder}. It should be a local directory '
    assert  os.path.isdir(dst_folder), f'dst_fodler={dst_folder} is not a folder'
    assert sample_location in SAMPLE_LOCATIONS, f'sample_location={sample_location} is invalid. Valid locations are {SAMPLE_LOCATIONS}'

    logtrack = []


    async with ContainerClient.from_container_url(container_url=sas_token) as cclient:

        async for blob in cclient.list_blobs():
            if sample_location in blob.name:
                _, bname = os.path.split(blob.name)
                local_cog_path = os.path.join(dst_folder, bname)
                with open(local_cog_path, 'wb') as local_cog:
                    try:
                        stream = await cclient.download_blob(blob.name)
                        logger.debug(f'Downloading ')
                        await stream.readinto(local_cog)
                    except KeyboardInterrupt:
                        logger.info(f'Cancelling download')
                        raise
                logger.info(f'Downloaded {bname}')









if __name__ == '__main__':
    import sys
    import argparse
    logging.basicConfig()

    logger.name = os.path.split(__file__)[-1]

    logger.setLevel(logging.INFO)
    # silence azure http logger
    azlogger = logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
    azlogger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description='Download sample HREA data from UNDP Azure blob')

    parser.add_argument('-l', '--sample-location',
                        help='A sample location to download data for',
                        type=str,choices=SAMPLE_LOCATIONS, required=True, default=SAMPLE_LOCATIONS[0])
    parser.add_argument('-f', '--folder-to',
                        help='FUll apbsolute path to the folder where the data will be downloaded',
                        type=str, required=True )

    hrea_sas = os.environ.get('HREA_SAMPLE_DATA_SAS', None)
    if hrea_sas is None:
        raise Exception(f'Environment variable "HREA_SAMPLE_DATA_SAS" is not defined')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    location = args.sample_location

    dst_folder = args.folder_to


    asyncio.run(download(sas_token=hrea_sas, sample_location=location, dst_folder='/tmp'))






