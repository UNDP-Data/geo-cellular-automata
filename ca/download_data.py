
import asyncio
from azure.storage.blob.aio import ContainerClient



class ConfigError(Exception):
    pass

async def download_kenya_lightscore(folder_to=None, connection_string:str=None, container_name=None, prefix:str = None ):


    async with ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name) as cc:
        async for blob in cc.list_blobs(name_starts_with='HREA_COGs/HREA_Kenya'):
            if 'lightscore' in blob.name:
                _, bname = os.path.split(blob.name)
                local_cog_path = os.path.join(folder_to, bname)
                with open(local_cog_path, 'wb') as local_cog:
                    stream = await cc.download_blob(blob.name, max_concurrency=8)
                    await stream.readinto(local_cog)
                print(f'Downloaded {bname}')













if __name__ == '__main__':
    import os
    connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    container_name = os.environ['CONTAINER_NAME']
    root_folder = os.environ['ROOT_FOLDER']
    account_name = connection_string.split(';')[1].split('=')[1]
    prefix = f"{os.path.join(root_folder, 'HREA_Kenya')}"
    asyncio.run(download_kenya_lightscore(connection_string=connection_string,
                                          container_name=container_name,
                                          prefix=prefix,
                                          folder_to='/data/hrea/kenya_lightscore'
                                          ))

