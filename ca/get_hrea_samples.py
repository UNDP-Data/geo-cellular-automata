
import asyncio
from azure.storage.blob.aio import ContainerClient
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import generate_container_sas, AccountSasPermissions, upload_blob_to_url
import datetime

class ConfigError(Exception):
    pass

async def download_kenya_lightscore(folder_to=None, connection_string:str=None, container_name=None, countries:str = None ):


    async with ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name) as cc:
        for country in countries:
            async for blob in cc.list_blobs(name_starts_with=f'HREA_COGs/HREA_{country}'):
                if 'lightscore' in blob.name:
                    print(f'/vsiaz/{container_name}/{blob.name}')
                    # _, bname = os.path.split(blob.name)
                    # local_cog_path = os.path.join(folder_to, bname)
                    # with open(local_cog_path, 'wb') as local_cog:
                    #     stream = await cc.download_blob(blob.name, max_concurrency=8)
                    #     await stream.readinto(local_cog)
                    # print(f'Downloaded {bname}')
import math
def upload_blob(src_path: str = None, connection_string: str = None, container_name: str = None,
                dst_blob_path: str = None, overwrite: bool = True, max_concurrency: int = 8) -> None:
    """
    Uploads the src_path file to Azure dst_blob_path located in container_name
    @param src_path: str, source file
    @param connection_string: strm the Azure storage account  connection string
    @param container_name: str, container name
    @param dst_blob_path: relative path to the container  where the src_path will be uploaded
    @param overwrite: bool
    @param max_concurrency: 8
    @return:  None
    """
    logtrack = []
    def _progress_(current, total) -> None:
        progress = current / total * 100
        rounded_progress = int(math.floor(progress))
        if rounded_progress not in logtrack and rounded_progress % 2 == 0:
            print(f'uploaded - {rounded_progress}%')
            logtrack.append(rounded_progress)

    for attempt in range(1,4):
        try:
            print(f'uploading {dst_blob_path}')
            with BlobServiceClient.from_connection_string(connection_string) as blob_service_client:
                with blob_service_client.get_blob_client(container=container_name, blob=dst_blob_path) as blob_client:
                    with open(src_path, "rb") as upload_file:
                        blob_client.upload_blob(upload_file, overwrite=overwrite, max_concurrency=max_concurrency, progress_hook=_progress_)
                    print(f"Successfully wrote {src_path} to {dst_blob_path}")
                #remove any error
                error_blob_path = f'{dst_blob_path}.error'
                with blob_service_client.get_blob_client(container=container_name, blob=error_blob_path) as error_blob_client:
                    if error_blob_client.exists():
                        error_blob_client.delete_blob(delete_snapshots=True)

            break
        except Exception as e:
            if attempt == 3:
                print(f'Failed to upload {src_path} in attempt no {attempt}.')
                raise
            print(f'Failed to upload {src_path} in attempt no {attempt}. Trying again... ')
            continue



async def list_lightscore(connection_string:str=None, container_name=None, ):

    proto, acc, key, suffix = connection_string.split(';')
    proto = proto.split('=')[1]
    account_name = acc.split('=')[1]
    account_key = key[11:]
    host_name = suffix.split('=')[1]


    async with ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name) as cc:
        with open('/home/janf/hrea_admin_links.txt', 'w') as out:
            async for blob in cc.list_blobs():
                url = f"{proto}://{account_name}.blob.{host_name}/{container_name}/{blob.name}"
                #print(f'/vsiaz/{container_name}/{blob.name}')
                sas_token = generate_container_sas(
                    account_name=account_name,
                    container_name=container_name,account_key=account_key,
                    permission=AccountSasPermissions(read=True),
                    expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=10000)

                )

                url_with_sas = f"{url}?{sas_token}"
                out.write(f'{url_with_sas}\n')

async def cp(connection_string:str=None, container_name=None, ):

    proto, acc, key, suffix = connection_string.split(';')
    proto = proto.split('=')[1]
    account_name = acc.split('=')[1]
    account_key = key[11:]
    host_name = suffix.split('=')[1]


    async with ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name) as cc:
        async for blob in cc.list_blobs(name_starts_with=f'ibmdata'):
            src_blob = cc.get_blob_client(blob)
            _, *dst_blob = blob.name.split('/')
            dst_blob_path = '/'.join(dst_blob)
            dst_blob = cc.get_blob_client(dst_blob_path)
            print(src_blob.url)
            cp = await dst_blob.start_copy_from_url(src_blob.url),


            print(cp)
            break

async def upload(folder_from=None, connection_string:str=None, container_name=None, ):

        for fname in os.listdir(folder_from):
            location = folder_from.split('/')[-1]
            fpath = os.path.join(folder_from, fname)
            blob_path = f'{container_name}/ca_samples/{location}/{fname}'

            upload_blob(src_path=fpath,connection_string=connection_string, container_name=container_name,
                        dst_blob_path=blob_path)









if __name__ == '__main__':
    import os
    connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    container_name = os.environ['CONTAINER_NAME']
    root_folder = os.environ['ROOT_FOLDER']
    account_name = connection_string.split(';')[1].split('=')[1]
    prefix = f"{os.path.join(root_folder, 'HREA_Kenya')}"
    countries = ('Malawi', 'Madagascar', 'Angola', 'Kenya', 'Peru', 'Vietnam')
    # asyncio.run(download_kenya_lightscore(connection_string=connection_string,
    #                                       container_name=container_name,
    #                                       countries=countries,
    #                                       folder_to='/data/hrea/kenya_lightscore'
    #                                       ))

    asyncio.run(
        list_lightscore(
                        connection_string=connection_string,
                        container_name='ibmdata',

                        )
    )


    # for folder in ('/data/hrea/ca_samples/Apurimac', '/data/hrea/ca_samples/Cunado'):
    #
    #     asyncio.run(upload(folder_from=folder,connection_string=connection_string, container_name='ibmdata',))


