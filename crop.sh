#!/bin/bash
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=undpngddlsgeohubdev01;AccountKey=r1XiTbCtrorE63XfmclS9YihGCPicHWqKIL1BWe2CdeWoSNvif2OXgSyGcIw4BjaGiwkjvZTQGcYNIXCmjvwFA==;EndpointSuffix=core.windows.net"



declare -A locations
declare -A countries
#locations[Mangochi]=MWIr104
#countries[Mangochi]=Malawi
#locations[Antsiranana]=MDGr122
#countries[Antsiranana]=Madagascar
locations[Cunado]=AGOr213
countries[Cunado]=Angola
#locations[Kisumu]=KENr105
#countries[Kisumu]=Kenya
#locations[Apurimac]=PERr103
#countries[Apurimac]=Peru
#locations[Apurimac]=PERr103
#countries[Apurimac]=Peru

for location in "${!locations[@]}"
do

  echo "${location} => ${locations[${location}]}"
  echo "select * from admin.admin1 as admin1 where gdlcode='"${locations[${location}]}"'"
  #docker run --rm -it -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING -v /home/janf/:/root -v /data:/data ghcr.io/osgeo/gdal:ubuntu-small-latest ogr2ogr -f FlatGeobuf /vsiaz/ibmdata/ca_samples/${location}/${location}.fgb PG:service=GEOHUBDB admin.admin1 -sql "select * from admin.admin1 as admin1 where gdlcode='"${locations[${location}]}"'" -progress
  docker run --rm -it -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING -v /home/janf/:/root -v /data:/data ghcr.io/osgeo/gdal:ubuntu-small-latest ogr2ogr -f GPKG /data/hrea/ca_samples/${location}/${location}.gpkg PG:service=GEOHUBDB admin.admin1 -sql "select * from admin.admin1 as admin1 where gdlcode='"${locations[${location}]}"'" -progress -t_srs EPSG:4326

  for year in 2012 2013 2014 2015 2016 2017 2018 2019 2020;do
    echo ${location}, ${locations[${location}]}, ${countries[${location}]}, ${year}
    #docker run --rm -it -e CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING -v /home/janf/:/root -v /data:/data ghcr.io/osgeo/gdal:ubuntu-small-latest gdalwarp -cutline /vsiaz/ibmdata/ca_samples/${location}/${location}.fgb -cl sql_statement -crop_to_cutline /vsiaz/hrea/HREA_COGs/HREA_${countries[${location}]}_${year}_v1/${countries[${location}]}_set_lightscore_sy_${year}.tif /vsiaz/ibmdata/ca_samples/${location}/hrea_lightscore_${location}_${year}.tif
    #docker run --rm -it -e CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING -v /home/janf/:/root -v /data:/data ghcr.io/osgeo/gdal:ubuntu-small-latest gdalwarp -cutline /vsiaz/ibmdata/ca_samples/${location}/${location}.fgb -cl sql_statement -crop_to_cutline /vsiaz/hrea/HREA_COGs/HREA_${countries[${location}]}_${year}_v1/${countries[${location}]}_set_lightscore_sy_${year}.tif /data/hrea/ca_samples/${location}/hrea_lightscore_${location}_${year}.tif
    echo "done"
  done
#  echo "key  : ${i}"
#  echo "value: ${locations[$i]}"
done

