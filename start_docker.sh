#!/bin/bash

#change the following to your needs:
# my_container
# uni_id
# docker_usr_name
# docker_img
# flag

docker run -it  \
-u $(id -u):$(id -g) \
--name my_container \
-v /media/data:/workspace/data_local \
-v /media/MedIP-Praxisprojekt:/workspace/MedIP-PP \
-v /home/WIN-UNI-DUE/smbohann:/workspace/smbohann \
--gpus all \
smbohann/bvm2021:v1
