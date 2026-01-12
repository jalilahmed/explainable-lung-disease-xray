#!/bin/bash

echo "Starting bash script: run_training.sh"

set -e

# Change working director

WORKDIR="/home/jahmed/Tutorials/explainable-lung-disease-xray"

cd "$WORKDIR" || { echo "Failed to change directory to $WORKDIR"; exit 1; }

echo "Changed directory to: $WORKDIR"

# Mount NFS

#if mountpoint -q /mnt/n4i_ds423/Software/MRI_Data/DallasData_2018-2023/Data_modified_structure/Segmentation; then
#    echo "NFS already mounted."
#else
#    echo "Trying to mount NFS."
#    sudo mount -t cifs //192.168.2.81/Software/MRI_Data/DallasData_2018-2023/Data_modified_structure/Segmentation /mnt/n4i_ds423/Software/MRI_Data/DallasData_2018-2023/Data_modified_structure/Segmentation -o credentials=/etc/smbcredentials/ja_n4i_ds423.cred,iocharset=utf8,uid=1000,gid=1000,vers=3.0
#fi

# Run DOCKER with mounted volumns

DOCKER_IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"

echo "Starting Docker container with image: $DOCKER_IMAGE"

docker run --name ja_device_1 -it --rm --gpus device=0 --shm-size 12G \
    -v "$WORKDIR":/usr/app/src \
    -v /mnt/n4i_ds423/Employee_Folder/Jalil_Ahmed/NIH_Chest_Xray_Dataset/datasets/nih-chest-xrays/data/versions/3:/usr/app/data \
    -w /usr/app/src \
    "$DOCKER_IMAGE" \
    bash -c "
        pip install --no-cache-dir -r /usr/app/src/requirements.txt --no-deps
        echo 'Running training script...'
        python3 /usr/app/src/scripts/exp_04122025.py
    "