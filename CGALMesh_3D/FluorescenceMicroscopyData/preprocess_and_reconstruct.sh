#!/bin/bash

# Define the input and output folders
folder_tiff_original='/path/to/your/segmented_tiffs'
folder_tiff='/path/to/output/processed_tiffs'
folder_tif_reconstruction='/path/to/output/reconstructed_tif'

folders=("$folder_tiff_original" "$folder_tiff" "$folder_tif_reconstruction")

# Create folders if they don't exist
for folder in "${folders[@]}"; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Folder created: $folder"
    else
        echo "Folder already exists: $folder"
    fi
done

# Run preprocessing and reconstruction

python3 process_tif_parallel.py "$folder_tiff_original" "$folder_tiff"


conda activate #/path/to/your/conda/env/u-segment3D
python3 recontruction3D.py "$folder_tiff" "$folder_tif_reconstruction"
conda deactivate
