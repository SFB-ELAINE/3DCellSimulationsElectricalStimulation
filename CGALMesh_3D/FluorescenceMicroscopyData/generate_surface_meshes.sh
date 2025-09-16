#!/bin/bash

folder_tif_reconstruction='/path/to/output/reconstructed_tif'
folder_inr='/path/to/output/inr_files'
folder_medit='/path/to/output/medit_mesh'
folder_netgen='/path/to/output/netgen_mesh'
MESH_TOOL_PATH='/path/to/CGAL/'

folders=("$folder_inr" "$folder_medit" "$folder_netgen")

# Create necessary folders
for folder in "${folders[@]}"; do
    if [ ! -d "$folder" ]; then
        mkdir -p "$folder"
        echo "Folder created: $folder"
    else
        echo "Folder already exists: $folder"
    fi
done

sizing_scale=1.0
edge_size=1000.0

# Convert TIF to INR
conda activate titk #change path to your titk if it is different
python3 convert_tif_to_inr.py "$folder_tif_reconstruction" "$folder_inr" 0.22 0.22 0.22
conda deactivate

# Generate Medit mesh
for filepath in "$folder_inr"/*.inr.gz; do
    filename=$(basename "$filepath")
    base_name="${filename%.inr.gz}"
    echo "Processing $base_name..."
    start_time=$(date +%s)
    if timeout 3000 "$MESH_TOOL_PATH/mesh_3D_image_with_weight_and_features" "$filepath" "$folder_medit" "$sizing_scale" "$edge_size"; then
        echo "Processed $base_name -> $folder_medit"
    else
        if [ $? -eq 124 ]; then
            echo "Timeout: $base_name"
        else
            echo "Error processing $base_name"
        fi
    fi
    end_time=$(date +%s)
    echo "Time taken: $((end_time - start_time))s"
done

# Convert Medit to NETGEN
python3 medit_to_netgen.py "$folder_medit" "$folder_netgen"
echo "Mesh conversion completed."
