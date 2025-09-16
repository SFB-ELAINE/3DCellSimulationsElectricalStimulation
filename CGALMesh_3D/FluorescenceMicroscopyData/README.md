# C.elegans + Astec Simulation
## Data

This project uses two publicly available biological datasets:

- **Astec Dataset** (Phallusia mammillata embryonic development)  
  [https://figshare.com/articles/dataset/Astec-Pm1_Wild_type_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_8-17_/8223890](https://figshare.com/articles/dataset/Astec-Pm1_Wild_type_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_8-17_/8223890)

- **C. elegans Dataset** (Cell segmentation and tracking)  
  [https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/C.elegans-Cells-HK.zip](https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/C.elegans-Cells-HK.zip)

Please download and extract the datasets before running any preprocessing or simulation steps.

## Python Package Requirements

| Package         | Version   | Link                                                                 |
|-----------------|-----------|----------------------------------------------------------------------|
| tifffile        | 2024.9.20 | https://pypi.org/project/tifffile/                                  |
| scikit-image    | 0.24.0    | https://pypi.org/project/scikit-image/                              |
| segment3D       | 0.1.1     | https://github.com/DanuserLab/u-segment3D (install using conda) |
## Usage

### Step 1: Prepare and Mesh the 3D surface

The full pipeline has been split into two steps to simplify debugging and intermediate checks.

---

#### 1. Preprocess and Reconstruct 3D Volumes

Run the following script to:
- Preprocess your segmented TIFF stacks
- Reconstruct them into 3D image volumes

```bash
bash preprocess_and_reconstruct.sh
```

Make sure to edit the script to set your input, output paths and activate the conda environment for `u-segment3D`:
```bash
folder_tiff_original='/path/to/your/segmented_tiffs'
folder_tiff='/path/to/output/processed_tiffs'
folder_tif_reconstruction='/path/to/output/reconstructed_tif'
conda activate #/path/to/your/conda/env/u-segment3D
```

---

#### 2. Generate Surface Meshes

Once the 3D reconstruction is complete, generate surface meshes using:

```bash
bash generate_surface_meshes.sh
```

This script:
- Converts `.tif` volumes to `.inr.gz` using [timagetk](CGALMesh_3D/README.md)  
- Generates Medit meshes using [`mesh_3D_image_with_weight_and_features`](CGALMesh_3D/mesh_3D_image_with_weight_and_features.cpp)
- Converts the result to NETGEN `.vol.gz` meshes

Edit paths in the script before running:
```bash
folder_tif_reconstruction='/path/to/output/reconstructed_tif'
folder_inr='/path/to/output/inr_files'
folder_medit='/path/to/output/medit_mesh'
folder_netgen='/path/to/output/netgen_mesh'
MESH_TOOL_PATH='/path/to/CGAL/bin'
```

   
### Step 2: Merge NETGEN Mesh with Electrode Geometry

Take the `.vol.gz` files from Step 1 and combine them with the electrode model to produce a single mesh with labeled domains and shared boundaries.

1. **Run the merge script**  
   Use `runMeshing.py` to process all mesh files in batch:

   ```bash
   python3 runMeshing.py
   ```

   This script automatically:
   - Loads each `.vol.gz` mesh
   - Merges it with the predefined electrode geometry (defined in `process_mesh_script.py`)
   - Saves the final combined mesh into the `processed_meshes` folder
   - Logs output and errors into the `logs` folder

Make sure paths in `runMeshing.py` point to your actual input/output directories before running.
## Precomputed Data

To save time or if you'd like to skip the mesh generation steps, we provide preprocessed data and meshes on Zenodo.

**Download the data archive here:**  
[https://zenodo.org/records/15609194](https://zenodo.org/records/15609194) #TODE: Change link when publish

The archive contains:

### Step 1.1: Post-Processed 3D Volumes
- **`post_processed_images/`**  
  Reconstructed 3D image volumes from segmented TIFF stacks.

### Step 1.2: Surface Meshes (Cells Only)
- **`surface_meshes_cells_only/`**  
  Surface-only NETGEN `.vol.gz` meshes converted from CGAL `.mesh` files.  
  *Note: These contain only the cell geometries—no electrodes yet.*

### Step 2: Final Meshes with Electrodes
- **`final_meshes_with_electrode/`**  
  Final volume meshes (`.vol.gz`) containing both the cell domains and the embedded electrode geometry.  
  These are ready for simulation in the solver scripts.

Use these files if you want to skip the image processing and meshing steps (Steps 1 & 2).

The solver scripts can be found in [Solver scripts](../../RealisticSimulations/SimulationPaper).
