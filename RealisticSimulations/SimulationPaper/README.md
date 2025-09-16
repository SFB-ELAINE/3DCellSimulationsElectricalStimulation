# C.elegans + Astec Simulation
## Data

This project uses two publicly available biological datasets:

- **Astec Dataset** (Phallusia mammillata embryonic development)  
  [https://figshare.com/articles/dataset/Astec-Pm1_Wild_type_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_8-17_/8223890](https://figshare.com/articles/dataset/Astec-Pm1_Wild_type_Phallusia_mammillata_embryo_live_SPIM_imaging_stages_8-17_/8223890)

- **C. elegans Dataset** (Cell segmentation and tracking)  
  [https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/C.elegans-Cells-HK.zip](https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/C.elegans-Cells-HK.zip)

For the raw meshes, see the
[meshes folder](../../CGALMesh_3D/C.elegans_ASTEC_meshes).

## Run the Solver on HPC (SLURM)

Once you have your final `.vol.gz` mesh files, you can run the solver on each mesh using a SLURM job array.

1. **Edit `runSolver.sh`**  
   Set the correct path to your processed mesh folder:

   ```bash
   DIRECTORY="/your/path/to/processed_meshes"
   ```

2. **Make sure the SLURM script is ready**  
   The file `runsingleshell.SLURM` will be submitted as an array job. Each SLURM task will run `solver.py` on one mesh file. 

3. **Run the submission script**

   Submit all solver jobs by running:

   ```bash
   bash runSolver.sh
   ```

   This script:
   - Counts the number of `.vol.gz` files
   - Submits a SLURM array job where each task runs the solver on one mesh
   - Logs output/errors to the `logs/` folder

`solver.py` accepts the input mesh file as a command-line argument.
#### Running Without SLURM (Local Testing)
If you're not using SLURM and just want to test the solver on a single mesh, you can run `solver.py` directly from the command line:

```bash
python3 solver.py [job_id]
```

Here, `[job_id]` is the index of the mesh file (based on sorted order) in the `processed_meshes/` folder. For example:

```bash
python3 solver.py 0
```

This would process the **first** mesh file in `processed_meshes/`.

---

### Optional: Run the Solver with Outer/Inner Cell Properties
Before running simulations that assign different electrical properties to outer and inner cells, you need to identify which cells are touching the background (outer cells).
#### 1. Prepare your labelled image
Use the `mark_outer_cell.py` script to detect outer cells and create the label list:

```bash
python3 mark_outer_cell.py
```
#### 2. Run the marking script
To simulate different electrical properties for outer and inner cells, use the `solver_outer_inner.py` script.
This version accepts two command-line arguments:
1. The path to the mesh file (`.vol.gz`)
2. The path to the NumPy file containing the list of outer cell labels (`marked_outer.npy`)

#### Usage
Run the solver manually for a single mesh like this:
```bash
python3 solver_outer_inner.py /path/to/mesh.vol.gz /path/to/marked_outer.npy
```
Example:
```bash
python3 solver_outer_inner.py processed_meshes/sample_093.vol.gz marked_outer_093.npy
```
This will:
- Load the mesh and outer/inner cell label list
- Apply separate conductivity and permittivity values to outer vs inner cells
- Run simulations
#### SLURM Integration
To run on SLURM, modify your `runsingleshell.SLURM` script to pass both file paths to the solver:
```bash
#!/bin/bash
#SBATCH --job-name=elegans_solver
#SBATCH --output=logs/solver_%A_%a.out
#SBATCH --error=logs/solver_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99  # Overwritten by runSolver.sh

# load module which configures everything 
MESH_DIR="/your/path/to/processed_meshes"
LABEL_PATH="/your/path/to/marked_outer.npy"
FILES=($(ls "$MESH_DIR"/*.vol.gz | sort))
MESH_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

python3 solver_outer_inner.py "$MESH_FILE" "$LABEL_PATH"
```
Make sure the `marked_outer.npy` file matches the mesh being processed.

---
This setup allows you to easily test or batch-run simulations where different materials are assigned based on spatial cell context (e.g., outer vs inner).
