# FEM Solvers and Meshes for Simple Simulations

## Data
- **`Sample02_093_segCell.nii_relabel.tif`**: One 3D image of the **C. elegans** dataset
  
- **`Sample02_093_segCell.nii_relabel.gz.vol.gz`**:  A mesh reconstructed from the **C. elegans** dataset inside a micro-cavity environment.

- **`twocells_in_cavity.vol.gz`**: A simplified geometry of **two spherical cells in a micro-cavity**, used for testing simulations.
---

##  Solver Scripts

- **`solver.py`**  
  Standard solver for the **`twocells_in_cavity.vol.gz`**  without voltage plateaus (floating electrodes).

- **`solver_draw.py`**  
  Similar to `solver.py`, but includes **visualization of the solution** on the mesh after solving.

- **`solver_manual.py`**  
  A **manual implementation** of the solver for comparison with `solver.py`.

- **`solver_celegans.py`**  
  Solver for the **C. elegans mesh** without applying plateau constraints.

- **`solver_celegans_plateau.py`**  
  Solver for the **C. elegans mesh** with voltage plateaus applied to designated electrode regions.

---

## Running the Solver with Outer/Inner Cell Properties

To simulate the effect of different dielectric properties between **outer** and **inner** cells, follow these steps:

### Step 1: Identify Outer Cells

Use the `mark_outer_cell.py` script to identify and save a list of outer cells (i.e., cells touching the background):

```bash
python3 mark_outer_cell.py
```

This will generate a NumPy file (e.g., `marked_outer_093.npy`) containing the list of cell labels classified as "outer" cells.

### Step 2: Run the Outer/Inner Solver

Use `solver_outer_inner.py` to assign different electrical properties to outer and inner cells:

```bash
python3 solver_outer_inner.py
```
