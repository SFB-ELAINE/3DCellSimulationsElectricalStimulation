# LinKK test

This repository provides tools to compute impedance spectra using an iterative solver and verify the results with the **linKK test**.

## Usage

### 1. Run the Solver
To compute the impedance spectra with different solver tolerances, run:

```bash
mpirun -np 24 python3 linkk_test_solver_convergence.py
mpirun -np 24 python3 linkk_test_membrane_admittance.py
```

- `-np 24` specifies the number of processes. Adjust this value based on your available cores.
- The solver will output the computed impedance data in the `results` folder.

### 2. Run the linKK Test and Plot Results
After obtaining the spectra, open the Jupyter notebook:

```bash
jupyter notebook Plot-LinKK-test.ipynb
```

This notebook loads the solver output and generates plots showing the results of the **linKK test**, including whether convergence or modeling errors are detected.
