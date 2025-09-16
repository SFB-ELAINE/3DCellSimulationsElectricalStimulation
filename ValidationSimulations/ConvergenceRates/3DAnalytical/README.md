# 3D Benchmark

The analytical solution of a cell in an external field
with isotropic membrane properties as described in [1]
with the thin layer approximation for the cell membrane
as described in [2] was used.
We implemented it using SymPy (see `sympysolution.py`).
This example requires MPI.
**Note that we run a shell script because we vary the number
of MPI processes depending on the mesh size.**
We used up to 42 cores here, please adjust it according to your
system specifications. 

Run to reproduce the convergence analysis (see also p. 184 in [3]):

```
./convergence_run.sh
python plot_convergence.py
python plot_convergence_l2.py
python plot_convergence_h1.py
```

This is the original run as in [3].
We also include now the pure NGSolve version using
CG preconditioned with a Jacobi preconditioner.
The NGSolve CG solver works also with a PETSc
preconditioner, which can be seen in another script.
The CG solver from PETSc does not work reliably
(without further investigation: might be because the matrix
is communicated without preserving its properties).

For comparison, we provide benchmark results in `benchmark/`.


[1] Jones, T.B. Electromechanics of Particles. 
    Cambridge University Press, Cambridge, 1995.
    doi: 10.1017/cbo9780511574498. 
[2] Sukhorukov, V.L. et al. A single-shell model for biological cells
    extended to account for the dielectric anisotropy of the
    plasma membrane. J. Electrostat., 50(3):191–204, 2001.
    doi: 10.1016/S0304-3886(00)00037-1.
[3] Zimmermann, J. Numerical modelling of electrical stimulation for
    cartilage tissue engineering. Phd Thesis, Universitaet Rostock, 2022.
    doi: 10.18453/rosdok_id00004117
