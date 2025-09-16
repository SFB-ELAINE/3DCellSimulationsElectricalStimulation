# 2D Benchmark

The analytical solution as described in [1] was used.
We implemented it using SymPy (see `sympysolution.py`).
This example runs with the standard NGSolve installation
and does not require MPI.

Run to reproduce the convergence analysis (see also p. 183 in [2]):

```
python solver.py
python plot_convergence.py
python plot_convergence_l2.py
python plot_convergence_h1.py
```

For comparison, we provide benchmark results in `benchmark/`.


[1] Ben Belgacem, F. et al. Finite Element Methods for the Temperature in
    Composite Media with Contact Resistance. J. Sci. Comput., 63(2):478–501,
    2015. doi: 10.1007/s10915-014-9907-0.
[2] Zimmermann, J. Numerical modelling of electrical stimulation for
    cartilage tissue engineering. Phd Thesis, Universitaet Rostock, 2022.
    doi: 10.18453/rosdok_id00004117
