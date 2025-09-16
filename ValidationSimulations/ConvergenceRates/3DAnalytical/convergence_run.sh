#!/bin/bash

mpirun -np 5 python solver.py 1 1 0
mpirun -np 11 python solver.py 1 2 0
mpirun -np 24 python solver.py 1 3 0
mpirun -np 42 python solver.py 1 4 0

mpirun -np 5 python solver.py 1 1 1
mpirun -np 11 python solver.py 1 2 1
mpirun -np 24 python solver.py 1 3 1
mpirun -np 42 python solver.py 1 4 1

mpirun -np 5 python solver.py 2 1 0
mpirun -np 11 python solver.py 2 2 0
mpirun -np 24 python solver.py 2 3 0
mpirun -np 42 python solver.py 2 4 0

mpirun -np 5 python solver.py 2 1 1
mpirun -np 11 python solver.py 2 2 1
mpirun -np 24 python solver.py 2 3 1
mpirun -np 42 python solver.py 2 4 1
