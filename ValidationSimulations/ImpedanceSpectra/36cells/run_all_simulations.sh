#! /bin/bash

cd SingleShell

mpirun -np 12 python solver.py &> log

cd ..

cd SingleShellWall

mpirun -np 12 python solver.py &> log

cd ..

cd DoubleShell

mpirun -np 12 python solver.py &> log

cd ..

cd DoubleShellWall

mpirun -np 12 python solver.py &> log

cd ..
