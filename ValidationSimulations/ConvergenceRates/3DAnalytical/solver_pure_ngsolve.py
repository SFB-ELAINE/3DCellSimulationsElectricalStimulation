import os
from ngsolve import CoefficientFunction, Mesh, SetNumThreads, ParameterC
from ngsolve.utils import printmaster
from geometry import cell_in_box
from netgen.meshing import meshsize
from netgen.meshing import Mesh as NGMesh
import pandas as pd
import numpy as np
from scipy.constants import epsilon_0 as e0
import mpi4py.MPI as MPI
from TMPsympyJones import return_solution
from sympy import diff
import sys
from cell_simulation_utilities import ThinLayerFunctionSpace
from mpi4py.MPI import COMM_WORLD

SetNumThreads(1)

# mpi communicators

comm = COMM_WORLD
rank = comm.rank
num_proc = comm.size

# discretization order (used for FESpace and for geometry approximation):
order = int(sys.argv[1])
n_ref = int(sys.argv[2])
curved = bool(int(sys.argv[3]))
if not curved:
    printmaster("Use uncurved elements")
else:
    printmaster("Use curved elements")
# maximum element size
maxh = 2.0

####
# unit system
####
L = 1e-6  # microns
M = 1.0  # kg
T = 1e-3  # milliseconds
I_unit = 1e-3  # milliamps

####
# constants in right unit system
####

conv_sigma = L ** (-3) * M ** (-1) * T**3 * I_unit**2
conv_eps = L ** (-3) * M ** (-1) * T**4 * I_unit**2
conv_field = L * M * T ** (-3) * I_unit ** (-1)
conv_current_density = conv_sigma * conv_field
conv_f = 1.0 / T

freq = 1e7 / conv_f
dm = 7e-9 / L
sig_m = 8.7e-6
sig_buf = 1.0
sig_cyt = 0.48
eps_m = 5.8
eps_buf = 80
eps_cyt = 60


ki = ParameterC(0.0)
km = ParameterC(0.0)
ke = ParameterC(0.0)
Ym = ParameterC(0.0)
kitmp = sig_cyt + 1j * 2.0 * np.pi * freq * eps_cyt * e0
kmtmp = sig_m + 1j * 2.0 * np.pi * freq * eps_m * e0
ketmp = sig_buf + 1j * 2.0 * np.pi * freq * eps_buf * e0
ki.Set(kitmp / conv_sigma)
km.Set(kmtmp / conv_sigma)
ke.Set(ketmp / conv_sigma)

l2errors = []
h1errors = []
ndofs = []
nels = []
TMPerr = []

h = 20e-6 / L  # using microns
R = 5e-6 / L
geo = cell_in_box(h, R)

# field! depends on geometry and boundary
U = 1  # apply 1V
E = U / (h * L) / conv_field
printmaster("Field in SI units: {}".format(E * conv_field))

# trig version:
if rank == 0:
    ngmesh = geo.GenerateMesh(meshsize.moderate, maxh=maxh, quad_dominated=False)
    for i in range(n_ref):
        ngmesh.Refine()
    ngmesh.Distribute(comm)

else:
    ngmesh = NGMesh.Receive(comm)
    ngmesh.SetGeometry(geo)

mesh = Mesh(ngmesh)
if curved:
    mesh.Curve(order)

printmaster("Materials: ", mesh.GetMaterials())
printmaster("Boundaries: ", mesh.GetBoundaries())

# wait before computing
comm.Barrier()

#####
# get analytical solution
#####
compi, compo, TMP_ana = return_solution(cartesian=True)
TMP_ana = CoefficientFunction(
    [eval(str(TMP_ana)) if bnd == "membrane" else 0 for bnd in mesh.GetBoundaries()]
)
cf_i = eval(str(compi))
cf_o = eval(str(compo))
gradcf_o = CoefficientFunction(
    (
        eval(str(diff(compo, "x"))),
        eval(str(diff(compo, "y"))),
        eval(str(diff(compo, "z"))),
    )
)

gradcf_i = CoefficientFunction(
    (
        eval(str(diff(compi, "x"))),
        eval(str(diff(compi, "y"))),
        eval(str(diff(compi, "z"))),
    )
)
cfs_analytical = {"ecm": cf_o, "cytoplasm": cf_i}
cfanalytical = CoefficientFunction([cfs_analytical[mat] for mat in mesh.GetMaterials()])
cfsgrad_analytical = {"ecm": gradcf_o, "cytoplasm": gradcf_i}
cfgradanalytical = CoefficientFunction(
    [cfsgrad_analytical[mat] for mat in mesh.GetMaterials()]
)

i = 0

printmaster("Create FESpaces")
# Creates function spaces
fes = ThinLayerFunctionSpace(
    mesh,
    domain_outside="ecm",
    domain_inside="cytoplasm",
    interface="membrane",
    dirichlet="cube",
    order=order,
    complex=True,
)

print("{} DOFs on rank {}.".format(fes.fes.ndof, comm.rank))
comm.Barrier()
printmaster("Global DOFs: ", fes.fes.ndofglobal)
print("unknowns on rank {}: {}".format(comm.rank, fes.fes.ndof))
ndofs.append(fes.fes.ndofglobal)
if comm.size == 1:
    nels.append(mesh.ne)
else:
    nels.append(MPI.COMM_WORLD.allreduce(mesh.ne, op=MPI.SUM))


fes.set_boundary_condition(cf_o)

# Assigns material properties
sigvals = {"cytoplasm": ki, "ecm": ke, "pcm": ke}
sig = mesh.MaterialCF(sigvals, default=0)

printmaster("######## Material parameters ######")
printmaster("Sig_m | Sig_buf | Sig_cyt |")
printmaster("{} | {} | {} |".format(sig_m, sig_buf, sig_cyt))
printmaster("Eps_m | Eps_buf | Eps_cyt")
printmaster("{} | {} | {} |".format(eps_m, eps_buf, eps_cyt))
printmaster("k_m | k_buf | k_cyt | Ym")
printmaster("| {} | {} | {} | {} |".format(km, ke, ki, km / dm))
printmaster("###################################")

printmaster("{:.1e}".format(freq))

# interface admittance
Ym.Set(km.Get() / dm)

a, f = fes.get_bilinear_form(sig, Ym)

fes.iterative_solver(
    a, f, "cg", "local", solver_kwargs={"maxsteps": 10000, "printrates": comm.rank == 0}
)

printmaster("Evaluating solution")
errors = fes.compute_error(cfanalytical, cfgradanalytical)
printmaster("TMP error")
tmperr = fes.compute_jump_error(TMP_ana)
printmaster("Done")

h1errors.append(errors["Energy"])
l2errors.append(errors["L2"])
TMPerr.append(tmperr)
# write results
if rank == 0:
    if not os.path.exists("results_pure"):
        os.makedirs("results_pure")
results = {"nels": nels, "ndofs": ndofs, "h1": h1errors, "l2": l2errors, "tmp": TMPerr}
df = pd.DataFrame(results)
filename = "results_pure/results_order"
if not curved:
    filename += "_uncurved"
filename += "_{}.csv".format(order)
if n_ref == 1:
    mode = "w"
    header = True
else:
    mode = "a"
    header = False
if rank == 0:
    df.to_csv(filename, index=False, mode=mode, header=header)
printmaster("Success")
