import ngsolve
try:
    from ngsolve.utils import printmaster
except ImportError:
    try:
        from ngsolve.utils import printonce as printmaster
    except ImportError:
        print("Neither printmaster nor printonce could be imported.")
from cell_simulation_utilities import (
    ShelledThinLayerFunctionSpace,
    load_mesh,
    UnitConverter,
    Conductivity,
    BoundaryAdmittance,
)
import mpi4py.MPI as MPI
import pandas as pd
import numpy as np
import os
import pprint

# no threads, can slow down mpi
ngsolve.SetNumThreads(1)

# mpi communicators

comm = MPI.COMM_WORLD
rank = comm.rank
num_proc = comm.size

# discretization order (used for FESpace and for geometry approximation):
order = 2

printmaster("Load mesh")
mesh = load_mesh("twocells_in_cavity.vol.gz", comm, curve_order=order)
printmaster("Done")

printmaster("Materials: ", mesh.GetMaterials())
printmaster("Boundaries: ", mesh.GetBoundaries())

# wait a bit, just in case
comm.Barrier()

freqs = np.logspace(3, 12, num=91)
Z = np.zeros(freqs.shape, dtype=np.complex128)

printmaster("Create FESpaces")
domains = mesh.ngmesh.GetRegionNames(dim=3)
interface_info = {}
outer_inner_pairs = {}
for fd in mesh.ngmesh.FaceDescriptors():
    # do not consider boundaries
    if fd.domout == 0:
        continue

    domin = domains[fd.domin - 1]
    domout = domains[fd.domout - 1]
    print(fd, domin, domout)

    if domin not in interface_info:
        interface_info[domin] = []
    if domout not in interface_info:
        interface_info[domout] = []
    # append only if not there
    if fd.bcname not in interface_info[domin]:
        interface_info[domin].append(fd.bcname)
    if fd.bcname not in interface_info[domout]:
        interface_info[domout].append(fd.bcname)
    if fd.bcname in outer_inner_pairs:
        print(f"Problem with {fd.bcname}, and ({domin}, {domout})")
    outer_inner_pairs[fd.bcname] = (domout, domin)
for interface in interface_info:
    interface_info[interface] = "|".join(interface_info[interface])
print("Interfaces:")
pprint.pprint(interface_info)
print("Pairs:")
pprint.pprint(outer_inner_pairs)
dirichlet = {"ecm": "electrode_1|electrode_3"}
fes = ShelledThinLayerFunctionSpace(
    mesh,
    domains,
    interface_info,
    outer_inner_pairs,
    dirichlet,
    order=order,
    complex=True,
)
print("{} DOFs on rank {}.".format(fes.fes.ndof, comm.rank))
comm.Barrier()
printmaster("Global DOFs: ", fes.fes.ndofglobal)

# set BC
bnd = {"electrode_1": 0.0, "electrode_3": 1.0}

volume_ratio = ngsolve.Integrate(
    ngsolve.CoefficientFunction(1.0),
    fes.mesh,
    definedon=fes.mesh.Materials("cytoplasm"),
)
volume_ratio /= ngsolve.Integrate(ngsolve.CoefficientFunction(1.0), fes.mesh)

printmaster("Volume ratio: ", volume_ratio)

####
# unit system
####
units = UnitConverter(
    length_unit=1e-6,  # microns
    mass_unit=1.0,  # kg
    time_unit=1e-3,  # milliseconds
    current_unit=1e-6,
)  # microamps

sigma = Conductivity(fes.domains, complex=True)
interface_admittance = BoundaryAdmittance(fes.interfaces, complex=True)

conductivities = {"ecm": 1.0}
permittivities = {"ecm": 80}
for domain in domains:
    if "Cell" in domain:
        conductivities[domain] = 0.48
        permittivities[domain] = 60

interface_conductivities = {}
interface_permittivities = {}
interface_thickness = {}
for interface in fes.interfaces:
    interface_conductivities[interface] = 8.7e-6
    interface_permittivities[interface] = 5.8
    interface_thickness[interface] = 7e-9

parameters_changed = True
# Set boundary condition
fes.set_boundary_condition(bnd)

sig_at_freq = sigma.get_coefficient_function(mesh)
y_at_freq = interface_admittance.get_coefficient_function(mesh)
printmaster("Bilinear form")
a, f = fes.get_bilinear_form(sig_at_freq, y_at_freq)
solver_kwargs = {"maxsteps": 10000, "printrates": comm.rank == 0, "atol": 1e-11}
preconditioner_kwargs = {"pc_type": "bjacobi"}

for i, freq in enumerate(freqs):
    printmaster("{:.1e}".format(freq))

    printmaster("Setting material and interface properties")
    material_properties = sigma.prepare_parameters(
        conductivities,
        permittivities,
        freq,
        scaling_factor=1.0 / units.conductivity_unit,
    )
    interface_properties = interface_admittance.prepare_parameters(
        interface_conductivities,
        interface_permittivities,
        freq,
        scaling_factor=1.0 / units.conductivity_unit,
    )

    # check if parameters changed
    parameters_changed = sigma.check_parameters_change(
        material_properties, tolerance=0.01
    )
    parameters_changed = interface_admittance.check_parameters_change(
        interface_properties, tolerance=0.01
    )
    sigma.set_parameters(material_properties)
    for interface in interface_properties:
        interface_properties[interface] = (
            units.L * interface_properties[interface] / interface_thickness[interface]
        )
    interface_admittance.set_parameters(interface_properties)

    if parameters_changed:
        printmaster("Solving")
        fes.iterative_solver(
            a,
            f,
            "CG",
            "local",
            solver_kwargs=solver_kwargs,
        )
        impedance = fes.compute_impedance(
            sig_at_freq, y_at_freq, verbose=comm.rank == 0
        )
        # convert to Ohm
        impedance /= units.conductivity_unit * units.L
        Z[i] = impedance
    else:
        printmaster(
            "Skipped simulations because material and interface parameters have not changed."
        )
        Z[i] = Z[i - 1]

    fes.export_VTK(filename=f"out_{int(freq)}.vtk")
    printmaster("Z:", Z[i])
# write results
if rank == 0:
    if not os.path.exists("results"):
        os.makedirs("results")
results = {"freq": freqs, "Zreal": Z.real, "Zimag": Z.imag}
df = pd.DataFrame(results)
filename = "results/results_single_shell"
filename += "_{}.csv".format(order)
df.to_csv(filename, index=False)
printmaster("Success")
