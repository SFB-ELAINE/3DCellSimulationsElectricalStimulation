import ngsolve
from ngsolve.utils import printonce
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
import sys
import re

# no threads, can slow down mpi
ngsolve.SetNumThreads(1)

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <mesh_path> <label_path>")
    sys.exit(1)

mesh_path = sys.argv[1]  # e.g. "/path/to/mesh.vol.gz"
label_path = sys.argv[2]  # e.g. "/path/to/marked_outer.npy"

# === extract a basename if you need it ===
basename = os.path.splitext(os.path.basename(mesh_path))[0]
print(f"Running solver on mesh = {basename}")

# === load the label array ===
marked_outer = np.load(label_path)
# mpi communicators
printonce("NAME:", basename)
comm = MPI.COMM_WORLD
rank = comm.rank
num_proc = comm.size

mesh = ngsolve.Mesh(mesh_path)
domains = mesh.ngmesh.GetRegionNames(dim=3)
interface_info = {}
outer_inner_pairs = {}
for fd in mesh.ngmesh.FaceDescriptors():
    # do not consider boundaries
    if fd.domout == 0:
        continue

    domin = domains[fd.domin - 1]
    domout = domains[fd.domout - 1]

    if domin not in interface_info:
        interface_info[domin] = []
    if domout not in interface_info:
        interface_info[domout] = []
    if fd.bcname not in interface_info[domin]:
        interface_info[domin].append(fd.bcname)
    if fd.bcname not in interface_info[domout]:
        interface_info[domout].append(fd.bcname)
    if fd.bcname in outer_inner_pairs:
        print(f"Problem with {fd.bcname}, and ({domin}, {domout})")
    outer_inner_pairs[fd.bcname] = (domout, domin)

for interface in interface_info:
    interface_info[interface] = "|".join(interface_info[interface])

# discretization order (used for FESpace and for geometry approximation):
order = 2

printonce("Load mesh")
mesh = load_mesh(mesh_path, comm, curve_order=order)
printonce("Done")

printonce("Materials: ", mesh.GetMaterials())

# wait a bit, just in case
comm.Barrier()

freqs = np.logspace(3, 8, num=60)
Z = np.zeros(freqs.shape, dtype=np.complex128)

printonce("Create FESpaces")

dirichlet = {"ecm": "electrode_1|electrode_2"}
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
printonce("Global DOFs: ", fes.fes.ndofglobal)

# set BC
bnd = {"electrode_1": 0.0, "electrode_2": 1.0}


cell_domains = []
for domain in fes.domains:
    cell_domains.append(domain)
# Filter to include only elements that start with "Cell_"
filtered_cells = [domain for domain in cell_domains if domain.startswith("Cell_")]

# Join the filtered list with "|"
str_domain = "|".join(filtered_cells)


volume_ratio = ngsolve.Integrate(
    ngsolve.CoefficientFunction(1.0),
    fes.mesh,
    definedon=fes.mesh.Materials(str_domain),
)
volume_ratio /= ngsolve.Integrate(ngsolve.CoefficientFunction(1.0), fes.mesh)

printonce("Volume ratio: ", volume_ratio)


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

conductivities = {}
permittivities = {}

for domain in domains:
    # Use regex to extract the first integer found in the domain string.
    match = re.search(r"\d+", domain)
    if match:
        number = int(match.group())
        # Check if the number is in the marked_outer array.
        if number not in marked_outer:
            conductivities[domain] = 0.48
            permittivities[domain] = 60
        else:
            conductivities[domain] = 0.23
            permittivities[domain] = 60

conductivities["ecm"] = 0.01
permittivities["ecm"] = 80


interface_conductivities = {}
interface_permittivities = {}
interface_thickness = {}

for interface in fes.interfaces:
    match = re.search(r"\d+", interface)
    if match:
        number = int(match.group())
        if number not in marked_outer:
            interface_conductivities[interface] = 8.7e-6
            interface_permittivities[interface] = 5.8
            interface_thickness[interface] = 7e-9
        else:
            interface_conductivities[interface] = 500e-6
            interface_permittivities[interface] = 5.8
            interface_thickness[interface] = 7e-9


parameters_changed = True
# Set boundary condition
fes.set_boundary_condition(bnd)

sig_at_freq = sigma.get_coefficient_function(mesh)
y_at_freq = interface_admittance.get_coefficient_function(mesh)
printonce("Bilinear form")
a, f = fes.get_bilinear_form(sig_at_freq, y_at_freq)
solver_kwargs = {"maxsteps": 30000, "printrates": comm.rank == 0, "atol": 1e-9}
preconditioner_kwargs = {"pc_type": "bjacobi"}

for i, freq in enumerate(freqs):
    printonce("frequency {:.1e}".format(freq))

    printonce("Setting material and interface properties")
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
    for interface in interface_properties:
        interface_properties[interface] = (
            units.L * interface_properties[interface] / interface_thickness[interface]
        )
    # check if parameters changed
    if i > 0:
        parameters_changed_sigma = sigma.check_parameters_change(
            material_properties, tolerance=0.00001
        )
        parameters_changed_admittance = interface_admittance.check_parameters_change(
            interface_properties, tolerance=0.00001
        )
        parameters_changed = parameters_changed_sigma and parameters_changed_admittance

    sigma.set_parameters(material_properties)
    interface_admittance.set_parameters(interface_properties)

    if parameters_changed:
        printonce("Solving")

        fes.iterative_solver(
            a,
            f,
            "CG",
            "local",
            solver_kwargs=solver_kwargs,
        )
        impedance = fes.compute_impedance(sig_at_freq, y_at_freq, verbose=False)
        # convert to Ohm
        impedance /= units.conductivity_unit * units.L
        Z[i] = impedance
    else:
        printonce(
            "Skipped simulations because material and interface parameters have not changed."
        )
        Z[i] = Z[i - 1]

    if freq == 1e3 or freq == 1e7 or freq == 1e8:
        fes.export_VTK(
            filename=f"field_out_in_{basename}_{int(freq / 1e3)}",
            sigma_dict=material_properties,
            subdivision=1,
            save_field=True,
        )
        fes.export_surface_jump_VTK(
            filename=f"TMP_out_in_{basename}_{int(freq / 1e3)}",
            subdivision=0,
            floatsize="single",
        )
    printonce("Z:", Z[i])
# write results
comm.Barrier()
if rank == 0:
    if not os.path.exists("results"):
        os.makedirs("results")
results = {"freq": freqs, "Zreal": Z.real, "Zimag": Z.imag}
df = pd.DataFrame(results)
filename = "results/results_outer_inner"
filename += "{}_{}.csv".format(order, basename)
df.to_csv(filename, index=False)
printonce("Success")
