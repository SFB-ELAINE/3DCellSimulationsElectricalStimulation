import ngsolve

try:
    from ngsolve.utils import printonce
except ImportError:
    from ngsolve.utils import printmaster as printonce
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
import glob

# no threads, can slow down mpi
ngsolve.SetNumThreads(1)


mesh_input = []  # geometry input mesh files
for file in sorted(glob.glob("../../RealisticSimulations/SimpleTest/Sample02_093_segCell.nii_relabel.gz.vol.gz")):
    mesh_input.append(file)
# Only one mesh
job_id = 0
file_name_mesh = mesh_input[job_id]
basename = os.path.splitext(os.path.basename(file_name_mesh))[0]
# mpi communicators
printonce("NAME:", basename)
comm = MPI.COMM_WORLD
rank = comm.rank
num_proc = comm.size

mesh = ngsolve.Mesh(file_name_mesh)
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

# discretization order (used for FESpace and for geometry approximation):
order = 2

printonce("Load mesh")
mesh = load_mesh(file_name_mesh, comm, curve_order=order)
printonce("Done")

printonce("Materials: ", mesh.GetMaterials())
printonce("Boundaries: ", mesh.GetBoundaries())

# wait a bit, just in case
comm.Barrier()

freqs = np.logspace(3, 8, num=50)
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

conductivities = {"ecm": 0.01}
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

parameters_changed = False
# Set boundary condition
fes.set_boundary_condition(bnd)

sig_at_freq = sigma.get_coefficient_function(mesh)
y_at_freq = interface_admittance.get_coefficient_function(mesh)
printonce("Bilinear form")
a, f = fes.get_bilinear_form(sig_at_freq, y_at_freq)
conditions = ["membrane", "no_membrane"]
preconditioner_kwargs = {"pc_type": "bjacobi"}
save_vtu = False
for condition in conditions:
    print(condition)
    solver_kwargs = {"maxsteps": 30000, "printrates": comm.rank == 0, "atol": 1e-9}
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
            if condition == "membrane":
                impedance = fes.compute_impedance(sig_at_freq, y_at_freq, verbose=False)
            elif condition == "no_membrane":
                # set interface impedance to zero
                impedance = fes.compute_impedance(sig_at_freq, ngsolve.CF(0.0), verbose=False)
            else:
                raise ValueError("Illegal condition for membrane chosen.")
            # convert to Ohm
            impedance /= units.conductivity_unit * units.L
            Z[i] = impedance
        else:
            printonce(
                "Skipped simulations because material and interface parameters have not changed."
            )
            Z[i] = Z[i - 1]
        if save_vtu:
            fes.export_VTK(
                filename=f"field_no_plateau_{basename}_{int(freq / 1e3)}",
                sigma_dict=material_properties,
                subdivision=1,
                save_field=True,
            )
            fes.export_surface_jump_VTK(
                filename=f"TMP_no_plateau_{basename}_{int(freq / 1e3)}",
                subdivision=0,
                floatsize="single",
            )
        printonce("Z:", Z[i])
    # write results
    if rank == 0:
        if not os.path.exists("results"):
            os.makedirs("results")
    results = {"freq": freqs, "Zreal": Z.real, "Zimag": Z.imag}
    df = pd.DataFrame(results)
    filename = "results/results_single_shell_"
    filename += "{}.csv".format(condition)
    df.to_csv(filename, index=False)
    printonce("Success")
