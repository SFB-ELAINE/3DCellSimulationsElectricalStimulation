import ngsolve
try:
    from ngsolve.utils import printmaster
except ImportError:
    try:
        from ngsolve.utils import printonce as printmaster
    except ImportError:
        print("Neither printmaster nor printonce could be imported.")
from cell_simulation_utilities import (
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

freqs = np.logspace(3, 3, num=1)
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

fes_list = []
for domain in domains:
    fes_parameters = {
        "order": order,
        "definedon": domain,
    }
    if domain in dirichlet:
        fes_parameters["dirichlet"] = dirichlet[domain]
    if domain in interface_info:
        print(interface_info[domain])
        boundary = mesh.Boundaries(interface_info[domain])
        fes_parameters["definedonbound"] = boundary
    print(fes_parameters)
    fes_list.append(ngsolve.H1(mesh, **fes_parameters))
fes = ngsolve.FESpace(fes_list)
fes = ngsolve.CompressCompound(fes)
fes_interface = ngsolve.FacetFESpace(mesh, order=order)

"""
fes = ShelledThinLayerFunctionSpace(
    mesh,
    domains,
    interface_info,
    outer_inner_pairs,
    dirichlet,
    order=order,
    complex=True,
)
"""
print("{} DOFs on rank {}.".format(fes.ndof, comm.rank))
printmaster("Global DOFs: ", fes.ndofglobal)
comm.Barrier()

interfaces = ["Cell_membrane_1", "Cell_membrane_2", "Cell_interface_1_2"]
####
# unit system
####
units = UnitConverter(
    length_unit=1e-6,  # microns
    mass_unit=1.0,  # kg
    time_unit=1e-3,  # milliseconds
    current_unit=1e-6,
)  # microamps

conductivity = Conductivity(domains, complex=True)
interface_admittance_class = BoundaryAdmittance(interfaces, complex=True)
interface_thickness = {"Cell_membrane_.*": 7e-9}

conductivities = {"ecm": 1.0, "Cell_.*": 0.48}
permittivities = {"ecm": 80, "Cell_.*": 60}
interface_conductivities = {"Cell_membrane_.*": 8.7e-6}
interface_permittivities = {"Cell_membrane_.*": 5.8}


def get_bilinear_form(sigma, interface_admittance):
    u_tuple = fes.TrialFunction()
    v_tuple = fes.TestFunction()

    a = ngsolve.BilinearForm(fes)

    i = 0
    enum_domains = {}
    for u, v, domain in zip(u_tuple, v_tuple, domains):
        enum_domains[domain] = i
        i += 1
        a += sigma * ngsolve.grad(u) * ngsolve.grad(v) * ngsolve.dx(domain)

    # Interface condition
    outer_inner_pairs = {
        "Cell_membrane_1": ("ecm", "Cell_1"),
        "Cell_membrane_2": ("ecm", "Cell_2"),
        "Cell_interface_1_2": ("Cell_1", "Cell_2"),
    }

    for boundary in interfaces:
        domain_out = outer_inner_pairs[boundary][0]
        out_index = enum_domains[domain_out]
        u_out = u_tuple[out_index]
        v_out = v_tuple[out_index]

        domain_in = outer_inner_pairs[boundary][1]
        in_index = enum_domains[domain_in]
        u_in = u_tuple[in_index]
        v_in = v_tuple[in_index]

        a += (
            interface_admittance
            * (u_out - u_in)
            * (v_out - v_in)
            * ngsolve.ds(boundary)
        )

    # is zero, no further definition needed
    f = ngsolve.LinearForm(fes)
    return a, f


# set BC
bnd = {"electrode_1": 0.0, "electrode_3": 1.0}

volume_ratio = ngsolve.Integrate(
    ngsolve.CoefficientFunction(1.0),
    mesh,
    definedon=mesh.Materials("Cell_1|Cell_2"),
)
volume_ratio /= ngsolve.Integrate(ngsolve.CoefficientFunction(1.0), mesh)

printmaster("Volume ratio: ", volume_ratio)

# Set boundary condition

gfu = ngsolve.GridFunction(fes)
gfu.components[0].Set(bnd)

sig_at_freq = conductivity.get_coefficient_function(mesh)
y_at_freq = interface_admittance_class.get_coefficient_function(mesh)
printmaster("Bilinear form")
a, f = get_bilinear_form(sig_at_freq, y_at_freq)
solver_kwargs = {"maxsteps": 10000, "printrates": comm.rank == 0, "atol": 1e-11}
preconditioner_kwargs = {"pc_type": "bjacobi"}

for i, freq in enumerate(freqs):
    printmaster("{:.1e}".format(freq))

    printmaster("Setting material and interface properties")
    material_properties = conductivity.prepare_parameters(
        conductivities,
        permittivities,
        freq,
        scaling_factor=1.0 / units.conductivity_unit,
    )
    interface_properties = interface_admittance_class.prepare_parameters(
        interface_conductivities,
        interface_permittivities,
        freq,
        scaling_factor=1.0 / units.conductivity_unit,
    )

    conductivity.set_parameters(material_properties)
    for interface in interface_properties:
        interface_properties[interface] = (
            units.L * interface_properties[interface] / interface_thickness[interface]
        )
    interface_admittance_class.set_parameters(interface_properties)

    printmaster("Solving")
    fes.iterative_solver(
        a,
        f,
        "CG",
        "PETScPC",
        solver_kwargs=solver_kwargs,
        preconditioner_kwargs=preconditioner_kwargs,
    )
    impedance = fes.compute_impedance(sig_at_freq, y_at_freq, verbose=comm.rank == 0)
    # convert to Ohm
    impedance /= units.conductivity_unit * units.L
    Z[i] = impedance

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
