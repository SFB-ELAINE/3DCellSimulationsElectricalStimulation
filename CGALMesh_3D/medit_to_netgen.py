import meshio
from netgen.meshing import Mesh, FaceDescriptor
import numpy as np
import ngsolve
import sys


def name_surface(domain_in, domain_out):
    if domain_in == 0 and domain_out == 0:
        return "background"
    if domain_in == 0:
        return f"Cell_membrane_{domain_out}"
    elif domain_out == 0:
        return f"Cell_membrane_{domain_in}"
    elif domain_in > domain_out:
        # first domain always smaller than other
        return f"Cell_interface_{domain_out}_{domain_in}"
    return f"Cell_interface_{domain_in}_{domain_out}"


if len(sys.argv) != 2:
    raise ValueError("Pass only meshname to script.")

meshname = sys.argv[1]
meshout = meshname.replace(".mesh", "")
m = meshio.read(meshname)
mesh = Mesh()
mesh.AddPoints(m.points)

surface_labels = m.cell_data["medit:ref"][0]
domains = np.unique(surface_labels)
assert 0 not in domains
domain_names = {}
for domain in domains:
    domain_names[domain] = f"Cell_{domain}"

domin_domout_label = np.array([surface_labels[::2], surface_labels[1::2]])
all_surfaces = np.unique(domin_domout_label, axis=1)
all_surfaces = all_surfaces.T
domin_domout_label = domin_domout_label.T
domin_domout_label[
    np.argwhere(domin_domout_label[:, 0] == domin_domout_label[:, 1]), 1
] = 0
all_surfaces[np.argwhere(all_surfaces[:, 0] == all_surfaces[:, 1]), 1] = 0

fd_idxs = []
for idx, surface in enumerate(all_surfaces, start=1):
    domin = surface[0]
    domout = surface[1]
    bc = idx
    fd = FaceDescriptor(surfnr=idx, domin=domin, domout=domout, bc=bc)
    fd.bcname = name_surface(domin, domout)
    fd_idx = mesh.Add(fd)
    fd_idxs.append(fd_idx)
    mesh.SetBCName(idx - 1, fd.bcname)

fds = mesh.FaceDescriptors()


mat = m.cell_data["medit:ref"][1]
for cb in m.cells:
    if cb.dim == 3:
        chosen_data = cb.data[mat > 0]
        mesh.AddElements(dim=cb.dim, index=1, data=chosen_data, base=0)
    elif cb.dim == 2:
        triangles = cb.data[::2]
        for idx, surface in enumerate(all_surfaces, start=1):
            triangle_idxs = np.logical_and.reduce(
                domin_domout_label == surface, axis=-1
            )
            chosen_triangles = triangles[triangle_idxs]
            name = fds[idx - 1].bcname

            # elements need to be reordered, otherwise the orientation does not match
            if "interface" in name:
                mesh.AddElements(dim=cb.dim, index=idx, data=chosen_triangles, base=0)
            else:
                mesh.AddElements(
                    dim=cb.dim, index=idx, data=np.flip(chosen_triangles, 1), base=0
                )

mat = mat[mat > 0]
for idx, e in enumerate(mesh.Elements3D()):
    mat_idx = mat[idx]
    if mat_idx == 0:
        raise RuntimeError("Something went wrong")
    e.index = mat_idx
for domain in domains:
    # one-indexed
    mesh.SetMaterial(domain, domain_names[domain])


mesh.Save(f"{meshout}.vol.gz")
ngsolvemesh = ngsolve.Mesh(mesh)
print(ngsolvemesh.GetBoundaries())
print(ngsolvemesh.GetMaterials())

bdn_dict = {}
mat_dict = {}
for idx, b in enumerate(ngsolvemesh.GetBoundaries()):
    bdn_dict[b] = idx
for idx, b in enumerate(ngsolvemesh.GetMaterials()):
    mat_dict[b] = idx + 1
bndcf = ngsolvemesh.BoundaryCF(bdn_dict)
matcf = ngsolvemesh.MaterialCF(mat_dict)
ngsolve.Draw(bndcf, ngsolvemesh, "bnd")
ngsolve.Draw(matcf, ngsolvemesh, "mat")

vtk = ngsolve.VTKOutput(
    ngsolvemesh,
    coefs=[matcf],
    names=["mat"],
    filename=f"vtkout_{meshout}",
    subdivision=1,
)
vtk.Do()
