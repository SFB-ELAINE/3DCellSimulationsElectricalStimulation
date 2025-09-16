import ngsolve
import sys

if len(sys.argv) != 2:
    raise ValueError("Provide only the filename to be shown.")
mesh_file = sys.argv[1]

mesh = ngsolve.Mesh(mesh_file)

domains = mesh.GetMaterials()
mat_dict = {}
for idx, domain in enumerate(domains):
    mat_dict[domain] = idx
print("Domains: ", mat_dict)
matcf = mesh.MaterialCF(mat_dict)

surfaces = mesh.GetBoundaries()
bnd_dict = {}
for idx, surface in enumerate(surfaces):
    if surface == "default":
        continue
    bnd_dict[surface] = idx
print("Boundaries: ", bnd_dict)
bndcf = mesh.BoundaryCF(bnd_dict, default=-1)

ngsolve.Draw(matcf, mesh, "mat")
ngsolve.Draw(bndcf, mesh, "bnd")
