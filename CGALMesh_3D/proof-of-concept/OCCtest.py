import netgen.occ as occ
import ngsolve

origin = occ.Pnt(0, 0, 0)
sphere1 = occ.Sphere(origin, r=1)
sphere2 = sphere1.Move((0.5, 0, 0))

sphere1.mat("Cell_1")
sphere2.mat("Cell_2")
sphere1.bc("Cell_membrane_1")
sphere2.bc("Cell_membrane_2")

sphere1cut = sphere1 - sphere2
glue = occ.Glue([sphere1cut, sphere2])
occgeo = occ.OCCGeometry(glue)
ngmesh = occgeo.GenerateMesh()
ngmesh.SetBCName(1, "Cell_interface_1_2")

for fd in ngmesh.FaceDescriptors():
    print(fd)

mesh = ngsolve.Mesh(ngmesh)
print(mesh.GetMaterials())
print(mesh.GetBoundaries())

matcf = mesh.MaterialCF({"Cell_1": 1, "Cell_2": 2})
ngsolve.Draw(matcf, mesh, "mat")
ngmesh.Save("test_occ.vol.gz")
