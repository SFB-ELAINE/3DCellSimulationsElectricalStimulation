from netgen.meshing import Mesh, Element2D
import ngsolve
import sys

ngsolve.ngsglobals.msg_level = 2
ngsolve.ngsglobals.testout = "mergengs.log"

if len(sys.argv) != 2:
    raise ValueError("Provide only meshname")
m1 = Mesh()
meshname = sys.argv[1]
m1.Load(meshname)

print("****************************")
print("** merging surface meshes **")
print("****************************")

# create an empty mesh
mesh = Mesh(dim=3)

# a face-descriptor stores properties associated with a set of surface elements
# bc .. boundary condition marker,
# domin/domout .. domain-number in front/back of surface elements (0 = void),
# surfnr .. number of the surface described by the face-descriptor

fds = m1.FaceDescriptors()
fd_idxs = []
for fd in m1.FaceDescriptors():
    print(fd)
    fd_idx = mesh.Add(fd)
    fd_idxs.append(fd_idx)
    mesh.SetBCName(fd_idx - 1, fd.bcname)

domain_names = m1.GetRegionNames(dim=3)
print("m1 Domains: ", m1.GetNDomains())
print("Domains: ", domain_names)

# copy all boundary points from first mesh to new mesh.
# pmap1 maps point-numbers from old to new mesh
pmap1 = {}
for e in m1.Elements2D():
    for v in e.vertices:
        if v not in pmap1:
            pmap1[v] = mesh.Add(m1[v])


# copy surface elements from first mesh to new mesh
# we have to map point-numbers:
for e in m1.Elements2D():
    fd = fd_idxs[e.index - 1]
    mesh.Add(Element2D(fd, [pmap1[v] for v in e.vertices]))


mesh.Save("surfaceonly.mesh.vol.gz")
print("******************")
print("** merging done **")
print("******************")

mesh.GenerateVolumeMesh(check_impossible=True)
if mesh.ne == 0:
    print("WARNING: 3D meshing did not work")

# problem: domains do not get constructed?!
for e in mesh.Elements3D():
    if e.index != 1:
        print(e.index)
print("m1 Domains: ", m1.GetNDomains())
print("Domains: ", domain_names)
for idx, domain_name in enumerate(domain_names, start=1):
    # one-indexed
    mesh.SetMaterial(idx, domain_name)


print("Final domains: ", mesh.GetNDomains())
print("Names: ", mesh.GetRegionNames(dim=3))
print("Done")

mesh.Save("newmesh.vol.gz")

ngsolvemesh = ngsolve.Mesh(mesh)
print(ngsolvemesh.GetBoundaries())
print(ngsolvemesh.GetMaterials())
