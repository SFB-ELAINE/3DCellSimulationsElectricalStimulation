from netgen.meshing import Mesh, Element2D, Element3D

m1 = Mesh()
m1.Load("test_occ.vol.gz")

mesh = Mesh()
domain_names = m1.GetRegionNames(dim=3)
fds = m1.FaceDescriptors()
for fd in fds:
    fd_idx = mesh.Add(fd)
    mesh.SetBCName(fd_idx - 1, fd.bcname)

pmap1 = {}
for e in m1.Elements2D():
    for v in e.vertices:
        if v not in pmap1:
            pmap1[v] = mesh.Add(m1[v])

for e in m1.Elements2D():
    fd = e.index
    mesh.Add(Element2D(fd, [pmap1[v] for v in e.vertices]))


for e in m1.Elements3D():
    for v in e.vertices:
        if v not in pmap1:
            pmap1[v] = mesh.Add(m1[v])

for e in m1.Elements3D():
    mesh.Add(Element3D(e.index, [pmap1[v] for v in e.vertices]))

for idx, domain_name in enumerate(domain_names, start=1):
    # one-indexed
    mesh.SetMaterial(idx, domain_name)

mesh.Save("test")
