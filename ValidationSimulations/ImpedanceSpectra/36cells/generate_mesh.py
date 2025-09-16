import os
from cell_simulation_utilities.geo_bennets import BuildCartilageGeometry, GenerateMesh
import ngsolve
import netgen.occ as occ
from netgen.meshing import meshsize
import sys

ngsolve.SetNumThreads(12)

if len(sys.argv) != 2:
    raise RuntimeError(
        "Provide only one argument!"
        "Either SingleShell, SingleShellWall,"
        " DoubleShell, DoubleShellWall"
    )

if sys.argv[1] == "SingleShell":
    with_PCM = False
    with_nucleus = False
    meshname = "singleshell"
elif sys.argv[1] == "SingleShellWall":
    with_PCM = True
    with_nucleus = False
    meshname = "singleshellwall"
elif sys.argv[1] == "DoubleShell":
    with_PCM = False
    with_nucleus = True
    meshname = "doubleshell"
elif sys.argv[1] == "DoubleShellWall":
    with_PCM = True
    with_nucleus = True
    meshname = "doubleshellwall"
else:
    raise RuntimeError("The model is not known!")
meshname += "mesh.vol.gz"

geo, cellvolume, volume, cellsurface = BuildCartilageGeometry(
    "model_input.txt", with_PCM=with_PCM, with_nucleus=with_nucleus, nucleus_scale=0.8
)
print("Cellvolume: ", cellvolume)
print("Volume: ", volume)
print("Volume ratio: ", cellvolume / volume)
print("Cell surface: ", cellsurface)

occgeo = occ.OCCGeometry(geo)
ngmesh = GenerateMesh(
    occgeo,
    meshsize=meshsize.fine,
)

# check if folder exists
if not os.path.isdir(sys.argv[1]):
    os.mkdir(sys.argv[1])
ngmesh.Save(os.path.join(sys.argv[1], meshname))
