import ngsolve
from mpi4py.MPI import COMM_WORLD


def load_mesh(filename: str, comm: COMM_WORLD, curve_order: int = None):
    mesh = ngsolve.Mesh(filename, comm=comm)
    if curve_order is not None:
        mesh.Curve(curve_order)
    return mesh
