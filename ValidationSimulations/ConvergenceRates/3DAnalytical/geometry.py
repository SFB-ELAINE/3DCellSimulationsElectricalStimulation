from netgen.csg import Plane, Vec, Pnt, CSGeometry, Sphere


def cell_in_box(h, R):
    left = Plane(Pnt(-h / 2.0, -h / 2.0, -h / 2.0), Vec(-1, 0, 0)).bc("cube")
    right = Plane(Pnt(h / 2.0, h / 2.0, h / 2.0), Vec(1, 0, 0)).bc("cube")
    front = Plane(Pnt(-h / 2.0, -h / 2.0, -h / 2.0), Vec(0, -1, 0)).bc("cube")
    back = Plane(Pnt(h / 2.0, h / 2.0, h / 2.0), Vec(0, 1, 0)).bc("cube")
    bot = Plane(Pnt(-h / 2.0, -h / 2.0, -h / 2.0), Vec(0, 0, -1)).bc("cube")
    top = Plane(Pnt(h / 2.0, h / 2.0, h / 2.0), Vec(0, 0, 1)).bc("cube")
    cube = left * right * front * back * bot * top

    sphere = Sphere(Pnt(0, 0, 0), R).bc("membrane")
    sphere.mat("cytoplasm")

    newcube = cube - sphere
    newcube.mat("ecm")

    # Geometry
    geo = CSGeometry()
    geo.Add(newcube)
    geo.Add(sphere)
    return geo
