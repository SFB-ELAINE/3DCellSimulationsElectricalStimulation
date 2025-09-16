from numpy import array, sqrt, arccos, cross, dot, pi, isclose, append
import netgen.occ as occ
from ngsolve import TaskManager

try:
    from ngsolve.utils import printmaster
except ImportError:
    try:
        from ngsolve.utils import printonce as printmaster
    except ImportError:
        print("Neither printmaster nor printonce could be imported.")

from scipy.spatial.transform import Rotation as scipy_Rotation


def BuildCartilageGeometry(
    inputfile,
    with_PCM=False,
    with_nucleus=False,
    nucleus_scale=0.8,
    check_intersection=False,
    random_rotation=False,
):
    """Generates the solid tissue model geometry.

    Parameters
    ----------

    inputfile: str
        name of input file
    with_PCM: bool
        Should the PCM be added?
    with_nucleus: bool
        Should the nucleus be added?
    nucleus_scale: double
        Ratio between cell and nucleus

    Notes
    -----

    * pericellular regions in a chondron with multiple cells are assumed to be overlapping
    * cellular regions in a chondron with multiple cells are assumed NOT to be overlapping
    * the nucleus is built by scaling the cell

    FILE FORMAT: [#] = line number, N = number of cells in chondron
    ------------
    [ 1 ] sizeX sizeY sizeZ                                        # size of rectangular cartilage tissue region (um)
    [ 2 ] z_partition_1                                            # depth in z for ECM partition (um)
    [...]
    [p+ 1] z_partition_p                                            # p = # of partitions (can be zero, i.e. no partitions)
    [p+2] numPericellElements ECMtoPericellSeedSizeRatio        # number of elements along pericell thickness (defining cell/pericell seed size as minimum pericell thickness divided by this number), ratio of ECM seed size to cell/pericell seed size
    [...]
    [ i ] N                                                        # if number of cells in chondron > 1, each defined in the subsequent lines
    [i+ 1] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # c_ = cell center, rc_ = radius of cell, rp_ = radius of pericell, u_ = local x vector for ellipsoid, v_ = local vector to +y side for ellipsoid, in each dimension _ (um)
    [...]                                                        # NOTE: pericell regions in multiple cell chondron must overlap!
    [i+N] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # NOTE: local ellipsoid vectors do not need to be unit vectors
    [...]
    [ j ] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # if number of cells in chondron = 1, no preceding line specifying number of cells in chondron
    """
    print("This routine is based on the following paper, please cite it: ")
    print("Bennetts C. J., Sibole S., Erdemir A. (2014). ")
    print("Comput. Methods Biomechanics Biomed. Eng. 18, 1293–1304. ")
    print("DOI: 10.1080/10255842.2014.900545")

    # Load tissue geometry
    InputFile = open(inputfile, "r")
    lines = InputFile.readlines()
    InputFile.close()

    # track cellvolume and total volume
    cellvolume = 0.0
    cellsurface = 0.0
    volume = 0.0

    index = 0  # index into file line list

    volume_size = list(map(float, lines[index].split(" ")))
    index += 1

    partitions = []  # list to store depth of partition (depth percent as decimal, i.e. range 0.0-1.0, 0.0 = tidemark @ z = 0, 1.0 = articulating surface)

    # determine partitions if defined
    while len(lines[index].split(" ")) == 1:
        partitions.append(float(lines[index]))
        index += 1

    # sort multiple partitions if not listed in the input file in order
    if len(partitions) > 1:
        partitions.sort()

    index += 1

    # CREATE GEOMETRY

    # create rectangular tissue volume
    Volume = occ.Box(
        occ.Pnt(0, 0, 0), occ.Pnt(volume_size[0], volume_size[1], volume_size[2])
    )
    # name the faces to later assign BCs
    Volume.faces.Max(occ.X).name = "front"
    Volume.faces.Min(occ.X).name = "back"
    Volume.faces.Max(occ.Y).name = "right"
    Volume.faces.Min(occ.Y).name = "left"
    Volume.faces.Max(occ.Z).name = "top"
    Volume.faces.Min(occ.Z).name = "bottom"
    Volume.mat("ecm")
    volume = Volume.mass
    # CREATE CELL AND PERICELLULAR GEOMETRY

    CellList = []
    if with_PCM:
        printmaster("Generate PCM")
        PericellList = []
        min_thickness = 0.0
    if with_nucleus:
        printmaster("Add nuclei")
        NucleusList = []
    Origin = occ.Pnt(0.0, 0.0, 0.0)

    # loop over each of the remaining lines in the input file
    while index < len(lines):
        # only process lines if they are not empty
        if lines[index] != "\n" and lines[index] != "":
            line = list(map(float, lines[index].split(" ")))

            # check to see if building a multiple cell chondron
            n = 1
            if len(line) == 1:
                n = int(line[0])
                index += 1

            j = 0

            while j < n:
                if n > 1:
                    line = list(map(float, lines[index].split(" ")))

                c = line[0:3]  # cell center
                rc = line[3:6]  # radius of cell
                if with_PCM:
                    rp = line[6:9]  # radius of pericell

                # Construct orthogonal unit vectors for cell/pericell local coordinate system
                u = array(line[9:12])
                u = u / norm(u)
                v = array(line[12:15])
                v = v / norm(v)
                w = cross(u, v)
                w = w / norm(w)
                v = cross(w, u)
                v = v / norm(v)

                # Construct rotation matrix for local coordinate system
                if not random_rotation:
                    R = array(
                        [u, v, w]
                    )  # makes row vector array of orthogonal unit vectors
                    R = R.transpose()  # transposes to column vector array
                    # Determine axis and angle to reorient cell/pericell to specified local coordinate system
                    AxisAngle = rotMat2axisAngle(R)

                else:
                    rotvec = scipy_Rotation.random().as_rotvec()
                    unit = rotvec / norm(rotvec)
                    AxisAngle = append(unit, norm(rotvec))

                CellAxis = occ.Axis(
                    Origin, occ.gp_Vec(AxisAngle[0], AxisAngle[1], AxisAngle[2])
                )
                Angle = AxisAngle[3] * 180.0 / pi

                # CONSTRUCT CELL GEOMETRY
                # start with max sphere and reduce other radii if ellipsoidal (i.e. not spherical)
                # this reduces the tolerance less than starting with a unit sphere and scaling up
                rcmax = max(rc)
                # for cells use spheres
                if isclose(rc[0], rc[1]) and isclose(rc[1], rc[2]):
                    Cell = occ.Sphere(Origin, rcmax)
                else:
                    # Ellipsoid with first radius in x-direction
                    Cell = occ.Ellipsoid(
                        occ.Axis(p=(0, 0, 0), d=occ.X), rc[0], rc[1], rc[2]
                    )
                Cell.mat("cytoplasm")
                Cell.bc("membrane")

                # reorient cell ellipsoid to specified orientation
                # only apply rotation for non-zero angle and non-spherical objects
                if Angle != 0.0 and not (
                    isclose(rc[0], rc[1]) and isclose(rc[1], rc[2])
                ):
                    Cell = Cell.Rotate(CellAxis, Angle)
                # translate the cell to its specified position
                Cell = Cell.Move((c[0], c[1], c[2]))
                # add new cell to the chondron cell list
                CellList.append(Cell)

                if with_nucleus:
                    Nucleus = Cell.Scale(Cell.center, nucleus_scale)
                    Nucleus.mat("nucleoplasm")
                    Nucleus.bc("nucleusmembrane")
                    # add new nucleus to the chondron nucleus list
                    NucleusList.append(Nucleus)

                if with_PCM:
                    # CONSTRUCT PERICELL GEOMETRY
                    rpmax = max(rp)
                    Pericell = occ.Sphere(Origin, rpmax)
                    Pericell.mat("pcm")
                    Pericell.bc("wall")
                    if not (isclose(rp[0], rp[1]) and isclose(rp[1], rp[2])):
                        trafo = occ.gp_GTrsf(
                            mat=[
                                rp[0] / rpmax,
                                0,
                                0,
                                0,
                                rp[1] / rpmax,
                                0,
                                0,
                                0,
                                rp[2] / rpmax,
                            ]
                        )
                        Pericell = trafo(Pericell)

                    # reorient pericell ellipsoid to specified orientation
                    # only apply rotation for non-zero angle and non-spherical objects
                    if not isclose(Angle, 0.0) and not (
                        isclose(rp[0], rp[1]) and isclose(rp[1], rp[2])
                    ):
                        Pericell = Pericell.Rotate(CellAxis, Angle)
                    Pericell = Pericell.Move((c[0], c[1], c[2]))

                    PericellList.append(Pericell)

                    # update thinnest pericellular thickness
                    new_min_thickness = min(array(rp) - array(rc))
                    if isclose(min_thickness, 0):
                        min_thickness = new_min_thickness
                    elif new_min_thickness < min_thickness:
                        min_thickness = new_min_thickness

                j += 1
                index += 1

            # END OF CHONDRON GEOMETRY

            layer = 0
            for i in range(len(partitions)):
                if c[2] < partitions[i]:
                    break
                else:
                    layer += 1

        else:
            index += 1

    # END OF GEOMETRY

    # MAKE COMPOUNDS

    for c in CellList:
        cellvolume += c.mass
        for s in c.faces:
            cellsurface += s.mass

    if check_intersection:
        # last test for intersections
        for i, c1 in enumerate(CellList):
            for j, c2 in enumerate(CellList):
                # do not check too often
                if j <= i:
                    continue
                else:
                    dist = c1.Distance(c2)
                    if isclose(dist, 0.0):
                        raise RuntimeError(
                            "Cell {} and cell {} are intersecting!".format(i, j)
                        )
        # TODO
        # the following routine to deal with intersecting PCMs needs to be fixed.
        # A first solution was suggested by Julius but does not work sufficiently well.
        # index_to_be_removed = []
        # solid_to_add = []
        if with_PCM:
            for i, c1 in enumerate(PericellList):
                for j, c2 in enumerate(PericellList):
                    # do not check too often
                    if j <= i:
                        continue
                    else:
                        dist = c1.Distance(c2)
                        if isclose(dist, 0.0):
                            raise RuntimeError(
                                "Pericell {} and pericell {} are intersecting!".format(
                                    i, j
                                )
                            )
            """
                            print("Fixing this by uniting both solids.")
                            cnew = (c1 - CellList[i])
                            cnew = cnew + (c2 - CellList[j])
                            cnew.mat("pcm")
                            cnew.bc("wall")
                            print(type(cnew))
                            solid_to_add.append(cnew)
                            index_to_be_removed.append(i)
                            index_to_be_removed.append(j)
            if len(index_to_be_removed) > 0:
                print("Indices to be removed: ", index_to_be_removed)
                decrement = 0
                for idx in index_to_be_removed:
                     PericellList.pop(idx - decrement)
                     decrement += 1
                for s in solid_to_add:
                    PericellList.append(s)
            """

    # Make compounds for each object type
    if len(CellList) > 1:
        # do not use Glue here because they do not intersect
        CellCompound = occ.Compound(CellList)
    else:
        CellCompound = CellList[0]
    # Make compounds for each object type
    if with_PCM and len(PericellList) > 1:
        # do not use Glue here because they do not intersect
        PericellCompound = occ.Compound(PericellList)
    elif with_PCM and len(PericellList) == 1:
        PericellCompound = PericellList[0]

    if with_nucleus and len(NucleusList) > 1:
        # do not use Glue here because they do not intersect
        NucleusCompound = occ.Compound(NucleusList)
    elif with_nucleus and len(NucleusList) == 1:
        NucleusCompound = NucleusList[0]

    """
    if len(ECMSolidList) > 1:
        ExtracellularCompound = occ.Glue(ECMSolidList)
    else:
        ExtracellularCompound = ECMSolidList[0]
    """

    ExtracellularCompound = Volume

    # Make tissue compound containing all object type compounds
    if with_PCM and not with_nucleus:
        compoundlist = [CellCompound, PericellCompound, ExtracellularCompound]
    elif with_PCM and with_nucleus:
        compoundlist = [
            NucleusCompound,
            CellCompound,
            PericellCompound,
            ExtracellularCompound,
        ]
    elif with_nucleus and not with_PCM:
        compoundlist = [NucleusCompound, CellCompound, ExtracellularCompound]
    else:
        compoundlist = [CellCompound, ExtracellularCompound]
    TissueCompound = occ.Glue(compoundlist)

    return TissueCompound, cellvolume, volume, cellsurface


def GenerateMesh(
    occgeo,
    maxHECM=None,
    maxHPCM=None,
    maxHCell=None,
    maxHNucleus=None,
    maxHMembrane=None,
    maxHNMembrane=None,
    maxHWall=None,
    meshsize=None,
):
    """Generate NGSolve mesh for cartilage geo.

    Parameters
    ----------

    occgeo: netgen.occ.OCCGeometry
        Netgen OCCGeometry
    maxHECM: float
        Use a maxh for the ECM. The other kwargs are similar.
    meshsize: netgen.meshing.meshsize
        Use a pre-defined meshing hypothesis from netgen.

    Returns
    -------

    ngmesh: a Netgen mesh (needs to be converted to NGSolve!)
    """
    if (
        maxHECM is None
        and maxHPCM is None
        and maxHCell is None
        and maxHNucleus is None
        and maxHMembrane is None
        and maxHNMembrane is None
        and maxHWall is None
    ):
        with TaskManager():
            mesh = occgeo.GenerateMesh(mp=meshsize)
    else:
        for s in occgeo.shape.solids:
            if s.name == "ecm":
                if maxHECM is not None:
                    s.maxh = maxHECM
            elif s.name == "pcm":
                if maxHPCM is not None:
                    s.maxh = maxHPCM
                if maxHWall is not None:
                    for f in s.faces:
                        s.maxh = maxHWall
            elif s.name == "cytoplasm":
                if maxHCell is not None:
                    s.maxh = maxHCell
                if maxHMembrane is not None:
                    for f in s.faces:
                        s.maxh = maxHMembrane
            elif s.name == "nucleoplasm":
                if maxHNucleus is not None:
                    s.maxh = maxHNucleus
                if maxHNMembrane is not None:
                    for f in s.faces:
                        s.maxh = maxHNMembrane
            else:
                raise RuntimeError("Unnamed solid detected! Please check the geometry!")

            with TaskManager():
                mesh = occgeo.GenerateMesh(mp=meshsize)
    return mesh


def rotMat2axisAngle(R):
    """
    Converts a rotation matrix to an axis/angle representation
    adapted from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm
    INPUT: R, 3x3 array, rotation matrix
    OUTPUT: [x,y,z,angle]
    NOTE: [x,y,z] is a unit vector, [angle] in radians
    """

    epsilon = 0.01  # margin to allow for rounding errors
    epsilon2 = 0.1  # margin to distinguish between 0 and 180 degrees

    # singularity exists
    if (
        abs(R[0, 1] - R[1, 0]) < epsilon
        and abs(R[0, 2] - R[2, 0]) < epsilon
        and abs(R[1, 2] - R[2, 1]) < epsilon
    ):
        # check for identity matrix => arbitary axis, angle = 0.0
        if (
            abs(R[0, 1] + R[1, 0]) < epsilon2
            and abs(R[0, 2] + R[2, 0]) < epsilon2
            and abs(R[1, 2] + R[2, 1]) < epsilon2
            and abs(R[0, 0] + R[1, 1] + R[2, 2] - 3.0) < epsilon2
        ):
            return [1, 0, 0, 0]

        # otherwise singularity angle = 180 degress
        angle = pi
        xx = (R[0, 0] + 1) / 2
        yy = (R[1, 1] + 1) / 2
        zz = (R[2, 2] + 1) / 2
        xy = (R[0, 1] + R[1, 0]) / 4
        xz = (R[0, 2] + R[2, 0]) / 4
        yz = (R[1, 2] + R[2, 1]) / 4

        if xx > yy and xx > zz:  # R[0,0] is the largest diagonal term
            if xx < epsilon:
                x = 0.0
                y = 1.0 / sqrt(2.0)
                z = 1.0 / sqrt(2.0)
            else:
                x = sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:  # R[1,1] is the largest diagonal term
            if yy < epsilon:
                x = 1.0 / sqrt(2.0)
                y = 0.0
                z = 1.0 / sqrt(2.0)
            else:
                y = sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2,2] is the largest diagonal term
            if zz < epsilon:
                x = 1.0 / sqrt(2.0)
                y = 1.0 / sqrt(2.0)
                z = 0.0
            else:
                z = sqrt(zz)
                x = xz / z
                y = yz / z

        return [x, y, z, angle]

    # otherwise no singularity
    s = sqrt(
        (R[2, 1] - R[1, 2]) * (R[2, 1] - R[1, 2])
        + (R[0, 2] - R[2, 0]) * (R[0, 2] - R[2, 0])
        + (R[1, 0] - R[0, 1]) * (R[1, 0] - R[0, 1])
    )

    # prevent divide by zero
    # should not occur if matrix is orthogonal and caught by singularity test above
    if abs(s) < 0.001:
        s = 1.0

    angle = arccos((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0)
    x = (R[2, 1] - R[1, 2]) / s
    y = (R[0, 2] - R[2, 0]) / s
    z = (R[1, 0] - R[0, 1]) / s

    return [x, y, z, angle]


# END OF rotMat2axisAngle()


def norm(v):
    """
    Function to find 2-norm
    avoids using numpy.linalg library
    """
    return sqrt(dot(v, v))
