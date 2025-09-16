import netgen.occ as occ

try:
    from ngsolve.utils import printmaster
except ImportError:
    try:
        from ngsolve.utils import printonce as printmaster
    except ImportError:
        print("Neither printmaster nor printonce could be imported.")
import numpy as np
from .geo_bennets import norm
from scipy.spatial.transform import Rotation as scipy_Rotation


def label_step_file(single_shell_geo) -> None:
    bbox = single_shell_geo.bounding_box
    highest_z = bbox[1].z
    for solid in single_shell_geo.solids:
        # if ecm (has highest z) -> continue
        if np.isclose(solid.bounding_box[1].z, highest_z):
            name_ecm(solid)
        else:
            solid.mat("cytoplasm")
            solid.bc("membrane")
    return


def get_volume_properties(single_shell_geo):
    volume = single_shell_geo.mass
    cellvolume = 0.0
    cellsurface = 0.0
    for solid in single_shell_geo.solids:
        if solid.name == "ecm":
            continue
        cellvolume += solid.mass
        for face in solid.faces:
            cellsurface += face.mass
    return volume, cellvolume, cellsurface


def get_ecm(bbox):
    # get ecm, it is the bbox of geo
    ecm = occ.Box(*bbox)
    name_ecm(ecm)
    return ecm


def name_ecm(ecm):
    # name the faces to later assign BCs
    ecm.faces.Max(occ.X).name = "front"
    ecm.faces.Min(occ.X).name = "back"
    ecm.faces.Max(occ.Y).name = "right"
    ecm.faces.Min(occ.Y).name = "left"
    ecm.faces.Max(occ.Z).name = "top"
    ecm.faces.Min(occ.Z).name = "bottom"
    ecm.mat("ecm")


def random_rotation(occ_solid):
    rotvec = scipy_Rotation.random().as_rotvec()
    unit = rotvec / norm(rotvec)
    AxisAngle = np.append(unit, norm(rotvec))
    CellAxis = occ.Axis(
        occ_solid.center, occ.gp_Vec(AxisAngle[0], AxisAngle[1], AxisAngle[2])
    )
    Angle = AxisAngle[3] * 180.0 / np.pi
    return occ_solid.Rotate(CellAxis, Angle)


def RotateCellsRandomly(single_shell_geo):
    bbox = single_shell_geo.bounding_box
    highest_z = bbox[1].z
    ecm = get_ecm(bbox)
    CellList = []
    # go through all cells
    for solid in single_shell_geo.solids:
        # if ecm (has highest z) -> continue
        if np.isclose(solid.bounding_box[1].z, highest_z):
            continue

        rotated_solid = random_rotation(solid)
        rotated_solid.mat("cytoplasm")
        rotated_solid.bc("membrane")
        CellList.append(rotated_solid)

    # Make compounds for each object type
    if len(CellList) > 1:
        # do not use Glue here because they do not intersect
        CellCompound = occ.Compound(CellList)
    else:
        CellCompound = CellList[0]
    compoundlist = [CellCompound, ecm]
    TissueCompound = occ.Glue(compoundlist)
    return TissueCompound


def AddShellsToGeo(
    single_shell_geo,
    with_PCM=False,
    pcm_scale=None,
    with_nucleus=False,
    nucleus_scale=0.8,
):
    """Add shells to single shell geo.

    Parameters
    ----------
    with_PCM: bool
        Should the PCM be added?
    with_nucleus: bool
        Should the nucleus be added?
    nucleus_scale: double
        Ratio between cell and nucleus

    """
    bbox = single_shell_geo.bounding_box
    highest_z = bbox[1].z
    ecm = get_ecm(bbox)
    CellList = []
    if with_PCM:
        printmaster("Generate PCM")
        PericellList = []
    if with_nucleus:
        printmaster("Add nuclei")
        NucleusList = []

    # go through all cells
    for solid in single_shell_geo.solids:
        # if ecm (has highest z) -> continue
        if np.isclose(solid.bounding_box[1].z, highest_z):
            continue

        solid.mat("cytoplasm")
        solid.bc("membrane")
        CellList.append(solid)

        if with_nucleus:
            Nucleus = solid.Scale(solid.center, nucleus_scale)
            Nucleus.mat("nucleoplasm")
            Nucleus.bc("nucleusmembrane")
            # add new nucleus to the chondron nucleus list
            NucleusList.append(Nucleus)

        if with_PCM:
            if pcm_scale is None:
                raise ValueError("Provide the scale of the PCM")
            Pericell = solid.Scale(solid.center, pcm_scale)
            Pericell.mat("pcm")
            Pericell.bc("wall")
            PericellList.append(Pericell)

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

    ExtracellularCompound = ecm

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

    return TissueCompound
