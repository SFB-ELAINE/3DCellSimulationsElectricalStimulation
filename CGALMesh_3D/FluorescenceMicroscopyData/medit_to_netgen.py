import os
import meshio
from netgen.meshing import Mesh, FaceDescriptor
import numpy as np
import sys


def name_surface(domain_in, domain_out):
    if domain_in == 0 and domain_out == 0:
        return "background"
    if domain_in == 0:
        return f"Cell_membrane_{domain_out}"
    elif domain_out == 0:
        return f"Cell_membrane_{domain_in}"
    elif domain_in > domain_out:
        # first domain always smaller than the other
        return f"Cell_interface_{domain_out}_{domain_in}"
    return f"Cell_interface_{domain_in}_{domain_out}"


# Check if correct number of arguments is passed
if len(sys.argv) != 3:
    raise ValueError("Usage: python3 medit_to_netgen.py <input_folder> <output_folder>")

input_folder = sys.argv[1]
output_folder = sys.argv[2]

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each .mesh file in the input folder
for filename in os.listdir(input_folder):
    if not filename.endswith(".mesh"):
        continue

    meshname = os.path.join(input_folder, filename)
    meshout = os.path.join(output_folder, filename.replace(".mesh", ".vol.gz"))

    # Read the .mesh file with meshio
    m = meshio.read(meshname)
    mesh = Mesh()
    mesh.AddPoints(m.points)

    # Read surface (triangles) and volume (tets) tags
    surface_labels = m.cell_data["medit:ref"][0]
    vol_labels = m.cell_data["medit:ref"][1]

    # Determine unique positive volume IDs and build a remapping to consecutive indices
    unique_vol = np.unique(vol_labels[vol_labels > 0])
    if 0 in unique_vol:
        raise RuntimeError(
            "Found a tetrahedron tagged with 0. Every tet must have a positive ID."
        )
    # Sort to maintain consistency
    unique_vol = np.sort(unique_vol)
    # Map original ID -> new consecutive ID (1..N)
    remap = {orig: new_id for new_id, orig in enumerate(unique_vol, start=1)}

    # Build face descriptors from the surface tags (uses original IDs)
    domin_domout_label = np.array([surface_labels[::2], surface_labels[1::2]])
    all_surfaces = np.unique(domin_domout_label, axis=1).T
    domin_domout_label = domin_domout_label.T
    # Wherever domain_in == domain_out, treat as boundary to "outside" (0)
    same_idx = np.argwhere(domin_domout_label[:, 0] == domin_domout_label[:, 1])
    domin_domout_label[same_idx, 1] = 0
    same_idx2 = np.argwhere(all_surfaces[:, 0] == all_surfaces[:, 1])
    all_surfaces[same_idx2, 1] = 0

    fd_idxs = []
    for idx, surface in enumerate(all_surfaces, start=1):
        orig_domin = surface[0]
        orig_domout = surface[1]
        # look up each original ID in remap; if not found (i.e. “outside”), use 0
        new_domin = remap.get(orig_domin, 0)
        new_domout = remap.get(orig_domout, 0)

        fd = FaceDescriptor(surfnr=idx, domin=new_domin, domout=new_domout, bc=idx)
        # keep the original pair only for naming
        fd.bcname = name_surface(orig_domin, orig_domout)
        fd_idx = mesh.Add(fd)
        fd_idxs.append(fd_idx)
        mesh.SetBCName(idx - 1, fd.bcname)

    fds = mesh.FaceDescriptors()

    # Insert 3D elements (tets) and 2D elements (triangles)
    # For tets, use the remapped index
    for cb in m.cells:
        if cb.dim == 3:
            # Only keep tets that have a positive tag
            mask3 = vol_labels > 0
            chosen_data = cb.data[mask3]
            # Assign elements with index = 1 temporarily; we'll set the correct remapped index below
            mesh.AddElements(dim=3, index=1, data=chosen_data, base=0)
        elif cb.dim == 2:
            triangles = cb.data[::2]
            for idx, surface in enumerate(all_surfaces, start=1):
                # find triangles whose (domain_in,domain_out) matches
                mask = np.logical_and.reduce(domin_domout_label == surface, axis=-1)
                chosen_triangles = triangles[mask]
                name = fds[idx - 1].bcname
                if "interface" in name:
                    mesh.AddElements(dim=2, index=idx, data=chosen_triangles, base=0)
                else:
                    # flip orientation for membranes
                    mesh.AddElements(
                        dim=2,
                        index=idx,
                        data=np.flip(chosen_triangles, 1),
                        base=0,
                    )

    # Now assign the correct remapped indices to every 3D element
    mat_positive = vol_labels[vol_labels > 0]
    all_3d_elems = list(mesh.Elements3D())
    if len(all_3d_elems) != len(mat_positive):
        raise RuntimeError(
            f"Mismatch: {len(all_3d_elems)} tets in Netgen mesh vs. {len(mat_positive)} positive tags."
        )
    for i, e in enumerate(all_3d_elems):
        orig_id = mat_positive[i]
        new_id = remap[orig_id]
        e.index = new_id

    # Assign a material name for each new remapped index, using the original ID in the name
    for orig_id, new_id in remap.items():
        mesh.SetMaterial(new_id, f"Cell_{orig_id}")

    # Double-check that "default" does not appear before saving
    all3d_names = mesh.GetRegionNames(dim=3)
    if "default" in all3d_names:
        raise RuntimeError(
            "Found an unnamed (default) region at save-time—"
            "some volume index was never assigned a material."
        )

    # Save the converted mesh
    mesh.Save(meshout)
    print(f"Processed and saved: {meshout}")
