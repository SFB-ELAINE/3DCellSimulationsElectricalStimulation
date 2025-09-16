import os
import time
import sys
import traceback
from netgen.meshing import Mesh, Element2D, MeshingStep, meshsize
import ngsolve
import netgen.occ as occ
import numpy as np
from contextlib import redirect_stdout, redirect_stderr


def get_final_filename(meshname):
    # Extract the base name without the directory path
    base_name = os.path.basename(meshname)

    # Remove ".inr.vol" and append "_final" before the extension
    final_name = base_name.replace(".inr.vol", "")  # Remove intermediate part
    final_name = final_name.replace(".vol.gz", "_final.vol.gz")  # Append "_final"

    return final_name


def process_mesh(meshname, output_folder="processed_meshes", log_folder="logs"):
    # Create output and log folders if they don’t exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Generate unique log filename
    base_name = os.path.basename(meshname)
    log_filename = os.path.join(log_folder, f"{os.path.splitext(base_name)[0]}.log")
    final_name = get_final_filename(meshname)
    output_name = os.path.join(output_folder, final_name)

    # Redirect stdout and stderr to log file
    with open(log_filename, "w") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            start_time = time.time()
            print(f"Processing mesh: {meshname}")
            try:
                # Load and process the mesh as per your original code
                ngsolve.SetNumThreads(12)
                m1 = Mesh()
                m1.Load(meshname)
                m1.Scale(
                    5.0
                )  # to match the reality meshes as we downsampling the images
                bounding_box = m1.bounding_box
                start = bounding_box[0]
                end = bounding_box[1]
                midpoint = (
                    np.array([end[0] + start[0], end[1] + start[1], end[2] + start[2]])
                    / 2
                )

                length_bot = 500  # um
                theta = 55
                theta_rot = 90 - theta
                tan_theta = np.tan(theta * np.pi / 180)
                height = 260  # 235
                r_elec = 60
                vec_point = height / tan_theta
                length_top = length_bot - vec_point * 2
                bottom_face = (
                    occ.WorkPlane(occ.Axes((0, 0, 0), n=occ.Z, h=occ.X))
                    .Rectangle(length_bot, length_bot)
                    .Face()
                )
                box1 = occ.Prism((bottom_face), (vec_point, vec_point, height))
                box2 = occ.Prism((bottom_face), (-vec_point, -vec_point, height))
                cavity = box1 * box2
                # the volume of a truncated pyramid
                V = (height / 3) * (
                    length_top**2
                    + length_bot**2
                    + np.sqrt(length_top**2 * length_bot**2)
                )
                assert np.isclose(V, cavity.mass)
                e1 = (
                    occ.WorkPlane(occ.Axes((length_bot / 2, 0, 0), n=occ.Y, h=occ.X))
                    .Circle(r_elec)
                    .Face()
                )
                e1 = e1.Rotate(occ.Axis((0, 0, 0), -occ.X), theta_rot) * cavity
                e1.bc("electrode_1")
                e2 = (
                    occ.WorkPlane(
                        occ.Axes((length_bot / 2, length_bot, 0), n=occ.Y, h=occ.X)
                    )
                    .Circle(r_elec)
                    .Face()
                )
                e2 = (
                    e2.Rotate(occ.Axis((0, length_bot, 0), -occ.X), -theta_rot) * cavity
                )
                e2.bc("electrode_2")
                e3 = (
                    occ.WorkPlane(occ.Axes((0, length_bot / 2, 0), n=occ.X, h=occ.Y))
                    .Circle(r_elec)
                    .Face()
                )
                e3 = (
                    e3.Rotate(occ.Axis((0, length_bot, 0), -occ.Y), -theta_rot) * cavity
                )
                e3.bc("electrode_3")
                e4 = (
                    occ.WorkPlane(
                        occ.Axes((length_bot, length_bot / 2, 0), n=occ.X, h=occ.Y)
                    )
                    .Circle(r_elec)
                    .Face()
                )
                e4 = e4.Rotate(occ.Axis((length_bot, 0, 0), -occ.Y), theta_rot) * cavity
                e4.bc("electrode_4")
                cavity_geo = occ.Glue([cavity, e1, e2, e3, e4])

                cavity_geo = cavity_geo.Move(
                    (
                        (midpoint[0] - length_bot / 2),
                        (midpoint[1] - length_bot / 2),
                        (start[2] - 10),
                    )
                )
                cavity_geo.mat("ecm")

                maxh_e = 5.0
                geo = occ.OCCGeometry(cavity_geo)
                # the index was found by visual inspection after loading the geometry
                # attention: the index here is zero-based while the print out by Netgen is not!
                # i.e.: if clicking on the surface in netgen, it will tell for example
                # "Selected Surface: 11". Then enter 10 here!
                geo.SetFaceMeshsize(4, maxh_e)
                geo.SetFaceMeshsize(1, maxh_e)
                geo.SetFaceMeshsize(7, maxh_e)
                geo.SetFaceMeshsize(9, maxh_e)

                # maxh = 20.0  # 50 core dump
                with ngsolve.TaskManager():
                    # m2 = geo.GenerateMesh(maxh=maxh, perfstepsend= MeshingStep.MESHSURFACE)
                    m2 = geo.GenerateMesh(
                        meshsize.very_fine, perfstepsend=MeshingStep.MESHSURFACE
                    )

                print("****************************")
                print("** merging surface meshes **")
                print("****************************")

                # a face-descriptor stores properties associated with a set of surface elements
                # bc .. boundary condition marker,
                # domin/domout .. domain-number in front/back of surface elements (0 = void),
                # surfnr .. number of the surface described by the face-descriptor

                # empty mesh
                mesh = Mesh()

                fds = m1.FaceDescriptors()
                fd_idxs = []
                max_surface_m1 = 0
                for fd in fds:
                    # Update domains, ecm is 1, outside is 0
                    fd.domin += 1
                    fd.domout += 1
                    # print(fd)
                    fd_idx = mesh.Add(fd)
                    fd_idxs.append(fd_idx)
                    mesh.SetBCName(fd_idx - 1, fd.bcname)

                    if fd.surfnr > max_surface_m1:
                        max_surface_m1 = fd.surfnr

                index_map_mesh2 = {}
                bnd_name_to_index_dict = {}
                fds_2 = m2.FaceDescriptors()
                current_index = 1
                for fd in fds_2:
                    index_map_mesh2[fd.surfnr] = fd.bcname
                    if fd.bcname in mesh.GetRegionNames(dim=2):
                        continue
                    fd.surfnr = current_index + max_surface_m1
                    fd.bc = current_index + max_surface_m1
                    fd_idx = mesh.Add(fd)
                    bnd_name_to_index_dict[fd.bcname] = fd_idx
                    mesh.SetBCName(fd_idx - 1, fd.bcname)
                    current_index += 1
                    # print(fd)

                domain_names = m1.GetRegionNames(dim=3)
                domain_names = ["ecm"] + domain_names
                # print("Domains: ", domain_names)

                # copy all boundary points from first mesh to new mesh.
                # pmap1 maps point-numbers from old to new mesh
                pmap1 = {}
                for e in m1.Elements2D():
                    for v in e.vertices:
                        if v not in pmap1:
                            pmap1[v] = mesh.Add(m1[v])

                pmap2 = {}
                for e in m2.Elements2D():
                    for v in e.vertices:
                        if v not in pmap2:
                            pmap2[v] = mesh.Add(m2[v])

                # copy surface elements from first mesh to new mesh
                # we have to map point-numbers:
                for e in m1.Elements2D():
                    fd = fd_idxs[e.index - 1]
                    mesh.Add(Element2D(fd, [pmap1[v] for v in e.vertices]))

                for e in m2.Elements2D():
                    bcname = index_map_mesh2[e.index]
                    fd = bnd_name_to_index_dict[bcname]
                    # shift index
                    e.index = fd
                    mesh.Add(Element2D(fd, [pmap2[v] for v in e.vertices]))

                # for fd in mesh.FaceDescriptors():
                #    print(fd)
                mesh.Save("surfaceonly.mesh.vol.gz")
                print("******************")
                print("** merging done **")
                print("******************")

                with ngsolve.TaskManager():
                    mesh.GenerateVolumeMesh(meshsize.coarse)

                if mesh.ne == 0:
                    print("WARNING: 3D meshing did not work")

                for idx, domain_name in enumerate(domain_names, start=1):
                    # one-indexed
                    mesh.SetMaterial(idx, domain_name)

                print("Final domains: ", mesh.GetNDomains())
                # print("Names: ", mesh.GetRegionNames(dim=3))
                print("Done")

                # Save the final mesh to the output file
                mesh.Save(output_name)
                print(f"Finished processing {meshname} -> saved as {output_name}")
                ngsolvemesh = ngsolve.Mesh(mesh)
                # print(ngsolvemesh.GetBoundaries())
                print(ngsolvemesh.GetMaterials())
            except Exception as e:
                # Catch and log any exceptions
                print(f"Error processing {meshname}: {e}")
                traceback.print_exc()

            print(f"Completed {meshname} in {time.time() - start_time:.2f} seconds")
    pass


if __name__ == "__main__":
    mesh_file = sys.argv[1]
    output_folder = sys.argv[2]
    log_folder = sys.argv[3]
    process_mesh(mesh_file, output_folder, log_folder)
