#!/usr/bin/python3

# --------------------------------------------------------------------------
# Copyright (c) 2013 Computational Biomechanics (CoBi) Core,
# Department of Biomedical Engineering, Cleveland Clinic
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------

# model_layout.py
#
# DESCRIPTION:
# ------------
# Generates explicit tissue model geometry from an implicitly-defined, statistical description
# (normal distributions) of cellular geometry and distribution.
#
# WRITTEN BY:
# -----------
#   Craig Bennetts
#   Computational Biomodeling (CoBi) Core
#   Department of Biomedical Engineering
#   Lerner Research Institute
#   Cleveland Clinic
#   Cleveland, Ohio
#   bennetc2@ccf.org
#
#
# EDITED BY:
# ----------
#
# Julius Zimmermann
#
# USAGE:
# ------
#   ./model_layout.py <layout_input.txt>
#
# INPUT:
# ------
#   layout_input.txt
#
#     FILE FORMAT: [#] = line number, n = depth region number [0,N-1], N = total number of depth regions
#    ------------
#    [ 1  ] sizeX sizeY sizeZ                                    # size of rectangular cartilage tissue region (um)
#   [ 2  ] min_depth max_depth                                    # bounds of current layer in Z dimension (um)
#   [ 3  ] meanChondonDensity stdChondronDensity                # units of chondron density = # chondron / mm^3
#    [ 4  ] meanCellsPerChondron stdCellsPerChondron                # mean and std dev of number of cells in chondron
#    [ 5  ] cellRadiusX cellRadiusY cellRadiusZ stdCellRadius    # mean cell radii in each dimension and std dev (um)
#    [ 6  ] meanPericellThickness stdPericellThickness            # mean and std dev of pericellular thickness (um)
#    ...
#    [5n+2] "same as [2]"
#    [5n+3] "same as [3]"
#    [5n+4] "same as [4]"
#    [5n+5] "same as [5]"
#    [5n+6] "same as [6]"
#
#    NOTES:
#    ------
#   chondron placement only checked in current region for intersections
#       => different layers must not overlap, or chondrons will potentially overlap
#   user responsible for ensuring:
#      0 <= min_depth < max_depth <= sizeZ
#   regions do NOT need to be listed in any particular order (i.e. increasing z)
#
# OUTPUT:
# -------
#   salome_model_input.txt
#
#    FILE FORMAT: [#] = line number, N = number of cells in chondron
#   ------------
#    [ 1 ] sizeX sizeY sizeZ                                        # size of rectangular cartilage tissue region (um)
#   [ 2 ] z_partition_1                                            # depth in z for ECM partition (um)
#   [...]
#   [p+1] z_partition_p                                            # p = # of partitions (can be zero, i.e. no partitions)
#   [p+2] numPericellElements ECMtoPericellSeedSizeRatio        # number of elements along pericell thickness (defining cell/pericell seed size as minimum pericell thickness divided by this number), ratio of ECM seed size to cell/pericell seed size
#    [...]
#    [ i ] N                                                        # if number of cells in chondron > 1, each defined in the subsequent lines
#    [i+1] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # c_ = cell center, rc_ = radius of cell, rp_ = radius of pericell, u_ = local x vector for ellipsoid, v_ = local vector to +y side for ellipsoid, in each dimension _ (um)
#    [...]                                                        # NOTE: pericell regions in multiple cell chondron must overlap!
#    [i+N] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # NOTE: local ellipsoid vectors do not need to be unit vectors
#   [...]
#   [ j ] cx cy cz rcx rcy rcz rpx rpy rpz ux uy uz vx vy vz    # if number of cells in chondron = 1, no preceding line specifying number of cells in chondron
#

import sys
from numpy import array, random, sort, greater
from numpy import all as npall
from numpy.linalg import norm


def model_layout(input_file: str) -> None:
    InputFile = open(input_file, "r")
    lines = InputFile.readlines()
    InputFile.close()

    volume_size = list(map(float, lines[0].split(" ")))

    # Determine partitions in z dimensions
    partitions = []

    for i in range(int(len(lines) / 6)):
        z_range = list(map(float, lines[i * 6 + 1].split(" ")))
        if z_range[0] != 0.0:
            partitions.append(z_range[0])
        if z_range[1] != volume_size[2]:
            partitions.append(z_range[1])

    # Sort partitions ranges and make the set a list of unique members
    if len(partitions) != 0:
        partitions = list(set(sort(partitions)))

    # Open Output File
    outfile_name = "salome_model_input.txt"
    outfile = open(outfile_name, "w")

    # Write size of volume (um)
    outfile.write("%f %f %f\n" % (volume_size[0], volume_size[1], volume_size[2]))

    # Write layer partition locations (um)
    for i in range(len(partitions)):
        outfile.write("%f\n" % partitions[i])

    # Write default number of elements through pericellular thickness (to determine cell/pericell seed size)
    # and ratio of ECM seed size to cell/pericell seed size
    outfile.write("3 12\n")

    # Set seed for random number generator
    random.seed(42)

    # initialize cell data lists
    CellPositionList = []
    CellRadiusList = []
    PericellRadiusList = []

    # initialize counters
    region_num = 0
    line_num = 1

    while line_num < len(lines):
        # set the number of attempts to randomly place a chondron
        ATTEMPT_LIMIT = 100000000
        ATTEMPT_LIMIT_CELLRADIUS = 10

        region_num += 1

        print()
        print("Processing Region %d:" % region_num)

        # determine cell and distribution parameters for current region
        # z_range[0] = min z region depth location (um)
        # z_range[1] = max z region depth location (um)
        z_range = list(map(float, lines[line_num].split(" ")))

        if len(z_range) != 2:
            print()
            print("ERROR: Invalid z_range at line %d" % line_num)
            print()
            sys.exit(1)

        # thickness of current region (um)
        dz = z_range[1] - z_range[0]

        print("    Depth = %f - %f um" % (z_range[0], z_range[1]))

        # Chondron density
        # density[0] = average chondron density (chondrons/mm^3)
        # density[1] = standard deviation of chondron density (chondrons/mm^3)
        # convert density to chondrons/um^3
        density = list(map(float, lines[line_num + 1].split(" ")))
        num_chondrons = int(
            round(
                (volume_size[0] * volume_size[1] * dz)
                * (density[0] + density[1] * random.standard_normal())
                / 1000**3
            )
        )

        print("    # chondrons = %d" % num_chondrons)

        # Number of chondrocytes per chondron
        # num_cells[0] = average # of cells per chondron
        # num_cells[1] = standard deviation of # of cells per chondron
        num_cells = list(map(float, lines[line_num + 2].split(" ")))

        # Cell size
        # cell_size[0] = average cell radius in x dimension (um)
        # cell_size[1] = average cell radius in y dimension (um)
        # cell_size[2] = average cell radius in z dimension (um)
        # cell_size[3] = standard deviation of cell radius (um)
        cell_size = list(map(float, lines[line_num + 3].split(" ")))

        # Pericell thickness
        # pericell_thickness[0] = average pericellular thickness (um)
        # pericell_thickness[1] = standard deviation of pericellular thickness (um)
        pericell_thickness = list(map(float, lines[line_num + 4].split(" ")))

        # Set the minimum spacing between pericell regions and to volume bounds
        pad = pericell_thickness[0]

        chondron_count = 0

        RegionCellPositionList = []
        RegionCellRadiusList = []
        RegionPericellRadiusList = []

        # generate each valid chondron
        while chondron_count < num_chondrons:
            valid_placement = False
            num_attempts = 0

            # attempt to place a spherical cell in the tissue volume
            while not valid_placement and num_attempts < ATTEMPT_LIMIT:
                # determine the # of cells for the current chondron
                n_cells = int(
                    round(num_cells[0] + num_cells[1] * random.standard_normal())
                )

                if n_cells < 1:
                    n_cells = 1

                ChondronCellPositionList = []
                ChondronCellRadiusList = []
                ChondronPericellRadiusList = []

                bounds = array([0.0, 0.0, 0.0])

                # stack cells vertically (in z dimension)
                for i in range(n_cells):
                    for _ in range(ATTEMPT_LIMIT_CELLRADIUS):
                        # randomly generate a spherical cell within the tissue volume
                        CellRadius = array(
                            [
                                cell_size[0] + cell_size[3] * random.standard_normal(),
                                cell_size[1] + cell_size[3] * random.standard_normal(),
                                cell_size[2] + cell_size[3] * random.standard_normal(),
                            ]
                        )

                        PericellRadius = array(
                            [
                                CellRadius[0]
                                + pericell_thickness[0]
                                + pericell_thickness[1] * random.standard_normal(),
                                CellRadius[1]
                                + pericell_thickness[0]
                                + pericell_thickness[1] * random.standard_normal(),
                                CellRadius[2]
                                + pericell_thickness[0]
                                + pericell_thickness[1] * random.standard_normal(),
                            ]
                        )
                        if npall(greater(CellRadius, 0)) and npall(
                            greater(PericellRadius, 0)
                        ):
                            break
                        else:
                            CellRadius = None
                            PericellRadius = None

                    if CellRadius is None and PericellRadius is None:
                        raise RuntimeError("Negative cell or pericell radius detected!")

                    if len(ChondronCellPositionList) == 0:
                        Position = array([0.0, 0.0, PericellRadius[2]])
                    else:
                        Position = (
                            ChondronCellPositionList[-1]
                            + array([0.0, 0.0, ChondronCellRadiusList[-1][2]])
                            + array([0.0, 0.0, PericellRadius[2]])
                        )

                    if bounds[0] < PericellRadius[0]:
                        bounds[0] = PericellRadius[0]
                    if bounds[1] < PericellRadius[1]:
                        bounds[1] = PericellRadius[1]

                    ChondronCellPositionList.append(Position)
                    ChondronCellRadiusList.append(CellRadius)
                    ChondronPericellRadiusList.append(PericellRadius)

                # z dimension of bounding box about chondron
                bounds[2] = (
                    ChondronCellPositionList[-1][2] + ChondronPericellRadiusList[-1][2]
                )

                # place chondron if smaller than thickness of current layer
                if (bounds[2] + 2 * pad) < dz:
                    offset = array(
                        [
                            random.uniform(
                                bounds[0] + pad, volume_size[0] - bounds[0] - pad
                            ),
                            random.uniform(
                                bounds[1] + pad, volume_size[1] - bounds[1] - pad
                            ),
                            random.uniform(
                                z_range[0] + pad, z_range[1] - bounds[2] - pad
                            ),
                        ]
                    )

                    for i in range(len(ChondronCellPositionList)):
                        ChondronCellPositionList[i] += offset

                    # check to see if new cell placement respects spacing with previously placed cells
                    valid_placement = True

                    for i in range(len(RegionCellPositionList)):
                        max_radius_old = max(RegionPericellRadiusList[i])

                        for j in range(len(ChondronCellPositionList)):
                            max_radius_new = max(ChondronPericellRadiusList[j])

                            dist = norm(
                                RegionCellPositionList[i] - ChondronCellPositionList[j]
                            )

                            if dist < max_radius_old + max_radius_new + pad:
                                valid_placement = False
                                break

                        if not valid_placement:
                            break

                else:
                    print(
                        "Warning! chondron size (%d cells, %f um) > current layer thickness (%f um)"
                        % (n_cells, bounds[2], dz - 2 * pad)
                    )

                num_attempts += 1

            if num_attempts == ATTEMPT_LIMIT:
                print()
                print(
                    "ERROR: Exceeded chondron placement attempt limit ("
                    + str(ATTEMPT_LIMIT)
                    + ")"
                )
                print(
                    "       Successfully placed "
                    + str(chondron_count)
                    + "/"
                    + str(num_chondrons)
                    + " chondrons"
                )
                print(
                    "       Density is too high or chondrons are too large for specified layer."
                )
                print()
                outfile.close()
                sys.exit(1)

            if n_cells > 1:
                outfile.write("%d\n" % n_cells)

            for i in range(len(ChondronCellPositionList)):
                outfile.write(
                    "%f %f %f "
                    % (
                        ChondronCellPositionList[i][0],
                        ChondronCellPositionList[i][1],
                        ChondronCellPositionList[i][2],
                    )
                )
                outfile.write(
                    "%f %f %f "
                    % (
                        ChondronCellRadiusList[i][0],
                        ChondronCellRadiusList[i][1],
                        ChondronCellRadiusList[i][2],
                    )
                )
                outfile.write(
                    "%f %f %f "
                    % (
                        ChondronPericellRadiusList[i][0],
                        ChondronPericellRadiusList[i][1],
                        ChondronPericellRadiusList[i][2],
                    )
                )
                outfile.write("1.0 0.0 0.0 ")
                outfile.write("0.0 1.0 0.0\n")

            # Add chondron geometry info to region info
            RegionCellPositionList += ChondronCellPositionList
            RegionCellRadiusList += ChondronCellRadiusList
            RegionPericellRadiusList += ChondronPericellRadiusList

            chondron_count += 1

            sys.stdout.write(
                "\r    Progress = " + str(chondron_count) + "/" + str(num_chondrons)
            )
            sys.stdout.flush()

        # END of region chondron placement

        # Append cell/pericell info for the current region to the entire set
        CellPositionList += RegionCellPositionList
        CellRadiusList += RegionCellRadiusList
        PericellRadiusList += RegionPericellRadiusList

        # advance the file input line index to the next region
        line_num += 5

        print()
        print()

    # END of cartilage cell placement

    # Close output file
    outfile.close()

    print()


# END OF model_layout()
