# Install dependencies

## Ubuntu 22

```
sudo apt install libgdcm-tools libvtkgdcm-cil libvtkgdcm-dev libvtkgdcm-java python3-vtkgdcm
sudo apt install libinsighttoolkit5-dev
sudo apt install libtbb-dev
sudo apt install libmpfr-dev libgmp-dev libboost-all-dev
sudo apt install libeigen3-dev
sudo apt install libvtk9-dev
```

# Arch Linux

Use `yay` and install

```
yay insight-toolkit tbb boost eigen gmp mpfr
```


# Install CGAL code

Install CGAL as a header-only library (follow the instructions on their website).
Usually, this means to download the source code of the latest release.
We used version 5.6.1.

Then, compile the local C++ code here using
(under the assumption that CGAL is in the /home
directory)
```
cmake -DCMAKE_BUILD_TYPE=Release -DCGAL_DIR=~/CGAL-5.6.1 .
make -j 4
```

# Run CGAL

To convert the image `test.inr.gz`,
run
```
./mesh_image_3d test.inr.gz

```

# Python module (`cgalmesh3d`)

A thin nanobind wrapper exposes the same meshing pipeline as `mesh_3D_image`
to Python. It supports `.inr/.inr.gz` and, when VTK is available, also
`.nii/.nii.gz` and `.tif/.tiff` (multi-page TIFFs read as 3D stacks).

## Install (in a virtual environment)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e . \
  --config-settings=cmake.define.CGAL_DIR=~/CGAL-5.6.1
```

If GMP, MPFR or VTK live in a non-system prefix, add
`--config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/local`.

## Build dependencies without sudo

If system packages are not available, the dependencies can be built into a
user prefix (here `~/local`):

```
# GMP
curl -L -o gmp.tar.xz https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar xf gmp.tar.xz && cd gmp-6.3.0
./configure --prefix=$HOME/local --enable-cxx && make -j && make install && cd ..

# MPFR
curl -L -o mpfr.tar.xz https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.xz
tar xf mpfr.tar.xz && cd mpfr-4.2.1
./configure --prefix=$HOME/local --with-gmp=$HOME/local && make -j && make install && cd ..

# Minimal VTK (only IOImage + ImagingGeneral, no rendering / wrapping)
curl -L -o vtk.tar.gz https://www.vtk.org/files/release/9.3/VTK-9.3.1.tar.gz
tar xf vtk.tar.gz && mkdir VTK-9.3.1/build && cd VTK-9.3.1/build
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$HOME/local \
  -DBUILD_SHARED_LIBS=ON \
  -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF \
  -DVTK_WRAP_PYTHON=OFF -DVTK_WRAP_JAVA=OFF \
  -DVTK_GROUP_ENABLE_Rendering=NO \
  -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
  -DVTK_GROUP_ENABLE_Imaging=DONT_WANT \
  -DVTK_GROUP_ENABLE_Web=DONT_WANT \
  -DVTK_GROUP_ENABLE_Views=DONT_WANT \
  -DVTK_GROUP_ENABLE_MPI=DONT_WANT \
  -DVTK_GROUP_ENABLE_Qt=DONT_WANT \
  -DVTK_MODULE_ENABLE_VTK_IOImage=YES \
  -DVTK_MODULE_ENABLE_VTK_ImagingGeneral=YES ..
ninja && ninja install
```

The VTK install path is embedded as rpath in the compiled module, so no
`LD_LIBRARY_PATH` is needed at runtime.

ITK is not required for the Python module — the `generate_label_weights`
path is not exposed. The other CLI variants (`*_without_features`,
`*_with_polyline_features`, `*_with_weight_and_features`) still need ITK and
will only build when it is available.

## Usage

```python
import cgalmesh3d
cgalmesh3d.mesh_image(
    "Sample02_024_segCell.tif",  # .tif/.tiff, .nii/.nii.gz, or .inr/.inr.gz
    "Sample02_024.mesh",         # MEDIT output, consumed by medit_to_netgen.py
    sizing_scale=0.1,
    edge_size=1000.0,
    # Optional voxel spacing override (per axis, in physical units).
    # Leave at 0 to use the file's recorded spacing. TIFF headers carry
    # no spacing, so confocal stacks typically need at least vz set.
    vx=0.0, vy=0.0, vz=0.0,
)
```

`sizing_scale` is applied isotropically to a single bbox-diagonal-derived
length. Anisotropic *element* sizing is not supported and is not a goal
here: imaging data in this project is acquired with anisotropic voxel
spacing (confocal/fluorescence stacks typically have a Z spacing several
times larger than XY), but the resulting mesh elements should be uniform
in physical space. Setting the correct per-axis spacing via `vx/vy/vz` is
the supported way to compensate — the bbox diagonal is then in true
physical units and the elements are isotropic in physical space.

For the existing `.tif` → `.inr.gz` → `.mesh` workflow (using
`convert_tif_to_inr.py` and `mesh_3D_image_with_weight_and_features`), see
[`FluorescenceMicroscopyData/README.md`](FluorescenceMicroscopyData/README.md).

# Convert images to .inr format

CGAL accepts only .inr files.
The best option to convert .tif
(or other files) to .inr files is using
timagetk.
It can be best installed using conda:

```
conda create -n titk -c mosaic -c morpheme -c conda-forge timagetk
```
Run the following command to use it
```
conda activate titk
```

The script `convert_tif_to_inr.py` shows exemplarily
how to convert the image data.

# Working with MEDIT mesh

We enable `show_patches=False`. This means that every triangle is written twice.
Once with the element inside, once with the element outside.
See more info (here)[https://doc.cgal.org/latest/SMDS_3/group__PkgSMDS3IOFunctions.html#ga507712bcd0ba0d717be00a6bdf9207ba].

Use `medit_to_netgen.py` to convert to Netgen mesh
