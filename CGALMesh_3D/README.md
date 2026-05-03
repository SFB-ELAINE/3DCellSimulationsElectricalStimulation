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
  --config-settings=cmake.define.CGAL_DIR=~/CGAL-6.1.1
```

If any of GMP, MPFR, ZLIB, Eigen3, Boost, TBB or VTK live in a non-system
prefix (the no-sudo recipe below installs all of them into `~/local`), add
`--config-settings=cmake.define.CMAKE_PREFIX_PATH=$HOME/local`.

## Build dependencies without sudo

If system packages are not available (or unavailable to you — different
distro, no sudo, restricted environments), the dependencies can all be
built into a user prefix. The recipe below has been used on Ubuntu 24.04
and is distro-agnostic; it should work on Arch, RHEL, macOS, or any Unix
with a C++17 compiler. All deps land in `~/local` (≈ 1 GB after build).

CGAL itself stays separate at e.g. `~/CGAL-6.1.1` (header-only) and is
passed via `CGAL_DIR`. The project's CMakeLists is compatible with both
CGAL 5.6.x and 6.1.x.

```
mkdir -p ~/local/src && cd ~/local/src

# GMP
curl -L -o gmp.tar.xz https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar xf gmp.tar.xz && (cd gmp-6.3.0 && \
  ./configure --prefix=$HOME/local --enable-cxx && make -j && make install)

# MPFR (needs GMP)
curl -L -o mpfr.tar.xz https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.xz
tar xf mpfr.tar.xz && (cd mpfr-4.2.1 && \
  ./configure --prefix=$HOME/local --with-gmp=$HOME/local && make -j && make install)

# zlib (needed by CGAL ImageIO for .inr.gz; transitively by VTK)
curl -L -o zlib.tar.gz https://zlib.net/zlib-1.3.1.tar.gz
tar xf zlib.tar.gz && (cd zlib-1.3.1 && \
  ./configure --prefix=$HOME/local && make -j && make install)

# Eigen3 — header-only
curl -L -o eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xf eigen.tar.gz
mkdir -p ~/local/include
cp -r eigen-3.4.0/Eigen ~/local/include/
cp -r eigen-3.4.0/unsupported ~/local/include/

# Boost — header-only is sufficient for CGAL Mesh_3 in this project
curl -L -o boost.tar.bz2 https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.bz2
tar xf boost.tar.bz2
cp -r boost_1_84_0/boost ~/local/include/

# TBB (oneTBB) — needed for CGAL_CONCURRENT_MESH_3
curl -L -o onetbb.tar.gz https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.13.0.tar.gz
tar xf onetbb.tar.gz && mkdir -p oneTBB-2021.13.0/build && (cd oneTBB-2021.13.0/build && \
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/local -DTBB_TEST=OFF .. && \
  make -j && make install)

# Minimal VTK (only IOImage + ImagingGeneral, no rendering / wrapping)
curl -L -o vtk.tar.gz https://www.vtk.org/files/release/9.3/VTK-9.3.1.tar.gz
tar xf vtk.tar.gz && mkdir VTK-9.3.1/build && (cd VTK-9.3.1/build && \
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
    -DVTK_MODULE_ENABLE_VTK_ImagingGeneral=YES .. && \
  ninja && ninja install)

# CGAL (header-only) — choose 5.6.x or 6.1.x
curl -L -o cgal.tar.xz \
  https://github.com/CGAL/cgal/releases/download/v6.1.1/CGAL-6.1.1-library.tar.xz
tar xf cgal.tar.xz -C ~/   # extracts to ~/CGAL-6.1.1
```

Distro notes:
- **Arch / gcc-15+**: CGAL 5.6.x can warn on newer gcc; if it fails, either
  upgrade to CGAL 6.1.1 or pass `-Wno-deprecated-declarations` via
  `--config-settings=cmake.define.CMAKE_CXX_FLAGS=-Wno-deprecated-declarations`.
- **macOS**: replace the `curl ... | tar` pattern as needed; `make -j` works.
  Apple's clang lacks `<filesystem>` on very old SDKs — use a recent Xcode.

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
