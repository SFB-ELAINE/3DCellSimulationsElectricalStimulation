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
