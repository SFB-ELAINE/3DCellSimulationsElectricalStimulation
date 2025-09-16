from skimage.io import imread
from timagetk import SpatialImage
from timagetk.io.image import imsave
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="TIF-to-INR",
    description="Convert TIFF images to INR",
    epilog="Please report bugs and errors on GitHub",
)
parser.add_argument("filename", type=str, help="filename of TIFF file")
parser.add_argument(
    "voxelsize", type=float, nargs=3, help="voxel sizes ordered by [Z, Y, X]"
)

args = parser.parse_args()


file_name = args.filename

image = imread(file_name)
image = image.astype(np.uint16)
sp_img = SpatialImage(image, voxelsize=args.voxelsize)

outfile = file_name.replace(".tif", "")
imsave(f"{outfile}.inr.gz", sp_img)
