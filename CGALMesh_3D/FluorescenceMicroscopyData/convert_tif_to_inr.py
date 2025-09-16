import os
from skimage.io import imread
from timagetk import SpatialImage
from timagetk.io.image import imsave
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

# Define the argument parser
parser = argparse.ArgumentParser(
    prog="TIF-to-INR",
    description="Convert all TIFF images in a folder to INR format",
    epilog="Please report bugs and errors on GitHub",
)
parser.add_argument(
    "input_folder", type=str, help="path to the folder containing TIFF files"
)
parser.add_argument(
    "output_folder", type=str, help="path to the folder where INR files will be saved"
)
parser.add_argument(
    "voxelsize", type=float, nargs=3, help="voxel sizes ordered by [Z, Y, X]"
)

args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
voxelsize = args.voxelsize

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


def convert_to_inr(file_name):
    """Convert a single TIFF file to INR format."""
    # Full path to the input file
    input_file_path = os.path.join(input_folder, file_name)

    # Read and convert the image
    image = imread(input_file_path)
    image = image.astype(np.uint16)
    sp_img = SpatialImage(image, voxelsize=voxelsize)

    # Construct output file path
    output_file_name = file_name.replace(".tif", ".inr.gz").replace(".tiff", ".inr.gz")
    output_file_path = os.path.join(output_folder, output_file_name)

    # Save the INR file
    imsave(output_file_path, sp_img)
    print(f"Converted {file_name} to {output_file_name}")


# Get a list of all TIFF files in the input folder
tif_files = [
    f for f in os.listdir(input_folder) if f.lower().endswith((".tif", ".tiff"))
]

# Use multiprocessing to convert files in parallel
if __name__ == "__main__":
    # Set the number of processes to the number of available CPU cores
    num_processes = min(cpu_count(), len(tif_files))

    with Pool(processes=num_processes) as pool:
        pool.map(convert_to_inr, tif_files)

print(f"Conversion complete. INR files saved to {output_folder}")
