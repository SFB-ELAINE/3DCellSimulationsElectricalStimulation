import numpy as np
from skimage import io
import tifffile as tiff
import os
from scipy import ndimage
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
from multiprocessing import Pool
from tqdm import tqdm
import sys


def remove_small_masks_openning(masks, min_size=15):
    # Find slices of each labeled region in masks
    slices = find_objects(masks)
    new_masks = np.zeros_like(masks)
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            labeled_slice, num_features = label(msk)
            mask = np.zeros_like(labeled_slice)
            # Get properties of each connected component
            props = regionprops(labeled_slice)
            for prop in props:
                if prop.area >= min_size:
                    mask[labeled_slice == prop.label] = prop.label
            mask[mask > 0] = i + 1
            new_masks[slc] = np.where(mask > 0, mask, new_masks[slc])

    return new_masks


def process_single_image(args):
    input_path, output_path = args
    # Load the TIFF image as a numpy array
    image = io.imread(input_path)
    image_array = np.array(image)

    min_pixel_value = image_array.min()
    print(
        f"Processing {os.path.basename(input_path)} - Lowest pixel value: {min_pixel_value}"
    )

    # Get unique values (sorted) and create a mapping from old values to new labels
    unique_values = np.unique(image_array)

    # Map old unique values to new sequential labels starting from 0
    relabel_map = np.zeros(unique_values.max() + 1, dtype=int)
    # Map 256 to 0
    relabel_map[256] = 0

    # For the other unique values (excluding 256), relabel them starting from 1
    new_labels = np.arange(1, len(unique_values))
    other_values = unique_values[unique_values != 256]
    relabel_map[other_values] = new_labels

    # Apply the mapping to relabel the image in a single step
    relabelled_image = relabel_map[image_array]

    factor = 0.5
    relabelled_image = ndimage.zoom(
        relabelled_image, zoom=[factor, factor, factor], order=0, mode="nearest"
    )
    min_size = 50
    relabelled_image = remove_small_masks_openning(relabelled_image, min_size=min_size)

    binary_image = (relabelled_image > 0).astype(np.uint8)
    filled_image = ndimage.binary_fill_holes(binary_image)
    hole = filled_image - binary_image
    dilated_hole = ndimage.binary_dilation(hole).astype(hole.dtype)
    labeled_holes, num_holes = ndimage.label(dilated_hole)

    for hole_label in range(1, num_holes + 1):
        current_hole_mask = labeled_holes == hole_label
        labels_in_hole = relabelled_image[current_hole_mask > 0]
        labels_in_hole = labels_in_hole[labels_in_hole != 0]
        if labels_in_hole.size > 0:
            largest_touching_label = np.bincount(labels_in_hole).argmax()
            hole_labeled = np.where(
                current_hole_mask > 0, largest_touching_label, 0
            ).astype(np.uint16)
        else:
            hole_labeled = current_hole_mask
        relabelled_image = np.where(hole_labeled > 0, hole_labeled, relabelled_image)

    tiff.imwrite(output_path, relabelled_image.astype(np.uint16), compression="zlib")
    print(f"Processed image saved as compressed TIFF: {output_path}")


def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tasks = []

    # Prepare tasks for processing
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_relabel.tif"
        output_path = os.path.join(output_folder, output_filename)
        tasks.append((input_path, output_path))

    # Use multiprocessing to process images in parallel
    with Pool(processes=os.cpu_count()) as pool:
        # Use tqdm for progress tracking
        list(tqdm(pool.imap(process_single_image, tasks), total=len(tasks)))


input_folder = sys.argv[1]
output_folder = sys.argv[2]
process_images(input_folder, output_folder)
