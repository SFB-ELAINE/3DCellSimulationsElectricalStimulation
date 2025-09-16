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
from skimage.segmentation import expand_labels
from skimage.morphology import (
    opening,
)


def remove_small_masks_openning(masks, min_size=15):
    # Find slices of each labeled region in masks
    slices = find_objects(masks)
    new_masks = np.zeros_like(masks)
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            msk_open = opening(msk)  # , ball(radius))
            labeled_slice, num_features = label(msk_open)
            mask = np.zeros_like(labeled_slice)
            # Get properties of each connected component
            props = regionprops(labeled_slice)
            for prop in props:
                if (
                    prop.area >= min_size
                ):  # and prop.eccentricity < 0.99: #eleongated detection
                    mask[labeled_slice == prop.label] = prop.label
            mask[mask > 0] = i + 1
            new_masks[slc] = np.where(mask > 0, mask, new_masks[slc])

    return new_masks


def process_single_image(args):
    input_path, output_path = args
    image = io.imread(input_path)
    image_array = np.array(image).astype(np.int16)

    min_size = 20
    relabelled_image = remove_small_masks_openning(image_array, min_size=min_size)
    expanded = expand_labels(
        relabelled_image, distance=2, spacing=[relabelled_image.shape[0], 1, 1]
    )  # 15

    for i in range(expanded.shape[0]):
        binary_image = (expanded[i] > 0).astype(np.uint8)
        filled_image = ndimage.binary_fill_holes(binary_image)
        hole = filled_image - binary_image
        dilated_hole = ndimage.binary_dilation(hole).astype(hole.dtype)
        labeled_holes, num_holes = ndimage.label(dilated_hole)
        print(f"Layer {i}: Found {num_holes} holes")

        for hole_label in range(1, num_holes + 1):
            current_hole_mask = labeled_holes == hole_label
            labels_in_hole = expanded[i][current_hole_mask > 0]
            labels_in_hole = labels_in_hole[labels_in_hole != 0]
            if labels_in_hole.size > 0:
                largest_touching_label = np.bincount(labels_in_hole).argmax()
                hole_labeled = np.where(
                    current_hole_mask > 0, largest_touching_label, 0
                ).astype(np.uint16)
            else:
                hole_labeled = current_hole_mask
            expanded[i] = np.where(hole_labeled > 0, hole_labeled, expanded[i])

    tiff.imwrite(output_path, expanded.astype(np.uint16), compression="zlib")
    print(f"Saved processed image to {output_path}")


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
