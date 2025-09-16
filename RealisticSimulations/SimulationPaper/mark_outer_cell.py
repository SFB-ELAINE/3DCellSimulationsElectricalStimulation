import numpy as np
from scipy import ndimage
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage import color
import tifffile as tiff


def find_objects_touching_background(labeled_image, min_touch_pixels=5):
    """
    Given a 3D labeled image, returns:
      - An array of labels that have at least one voxel adjacent to a background (0) voxel,
        but only if the contact region has more than min_touch_pixels voxels.
      - The binary mask of the dilated background.

    Parameters:
      labeled_image : 3D numpy array
          The image where objects are labeled with non-zero integers.
      min_touch_pixels : int
          Minimum number of voxels that must be in contact with the background for an object to be counted.
    """
    # Create a binary mask for the background (voxels with label 0)
    background_mask = labeled_image == 0

    # Dilate the background mask using a 3x3x3 structuring element (26-neighborhood)
    structure = np.ones((3, 3, 3), dtype=bool)
    dilated_background = ndimage.binary_dilation(background_mask, structure=structure)

    # Find candidate labels where the dilated background is True (i.e. objects adjacent to background)
    candidate_labels = np.unique(labeled_image[dilated_background])
    candidate_labels = candidate_labels[
        candidate_labels != 0
    ]  # remove background label if present

    # Filter labels: only keep those with more than min_touch_pixels in contact with background.
    valid_labels = []
    for label in candidate_labels:
        # Count the number of voxels in the object that are also in the dilated background region.
        contact_count = np.sum((labeled_image == label) & dilated_background)
        if contact_count > min_touch_pixels:
            valid_labels.append(label)

    return np.array(valid_labels), dilated_background


def plot_dilated_background(labeled_image, dilated_background, slice_index=None):
    if slice_index is None:
        slice_index = (
            labeled_image.shape[0] // 2
        )  # use the central slice along the z-axis

    original_slice = labeled_image[slice_index, :, :]
    dilated_slice = dilated_background[slice_index, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original slice
    _ = axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title("Original Slice")
    plt.colorbar(_, ax=axes[0])

    # Dilated background mask slice
    _ = axes[1].imshow(dilated_slice, cmap="gray")
    axes[1].set_title("Dilated Background Mask")
    plt.colorbar(_, ax=axes[1])

    # Overlay of dilated background on original slice
    axes[2].imshow(original_slice, cmap="gray")
    # Mask the array: only display where dilated_slice is True
    overlay = np.ma.masked_where(~dilated_slice, dilated_slice)
    axes[2].imshow(overlay, cmap="autumn", alpha=0.5)
    axes[2].set_title("Overlay: Original & Dilated BG")

    plt.tight_layout()
    plt.show()


def plot_touching_and_non_touching(labeled_image, touching_labels, slice_index=None):
    """
    Plots a selected slice from the 3D image with two panels:
      - Left panel: Only the labels that touch the background.
      - Right panel: Only the labels that do NOT touch the background.
    """
    if slice_index is None:
        slice_index = labeled_image.shape[0] // 2  # Use central slice along z-axis

    original_slice = labeled_image[slice_index, :, :]

    # Create images for touching and non-touching objects
    touching_slice = np.where(
        np.isin(original_slice, touching_labels), original_slice, 0
    )
    non_touching_slice = np.where(
        (original_slice != 0) & (~np.isin(original_slice, touching_labels)),
        original_slice,
        0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    overlay = color.label2rgb(touching_slice, bg_label=0)
    _ = axes[0].imshow(overlay, cmap="nipy_spectral")
    axes[0].set_title("Touching Labels")
    # plt.colorbar(_, ax=axes[0])

    overlay = color.label2rgb(non_touching_slice, bg_label=0)
    _ = axes[1].imshow(overlay, cmap="nipy_spectral")
    axes[1].set_title("Non-Touching Labels")
    # plt.colorbar(_, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_dilated_background_zy(labeled_image, dilated_background, x_index=None):
    """
    Plots a selected zy plane (fixed x slice) from the 3D image showing:
      - The original slice.
      - The dilated background mask.
      - An overlay of the dilated background on the original slice.

    Parameters:
      labeled_image : 3D numpy array
          The labeled image.
      dilated_background : 3D boolean numpy array
          The binary mask obtained after dilation.
      x_index : int, optional
          The index along the x-axis to extract the zy plane.
          Defaults to the middle slice along the x-axis.
    """
    if x_index is None:
        x_index = labeled_image.shape[2] // 2  # Middle x index

    # Extract the zy plane (all z and y for fixed x)
    original_plane = labeled_image[:, :, x_index]
    dilated_plane = dilated_background[:, :, x_index]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original zy slice
    _ = axes[0].imshow(original_plane, cmap="gray")
    axes[0].set_title("Original zy Plane (x={})".format(x_index))
    plt.colorbar(_, ax=axes[0])

    # Dilated background mask in the zy plane
    _ = axes[1].imshow(dilated_plane, cmap="gray")
    axes[1].set_title("Dilated Background Mask (zy plane)")
    plt.colorbar(_, ax=axes[1])

    # Overlay: original with dilated background highlighted
    axes[2].imshow(original_plane, cmap="gray")
    overlay = np.ma.masked_where(~dilated_plane, dilated_plane)
    axes[2].imshow(overlay, cmap="autumn", alpha=0.5)
    axes[2].set_title("Overlay: Original & Dilated BG")

    plt.tight_layout()
    plt.show()


def plot_touching_and_non_touching_zy(labeled_image, touching_labels, x_index=None):
    """
    Plots a selected zy plane (fixed x slice) from the 3D image with two panels:
      - Left panel: Only the labels that touch the background.
      - Right panel: Only the labels that do NOT touch the background.

    Parameters:
      labeled_image : 3D numpy array
          The labeled image.
      touching_labels : array-like
          List or array of labels that are touching the background.
      x_index : int, optional
          The index along the x-axis to extract the zy plane.
          Defaults to the middle slice along the x-axis.
    """
    if x_index is None:
        x_index = labeled_image.shape[2] // 2  # Middle x index

    original_plane = labeled_image[:, :, x_index]

    # Create masks for touching and non-touching labels
    touching_plane = np.where(
        np.isin(original_plane, touching_labels), original_plane, 0
    )
    non_touching_plane = np.where(
        (original_plane != 0) & (~np.isin(original_plane, touching_labels)),
        original_plane,
        0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    overlay = color.label2rgb(touching_plane, bg_label=0)
    _ = axes[0].imshow(overlay, cmap="nipy_spectral")
    axes[0].set_title("Touching Labels (zy plane, x={})".format(x_index))
    # plt.colorbar(_, ax=axes[0])

    overlay = color.label2rgb(non_touching_plane, bg_label=0)
    _ = axes[1].imshow(overlay, cmap="nipy_spectral")
    axes[1].set_title("Non-Touching Labels (zy plane, x={})".format(x_index))
    # plt.colorbar(_, ax=axes[1])

    plt.tight_layout()
    plt.show()


def create_new_labeled_image(labeled_image, touching_labels):
    """
    Creates a new 3D image where:
      - Voxels belonging to touching objects are set to 1.
      - Voxels belonging to non-touching objects are set to 2.
      - Background voxels (0 in the original image) remain 0.

    Parameters:
      labeled_image : 3D numpy array
          The original labeled image.
      touching_labels : array-like
          List or array of labels that are touching the background.

    Returns:
      new_image : 3D numpy array
          The new labeled image.
    """
    new_image = np.zeros_like(labeled_image, dtype=np.uint8)

    # Mark touching objects as 1.
    mask_touching = np.isin(labeled_image, touching_labels)
    new_image[mask_touching] = 1

    # Mark non-touching objects (nonzero and not touching) as 2.
    mask_non_touching = (labeled_image != 0) & (
        ~np.isin(labeled_image, touching_labels)
    )
    new_image[mask_non_touching] = 2

    return new_image


image = skio.imread("path/to/your/.tif")
touching_labels, dilated_background = find_objects_touching_background(image)
new_labeled_image = create_new_labeled_image(image, touching_labels)
tiff.imwrite(
    "path/to/save/the/new/labeled/image.tif",
    new_labeled_image.astype(np.uint8),
    compression="zlib",
)
np.save("marked_outer.npy", touching_labels)

print("Labels touching the background (0):", touching_labels)
print("no. of touching cells: ", len(touching_labels))
print("no. of total cells: ", len(np.unique(image)))
plot_touching_and_non_touching(image, touching_labels)
plot_touching_and_non_touching_zy(image, touching_labels)
plot_touching_and_non_touching(image, touching_labels, slice_index=76)
plot_touching_and_non_touching_zy(image, touching_labels, x_index=76)
plot_touching_and_non_touching(image, touching_labels, slice_index=40)
plot_touching_and_non_touching_zy(image, touching_labels, x_index=40)
