if __name__ == "__main__":
    import os
    import skimage.io as skio
    import numpy as np
    import segment3D.parameters as uSegment3D_params
    import segment3D.filters as uSegment3D_filters
    import segment3D.usegment3d as uSegment3D
    import sys

    # =============================================================================
    # 1. Set up input and output folders
    # =============================================================================
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Debug: Print input and output folders
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    # =============================================================================
    # 2. Loop through all .tif files in the input folder and process them
    # =============================================================================
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing {input_path}")

            # Load image
            image = skio.imread(input_path)
            # Extract basename for saving
            basename = os.path.splitext(filename)[0]

            # Apply 2D filtering in each direction
            labels_xy = uSegment3D_filters.filter_2d_label_slices(
                image, bg_label=0, minsize=10
            )
            labels_xz = uSegment3D_filters.filter_2d_label_slices(
                image.transpose(1, 0, 2).copy(), bg_label=0, minsize=10
            )
            labels_yz = uSegment3D_filters.filter_2d_label_slices(
                image.transpose(2, 0, 1).copy(), bg_label=0, minsize=10
            )

            # =============================================================================
            # 3. Set up parameters for 2D-to-3D aggregation and segmentation
            # =============================================================================
            indirect_aggregation_params = (
                uSegment3D_params.get_2D_to_3D_aggregation_params()
            )

            # Adjust parameter settings (same as in your original code)
            indirect_aggregation_params["indirect_method"]["smooth_sigma"] = 1
            indirect_aggregation_params["combine_cell_probs"]["ksize"] = 1
            indirect_aggregation_params["combine_cell_probs"]["alpha"] = 0.5
            indirect_aggregation_params["combine_cell_probs"]["smooth_sigma"] = 1
            indirect_aggregation_params["indirect_method"]["dtform_method"] = (
                "cellpose_improve"
            )
            indirect_aggregation_params["gradient_descent"]["gradient_decay"] = 0.05
            indirect_aggregation_params["gradient_descent"]["n_iter"] = 250
            indirect_aggregation_params["gradient_descent"]["momenta"] = 0.98
            indirect_aggregation_params["combine_cell_probs"]["threshold_n_levels"] = 2
            indirect_aggregation_params["combine_cell_probs"]["threshold_level"] = -1
            indirect_aggregation_params["combine_cell_probs"]["min_prob_thresh"] = 0.1

            # =============================================================================
            # 4. Perform 2D-to-3D aggregation and segmentation
            # =============================================================================
            segmentation3D, (probability3D, gradients3D) = (
                uSegment3D.aggregate_2D_to_3D_segmentation_indirect_method(
                    segmentations=[labels_xy, labels_xz, labels_yz],
                    img_xy_shape=labels_xy.shape,
                    precomputed_binary=None,
                    params=indirect_aggregation_params,
                    savefolder=None,
                    basename=None,
                )
            )

            # segmentation3D = np.zeros(image.shape)
            # Save the segmented 3D output
            output_path = os.path.join(output_folder, f"{basename}.tif")
            skio.imsave(output_path, np.uint16(segmentation3D))
            print(f"Saved segmented 3D image to {output_path}")
