#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#include <CGAL/Image_3.h>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef CGAL_USE_VTK
#include <CGAL/IO/read_vtk_image_data.h>
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkNIFTIImageReader.h>
#include <vtkSmartPointer.h>
#include <vtkTIFFReader.h>
#endif

namespace fs = std::filesystem;

#ifdef CGAL_USE_VTK
inline bool load_via_vtk_reader(const std::string& fname,
                                vtkSmartPointer<vtkImageReader2> reader,
                                const char* label,
                                CGAL::Image_3& image)
{
  reader->SetFileName(fname.c_str());
  reader->Update();
  vtkSmartPointer<vtkImageData> vtk_image = reader->GetOutput();
  if (!vtk_image) {
    std::cerr << "Error: VTK " << label << " reader failed for " << fname << std::endl;
    return false;
  }
  image = CGAL::IO::read_vtk_image_data(vtk_image);
  if (image.image() == nullptr) {
    std::cerr << "Error: Could not convert VTK image to CGAL::Image_3" << std::endl;
    return false;
  }
  std::cout << "Image loaded (" << label << " via VTK): " << fname << std::endl;
  return true;
}
#endif

/// Load a 3D image from file.
/// Supported formats:
///   - INR (.inr, .inr.gz)          — always available (CGAL ImageIO)
///   - NIfTI (.nii, .nii.gz)        — requires VTK
///   - TIFF (.tif, .tiff)           — requires VTK; multi-page = 3D stack.
///                                    Voxel spacing is not recorded in TIFF
///                                    headers; sizing falls back to unit voxels.
///
/// Returns true on success.
inline bool load_image(const std::string& fname, CGAL::Image_3& image)
{
  fs::path path(fname);

  if (!fs::exists(path)) {
    std::cerr << "Error: File does not exist: " << fname << std::endl;
    return false;
  }

#ifdef CGAL_USE_VTK
  fs::path ext = path.extension();
  fs::path stem = path.stem();
  const bool is_nifti = (ext == ".nii")
                     || (ext == ".gz" && stem.extension() == ".nii");
  const bool is_tiff = (ext == ".tif") || (ext == ".tiff");

  if (is_nifti) {
    return load_via_vtk_reader(fname, vtkSmartPointer<vtkNIFTIImageReader>::New(),
                               "NIfTI", image);
  }
  if (is_tiff) {
    return load_via_vtk_reader(fname, vtkSmartPointer<vtkTIFFReader>::New(),
                               "TIFF", image);
  }
#endif

  // Default: INR via CGAL ImageIO
  if (!image.read(fname)) {
    std::cerr << "Error: Cannot read file " << fname << std::endl;
    return false;
  }
  std::cout << "Image loaded (INR): " << fname << std::endl;
  return true;
}

#endif // LOAD_IMAGE_H
