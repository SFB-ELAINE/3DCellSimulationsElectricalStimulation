#ifndef LOAD_IMAGE_H
#define LOAD_IMAGE_H

#include <CGAL/Image_3.h>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef CGAL_USE_VTK
#include <CGAL/IO/read_vtk_image_data.h>
#include <vtkImageData.h>
#include <vtkNIFTIImageReader.h>
#include <vtkSmartPointer.h>
#endif

namespace fs = std::filesystem;

/// Load a 3D image from file.
/// Supported formats:
///   - INR (.inr, .inr.gz)          — always available (CGAL ImageIO)
///   - NIfTI (.nii, .nii.gz)        — requires VTK
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
  // Check for NIfTI extension: .nii or .nii.gz
  fs::path ext = path.extension();
  fs::path stem = path.stem();
  bool is_nifti = (ext == ".nii")
               || (ext == ".gz" && stem.extension() == ".nii");

  if (is_nifti) {
    auto reader = vtkSmartPointer<vtkNIFTIImageReader>::New();
    reader->SetFileName(fname.c_str());
    reader->Update();
    vtkSmartPointer<vtkImageData> vtk_image = reader->GetOutput();
    if (!vtk_image) {
      std::cerr << "Error: VTK NIfTI reader failed for " << fname << std::endl;
      return false;
    }
    image = CGAL::IO::read_vtk_image_data(vtk_image);
    if (image.image() == nullptr) {
      std::cerr << "Error: Could not convert VTK image to CGAL::Image_3" << std::endl;
      return false;
    }
    std::cout << "Image loaded (NIfTI via VTK): " << fname << std::endl;
    return true;
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
