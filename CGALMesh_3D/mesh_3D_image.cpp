#include "mesh_image.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
  if (argc == 1) {
    throw std::invalid_argument(
      "Provide the name of an image file (.inr.gz or .nii/.nii.gz).");
  }
  const std::string input_path = argv[1];
  const double sizing_scale = (argc > 2) ? std::atof(argv[2]) : 0.1;
  const double edge_size = (argc > 3) ? std::atof(argv[3]) : 1000.0;

  return mesh_image(input_path, "out.mesh", sizing_scale, edge_size)
           ? EXIT_SUCCESS
           : EXIT_FAILURE;
}
