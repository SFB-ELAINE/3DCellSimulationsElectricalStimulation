// Find more information here: https://doc.cgal.org/latest/Mesh_3/index.html
// The code is based on these examples

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_3/generate_label_weights.h>
#include <CGAL/Mesh_3/Detect_features_in_image.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/IO/output_to_vtu.h>
#include <CGAL/facets_in_complex_3_to_triangle_mesh.h>
// for multithreading
#include <oneapi/tbb/global_control.h>
#include <filesystem>  // C++17 for directory handling
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <CGAL/Random.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Image_domain = CGAL::Labeled_mesh_domain_3<K>;
using Mesh_domain = CGAL::Mesh_domain_with_polyline_features_3<Image_domain>;


namespace fs = std::filesystem;
namespace params = CGAL::parameters;
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Image_domain = CGAL::Labeled_mesh_domain_3<K>;
using Mesh_domain = CGAL::Mesh_domain_with_polyline_features_3<Image_domain>;
 
#ifdef CGAL_CONCURRENT_MESH_3
using Concurrency_tag = CGAL::Parallel_tag;
#else
using Concurrency_tag = CGAL::Sequential_tag;
#endif
 
// Triangulation
using Tr = CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default,Concurrency_tag>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
 
// Criteria
using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
 
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;



int main(int argc, char* argv[])
{
  // Limit number of cores to 12
  oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 12);

  // Check for input arguments
  if (argc < 3) {
    throw std::invalid_argument("Usage: <program> <input_file.inr.gz> <output_folder> [sizing_scale] [edge_size]");
  }
  


  const std::string fname = argv[1];
  const std::string output_folder = argv[2];
  double sizing_scale = (argc > 3) ? std::atof(argv[3]) : 0.1;
  double edge_size = (argc > 4) ? std::atof(argv[4]) : 1000.0;
  
  printf("sizing_scale: %f\n", sizing_scale);
  printf("edge_size: %f\n", edge_size);
  // Load the image
  CGAL::Image_3 image;
  if(!image.read(fname)){
    std::cerr << "Error: Cannot read file " <<  fname << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Image loaded" << std::endl;

  const float sigma = (std::max)(image.vx(), (std::max)(image.vy(), image.vz()));
  CGAL::Image_3 img_weights = CGAL::Mesh_3::generate_label_weights(image, sigma);
  std::cout << "Label weights created" << std::endl;
  Mesh_domain domain
    = Mesh_domain::create_labeled_image_mesh_domain(image,
                                                    weights = std::ref(img_weights),
                                                    relative_error_bound = 1e-10,
                                                    features_detector = CGAL::Mesh_3::Detect_features_in_image());
  std::cout << "Domains created" << std::endl;
   
  // Ensure the output folder exists
  if (!fs::exists(output_folder)) {
    fs::create_directories(output_folder);
  }
  // Extract filename without path and extension
  std::string base_filename = fs::path(fname).stem().string();

  // Create the full output path
  std::string output_file = output_folder + "/" + base_filename + ".mesh";
  
  
  CGAL::Bbox_3 bbox = domain.bbox();
  double diag = CGAL::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                           CGAL::square(bbox.ymax() - bbox.ymin()) +
                           CGAL::square(bbox.zmax() - bbox.zmin()));
  double sizing_default = diag * sizing_scale;
  printf("diag: %f\n", diag);
  //params::edge_size = sizing_default,edge_size
  Mesh_criteria criteria(params::edge_size = sizing_default,
    params::edge_distance = sizing_default / 10,
    params::facet_angle = 25,
    params::facet_size = sizing_default,
    params::facet_distance = sizing_default / 10,
    //params::facet_topology = CGAL::FACET_VERTICES_ON_SAME_SURFACE_PATCH,
    params::cell_radius_edge_ratio = 0,
    params::cell_size = 0 );
    //
 


 
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
  std::cout << "Meshed" << std::endl;
  

  c3t3.remove_isolated_vertices();

  /// [Meshing]
  

  // Write the mesh to the output file
  std::ofstream medit_file(output_file);
  if (!medit_file) {
    std::cerr << "Error: Cannot write to output file " << output_file << std::endl;
    return EXIT_FAILURE;
  }
  CGAL::IO::write_MEDIT(medit_file, c3t3, params::all_cells(false).all_vertices(false).show_patches(false));
  medit_file.close();
  
  std::cout << "Mesh saved to " << output_file << std::endl;

  return 0;
}
