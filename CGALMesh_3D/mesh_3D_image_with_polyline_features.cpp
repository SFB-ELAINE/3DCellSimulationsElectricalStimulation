#include <vector>
#include <iostream>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_3/Detect_features_in_image.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/IO/output_to_vtu.h>
// #include <CGAL/facets_in_complex_3_to_triangle_mesh.h>
// for multithreading
#include <oneapi/tbb/global_control.h>

// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Labeled_mesh_domain_3<K> Image_domain;
typedef CGAL::Mesh_domain_with_polyline_features_3<Image_domain> Mesh_domain;

#include <CGAL/Mesh_3/Detect_features_on_image_bbox.h>

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default, Concurrency_tag>::type Tr;

typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

namespace params = CGAL::parameters;

// Read input features
#include "read_polylines.h"

int main(int argc, char* argv[])
{

  // Limit number of cores to 12
  oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 12);

  /// [Loads image]
  if (argc == 1){
	  throw std::invalid_argument("Provide the name of a file in .inr.gz format and the polylines file.");
  }
  const std::string fname = argv[1];
  const std::string lines_fname = argv[2];
  double sizing_scale = (argc>3) ? std::atof(argv[3]) : 0.1;
  // use a very large default
  double edge_size = (argc>4) ? std::atof(argv[4]) : 1000.0;
  CGAL::Image_3 image;
  if(!image.read(fname)){
    std::cerr << "Error: Cannot read file " <<  fname << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Image loaded" << std::endl;
  
  std::vector<std::vector<K::Point_3> > features_inside;
  if (!read_polylines(lines_fname, features_inside)) // see file "read_polylines.h"
  {
    std::cerr << "Error: Cannot read file " << lines_fname << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Polylines loaded" << std::endl;

   /// [Extracting features to improve mesh]
  //TODO: try CGAL::Mesh_3::Detect_features_on_image_bbox()
   Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(image, params::features_detector = CGAL::Mesh_3::Detect_features_in_image(), params::input_features  = std::cref(features_inside));
   /// [Extracting features to improve mesh]

   std::cout << "Domains created" << std::endl;
   
  /// [Meshing]
  // Mesh criteria
  // Tricks:
  // * setting an edge size helps to refine the mesh at intersections
  // More infos: https://doc.cgal.org/latest/Mesh_3/classCGAL_1_1Mesh__criteria__3.html
  //  Mesh_criteria criteria(params::edge_size(1.0).facet_angle(30).facet_size(4).facet_distance(1).cell_radius_edge_ratio(3).cell_size(6));

  CGAL::Bbox_3 bbox = domain.bbox();
  double diag = CGAL::sqrt(CGAL::square(bbox.xmax() - bbox.xmin()) +
                           CGAL::square(bbox.ymax() - bbox.ymin()) +
                           CGAL::square(bbox.zmax() - bbox.zmin()));
  double sizing_default = diag * sizing_scale;
  Mesh_criteria criteria(params::edge_size = sizing_default,
    params::facet_angle = 30,
    params::facet_size = sizing_default,
    params::facet_distance = sizing_default / 10,
    params::facet_topology = CGAL::FACET_VERTICES_ON_SAME_SURFACE_PATCH,
    params::cell_radius_edge_ratio = 0,
    params::cell_size = 0,
    params::edge_size = edge_size
  );

  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
  // CGAL::perturb_mesh_3(c3t3, domain, params::time_limit(15));
  //CGAL::lloyd_optimize_mesh_3(c3t3, domain, params::time_limit(30));
  //CGAL::exude_mesh_3(c3t3, params::sliver_bound(10), params::time_limit(15));

  /// [Meshing]
  std::cout << "Meshed" << std::endl;



  // Output
  // show_patches to get adjacent names
  // do not include cells outside complex
  // do not include vertices outside compls
  std::ofstream medit_file("out.mesh");
  CGAL::IO::write_MEDIT(medit_file, c3t3, params::all_cells(false).all_vertices(false).show_patches(false));
  medit_file.close();

  return 0;
}
