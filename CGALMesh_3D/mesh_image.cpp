#include "mesh_image.h"
#include "load_image.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_3/Detect_features_in_image.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
#include <oneapi/tbb/global_control.h>

#include <fstream>
#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Labeled_mesh_domain_3<K> Image_domain;
typedef CGAL::Mesh_domain_with_polyline_features_3<Image_domain> Mesh_domain;

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

namespace params = CGAL::parameters;

bool mesh_image(const std::string& input_path,
                const std::string& output_path,
                double sizing_scale,
                double edge_size)
{
  oneapi::tbb::global_control global_limit(
    oneapi::tbb::global_control::max_allowed_parallelism, 12);

  CGAL::Image_3 image;
  if (!load_image(input_path, image)) {
    return false;
  }

  Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(
    image, params::features_detector = CGAL::Mesh_3::Detect_features_in_image());
  std::cout << "Domains created" << std::endl;

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

  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria,
                                       params::no_exude(), params::no_perturb());
  std::cout << "Meshed" << std::endl;

  std::ofstream medit_file(output_path);
  if (!medit_file) {
    std::cerr << "Error: cannot open output file " << output_path << std::endl;
    return false;
  }
  CGAL::IO::write_MEDIT(medit_file, c3t3,
    params::all_cells(false).all_vertices(false).show_patches(false));
  return true;
}
