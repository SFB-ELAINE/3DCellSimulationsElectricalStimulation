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

// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
// commented, only used if no features are computed (see below [Extracting features to improve mesh])
// typedef CGAL::Labeled_mesh_domain_3<K> Mesh_domain;
typedef CGAL::Labeled_mesh_domain_3<K> Image_domain;
typedef CGAL::Mesh_domain_with_polyline_features_3<Image_domain> Mesh_domain;

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default, Concurrency_tag>::type Tr;

typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

// Triangulation for Remeshing
typedef CGAL::Triangulation_3<typename Tr::Geom_traits,
  typename Tr::Triangulation_data_structure> Triangulation_3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

namespace params = CGAL::parameters;

int main(int argc, char* argv[])
{

  // Limit number of cores to 12
  oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 12);

  /// [Loads image]
  if (argc == 1){
	  throw std::invalid_argument("Provide the name of a file in .inr.gz format.");
  }
  const std::string fname = argv[1];
  double sizing_scale = (argc>2) ? std::atof(argv[2]) : 0.1;
  // use a very large default
  double edge_size = (argc>3) ? std::atof(argv[3]) : 1000.0;
  CGAL::Image_3 image;
  if(!image.read(fname)){
    std::cerr << "Error: Cannot read file " <<  fname << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Image loaded" << std::endl;
  /// [Loads image]

   /// [Domain creation]
   // Is the standard approach
   // Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(image);
   /// [Domain creation]
 
   /// [Mesh using weights]
   // Using weights to smooth mesh: is ok but other method is better
   //const float sigma = (std::max)(image.vx(), (std::max)(image.vy(), image.vz()));
   //CGAL::Image_3 img_weights = CGAL::Mesh_3::generate_label_weights(image, sigma);
   //std::cout << "Weights created" << std::endl;
   //Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(image, params::weights(img_weights).relative_error_bound(1e-6));
   /// [Mesh using weights]
  
   /// [Extracting features to improve mesh]
   Mesh_domain domain = Mesh_domain::create_labeled_image_mesh_domain(image, params::features_detector = CGAL::Mesh_3::Detect_features_in_image());
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

  // C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, params::no_perturb().no_exude());
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, params::no_exude(), params::no_perturb());
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
